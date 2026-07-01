# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict
from typing import Any

from robot_rl.env import VecEnv
from robot_rl.extensions import RandomNetworkDistillation, Symmetry, resolve_rnd_config, resolve_symmetry_config
from robot_rl.models import FuseModel, MLPModel
from robot_rl.modules import TargetNetwork
from robot_rl.storage import ReplayBuffer
from robot_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer


class SAC:
    """Soft Actor-Critic (off-policy, entropy-regularized).

    Reuses the shared building blocks: a generic :class:`~robot_rl.models.MLPModel` actor with a
    :class:`~robot_rl.modules.SquashedTanhGaussianDistribution` output, twin :class:`~robot_rl.models.FuseModel`
    Q-critics (obs+action fusion) each wrapped in a :class:`~robot_rl.modules.TargetNetwork`, and the shared
    :class:`~robot_rl.storage.ReplayBuffer` in ``keep_terminal`` mode. The temperature ``alpha`` is optionally
    learned against a target entropy. Exposes the standard runner interface: :meth:`act`,
    :meth:`process_env_step`, :meth:`update`.
    """

    def __init__(
        self,
        actor: MLPModel,
        critic_1: FuseModel,
        critic_2: FuseModel,
        replay_buffer: ReplayBuffer,
        num_actions: int,
        replay_buffer_size: int = 1_000_000,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        mini_batch_size: int = 256,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        alpha_learning_rate: float = 1e-3,
        actor_optimizer: str = "adam",
        critic_optimizer: str = "adam",
        auto_alpha: bool = True,
        alpha: float = 0.05,
        tau: float = 0.005,
        gamma: float = 0.99,
        target_entropy_scale: float = 1.0,
        max_grad_norm: float = 1.0,
        policy_frequency: int = 1,
        n_steps: int = 1,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the SAC algorithm. See the module docstring for the design; args mirror the SAC config."""
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND (intrinsic reward) -- optional, shared extension (owns its own predictor optimizer).
        self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg) if rnd_cfg else None

        # Symmetry augmentation -- optional, shared extension.
        if symmetry_cfg is not None and (actor.is_recurrent or critic_1.is_recurrent):
            raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
        self.symmetry = Symmetry(**symmetry_cfg) if symmetry_cfg else None

        # Models
        self.actor = actor.to(device)
        self.critic_1 = critic_1.to(device)
        self.critic_2 = critic_2.to(device)
        self.critic_1_target = TargetNetwork(self.critic_1, tau).to(device)
        self.critic_2_target = TargetNetwork(self.critic_2, tau).to(device)

        # Replay buffer
        self.replay_buffer = replay_buffer
        self.replay_buffer_size = replay_buffer_size
        self.transition = ReplayBuffer.Transition()

        # Hyperparameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        self.alpha = alpha
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.alpha_learning_rate = alpha_learning_rate
        self.policy_frequency = policy_frequency
        self.n_steps = n_steps
        self.max_grad_norm = max_grad_norm
        self.num_actions = num_actions
        self.update_step = 0
        self.intrinsic_rewards: torch.Tensor | None = None

        self.target_entropy = -target_entropy_scale * num_actions

        # Temperature (log-space); learned against the target entropy when auto_alpha.
        self.log_alpha = torch.log(torch.tensor(self.alpha, device=self.device)).detach().clone()
        self.log_alpha.requires_grad_(auto_alpha)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_learning_rate) if auto_alpha else None

        # Optimizers over the trainable (online) parameters. Target params are frozen and excluded.
        self.actor_parameters = [p for p in self.actor.parameters() if p.requires_grad]
        self.critic_parameters = [
            p for p in chain(self.critic_1.parameters(), self.critic_2.parameters()) if p.requires_grad
        ]
        self.actor_optimizer = resolve_optimizer(actor_optimizer)(self.actor_parameters, lr=actor_learning_rate)
        self.critic_optimizer = resolve_optimizer(critic_optimizer)(self.critic_parameters, lr=critic_learning_rate)

    # -- rollout ---------------------------------------------------------------------------------------------

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample a stochastic action and record the transition's observation/action."""
        with torch.no_grad():
            action = self.actor(obs, stochastic_output=True)
        self.transition.observations = obs
        self.transition.actions = action
        return action

    def process_env_step(
        self, next_obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
    ) -> None:
        """Record a step and insert the transition into the replay buffer.

        Handles the off-policy timeout distinction: on a *timeout* the true (pre-reset) next observation comes
        from ``extras['time_outs_obs']`` and the bootstrap continues (``next_terminated=0``); on a true
        *termination* the bootstrap is masked out. Degrades gracefully (uses ``next_obs`` as-is) when the env
        does not provide ``time_outs``/``time_outs_obs``.
        """
        num_envs = dones.shape[0]
        dones_bool = dones.view(-1).bool().to(self.device)

        if "time_outs" in extras and extras["time_outs"] is not None:
            time_outs = extras["time_outs"].view(-1).bool().to(self.device)
            if extras.get("time_outs_obs") is not None:
                mask = time_outs[:, None]
                true_next_obs = TensorDict(
                    {
                        key: torch.where(mask, extras["time_outs_obs"][key].to(self.device), next_obs[key])
                        for key in next_obs.keys()
                    },
                    batch_size=next_obs.batch_size,
                )
            else:
                true_next_obs = next_obs
            next_terminated = dones_bool & ~time_outs
        else:
            true_next_obs = next_obs
            next_terminated = dones_bool

        # Update normalizers on the observed next states.
        self.actor.update_normalization(true_next_obs)
        self.critic_1.update_normalization(true_next_obs)
        self.critic_2.update_normalization(true_next_obs)
        if self.rnd:
            self.rnd.update_normalization(true_next_obs)

        rew = rewards.clone().to(self.device)
        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(true_next_obs)
            rew = rew + self.intrinsic_rewards

        self.transition.rewards = rew
        self.transition.next_observations = true_next_obs
        self.transition.dones = dones_bool
        self.transition.next_terminated = next_terminated.byte()
        # SAC carries no latent context; provide an empty (z_dim=0) tensor for the shared buffer.
        self.transition.context = torch.zeros(num_envs, 0, device=self.device)

        self.replay_buffer.add_transitions(self.transition)
        self.transition.clear()

    # -- learning --------------------------------------------------------------------------------------------

    def update(self) -> dict:
        """Run the off-policy SAC updates over sampled mini-batches; returns mean losses."""
        mean_critic_1_loss = 0.0
        mean_critic_2_loss = 0.0
        mean_actor_loss = 0.0
        mean_alpha_loss = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        num_actor_updates = 0

        n_updates = self.num_learning_epochs * self.num_mini_batches
        for _ in range(n_updates):
            batch = self.replay_buffer.sample_mini_batch(self.device)
            obs_b = batch.observations
            next_obs_b = batch.next_observations
            actions_b = batch.actions
            rewards_b = batch.rewards.view(-1)
            not_terminated = 1.0 - batch.next_terminated.view(-1).float()

            # 1) Critic update -- bootstrapped target with entropy and (n-step) discount.
            with torch.no_grad():
                next_actions, next_logp = self.actor.act_and_log_prob(next_obs_b)
                q1_t = self.critic_1_target(next_obs_b, next_actions).view(-1)
                q2_t = self.critic_2_target(next_obs_b, next_actions).view(-1)
                min_q_t = torch.min(q1_t, q2_t) - self.log_alpha.exp() * next_logp
                discount = self.gamma**self.n_steps
                target_q = rewards_b + discount * not_terminated * min_q_t

            q1 = self.critic_1(obs_b, actions_b).view(-1)
            q2 = self.critic_2(obs_b, actions_b).view(-1)
            critic_1_loss = nn.functional.mse_loss(q1, target_q)
            critic_2_loss = nn.functional.mse_loss(q2, target_q)
            critic_loss = critic_1_loss + critic_2_loss

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters(self.critic_parameters)
            nn.utils.clip_grad_norm_(self.critic_parameters, self.max_grad_norm)
            self.critic_optimizer.step()

            # 2) Alpha + 3) actor (delayed by policy_frequency).
            new_actions, logp = self.actor.act_and_log_prob(obs_b)

            if self.auto_alpha:
                alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                if self.is_multi_gpu and self.log_alpha.grad is not None:
                    torch.distributed.all_reduce(self.log_alpha.grad, op=torch.distributed.ReduceOp.SUM)
                    self.log_alpha.grad /= self.gpu_world_size
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
                mean_alpha_loss += alpha_loss.item()

            if self.update_step % self.policy_frequency == 0:
                for p in self.critic_parameters:
                    p.requires_grad_(False)
                q1_pi = self.critic_1(obs_b, new_actions).view(-1)
                q2_pi = self.critic_2(obs_b, new_actions).view(-1)
                actor_loss = (self.log_alpha.exp().detach() * logp - torch.min(q1_pi, q2_pi)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.is_multi_gpu:
                    self.reduce_parameters(self.actor_parameters)
                nn.utils.clip_grad_norm_(self.actor_parameters, self.max_grad_norm)
                self.actor_optimizer.step()
                for p in self.critic_parameters:
                    p.requires_grad_(True)
                mean_actor_loss += actor_loss.item()
                num_actor_updates += 1

            # 4) Soft-update the target critics.
            self.critic_1_target.update()
            self.critic_2_target.update()

            # RND predictor update (RND owns its optimizer).
            if self.rnd:
                rnd_loss = self.rnd.compute_loss(obs_b)
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()
                if self.is_multi_gpu:
                    self.reduce_parameters(list(self.rnd.predictor.parameters()))
                self.rnd.optimizer.step()
                mean_rnd_loss += rnd_loss.item()

            mean_critic_1_loss += critic_1_loss.item()
            mean_critic_2_loss += critic_2_loss.item()
            self.update_step += 1

        loss_dict = {
            "critic_1": mean_critic_1_loss / n_updates,
            "critic_2": mean_critic_2_loss / n_updates,
            "actor": mean_actor_loss / max(num_actor_updates, 1),
            "alpha": mean_alpha_loss / n_updates if self.auto_alpha else 0.0,
            "alpha_value": self.alpha,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss / n_updates
        return loss_dict

    # -- mode / persistence ----------------------------------------------------------------------------------

    def train_mode(self) -> None:
        """Set the actor and critics to training mode."""
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        if self.rnd:
            self.rnd.train()

    def eval_mode(self) -> None:
        """Set the actor and critics to evaluation mode."""
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        if self.rnd:
            self.rnd.eval()

    def get_policy(self) -> MLPModel:
        """Return the actor (policy) model."""
        return self.actor

    def save(self) -> dict:
        """Return a dict of model/optimizer/temperature states for checkpointing."""
        saved = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_1_state_dict": self.critic_1.state_dict(),
            "critic_2_state_dict": self.critic_2.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }
        if self.auto_alpha and self.alpha_optimizer is not None:
            saved["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()
        if self.rnd:
            saved["rnd_state_dict"] = self.rnd.state_dict()
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        """Load model/optimizer/temperature states; targets are re-synced from the loaded critics."""
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": True, "iteration": True, "rnd": True}
        if load_cfg.get("actor"):
            self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self.critic_1.load_state_dict(loaded_dict["critic_1_state_dict"], strict=strict)
            self.critic_2.load_state_dict(loaded_dict["critic_2_state_dict"], strict=strict)
            self.critic_1_target.hard_sync()
            self.critic_2_target.hard_sync()
        if load_cfg.get("optimizer"):
            self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
            if self.auto_alpha and "alpha_optimizer_state_dict" in loaded_dict:
                self.alpha_optimizer.load_state_dict(loaded_dict["alpha_optimizer_state_dict"])
            if "log_alpha" in loaded_dict:
                self.log_alpha.data.copy_(loaded_dict["log_alpha"].to(self.device))
                self.alpha = self.log_alpha.exp().item()
        if load_cfg.get("rnd") and self.rnd and "rnd_state_dict" in loaded_dict:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
        return load_cfg.get("iteration", False)

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters from rank 0 to all GPUs."""
        params = [self.actor.state_dict(), self.critic_1.state_dict(), self.critic_2.state_dict()]
        torch.distributed.broadcast_object_list(params, src=0)
        self.actor.load_state_dict(params[0])
        self.critic_1.load_state_dict(params[1])
        self.critic_2.load_state_dict(params[2])
        self.critic_1_target.hard_sync()
        self.critic_2_target.hard_sync()

    def reduce_parameters(self, parameters: list) -> None:
        """Average gradients across GPUs for the given parameters (in place)."""
        grads = [p.grad.view(-1) for p in parameters if p.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for p in parameters:
            if p.grad is not None:
                numel = p.numel()
                p.grad.data.copy_(all_grads[offset : offset + numel].view_as(p.grad.data))
                offset += numel

    # -- construction ----------------------------------------------------------------------------------------

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> SAC:
        """Build the SAC algorithm (actor, twin critics, replay buffer) from a config dict."""
        alg_class: type[SAC] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[FuseModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic"]
        if cfg["algorithm"].get("rnd_cfg") is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        num_actions = env.num_actions
        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", num_actions, **cfg["actor"]).to(device)
        # Twin Q-critics: generic FuseModel fusing obs + action -> scalar Q.
        critic_1: FuseModel = critic_class(
            obs, cfg["obs_groups"], "critic", input_dims=[num_actions], output_dim=1, **cfg["critic"]
        ).to(device)
        critic_2: FuseModel = critic_class(
            obs, cfg["obs_groups"], "critic", input_dims=[num_actions], output_dim=1, **cfg["critic"]
        ).to(device)

        buffer_size = int(cfg["algorithm"].get("replay_buffer_size", 1_000_000))
        capacity_per_env = max(buffer_size // env.num_envs, 1)
        replay_buffer = ReplayBuffer(
            env.num_envs,
            capacity_per_env,
            obs,
            [num_actions],
            z_dim=0,
            batch_size=cfg["algorithm"].get("mini_batch_size", 256),
            device=device,
            keep_terminal=True,
        )

        return alg_class(
            actor,
            critic_1,
            critic_2,
            replay_buffer,
            num_actions,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg.get("multi_gpu"),
        )
