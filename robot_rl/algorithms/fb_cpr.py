from __future__ import annotations

from typing import Literal
import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from robot_rl.modules import ForwardBackward
from robot_rl.storage import ReplayBuffer
from robot_rl.utils import uncertainty_penalized_mean


class FbCpr:
    """FB-CPR algorithm (https://arxiv.org/pdf/2504.11054)."""

    policy: ForwardBackward
    """The actor critic module."""

    def __init__(
        self,
        policy: ForwardBackward,
        actor_learning_rate: float = 1e-4,
        forward_learning_rate: float = 1e-4,
        backward_learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        max_grad_norm: float | None = None,
        clip_actions: float = 0.2,
        tau: float = 0.01,
        gamma: float = 0.99,
        train_goal_ratio: float = 0.5,
        forward_backward_pessimism: float = 0.0,
        actor_pessimism: float = 0.5,
        value_loss_coef: float = 1.0,
        ortho_loss_coef: float = 1.0,
        batch_size: int = 1024,
        device: str = "cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # TODO: multi-gpu compatibility

        self.policy = policy
        self.policy.to(self.device)
        # Create optimizers
        self.actor_optimizer = optim.Adam(
            self.policy.actor.parameters(),
            lr=actor_learning_rate,
            weight_decay=weight_decay,
        )
        self.forward_optimizer = optim.Adam(
            self.policy.forward_map.parameters(),
            lr=forward_learning_rate,
            weight_decay=weight_decay,
        )
        self.backward_optimizer = optim.Adam(
            self.policy.backward_map.parameters(),
            lr=backward_learning_rate,
            weight_decay=weight_decay,
        )

        self.replay_buffer: ReplayBuffer | None = None
        self.transition = ReplayBuffer.Transition()

        self.train_goal_ratio = train_goal_ratio
        self.forward_backward_pessimism = forward_backward_pessimism
        self.actor_pessimism = actor_pessimism
        self.value_loss_coef = value_loss_coef
        self.ortho_loss_coef = ortho_loss_coef
        self.batch_size = batch_size
        self.clip_actions = clip_actions
        self.max_grad_norm = max_grad_norm
        self.tau = tau
        self.gamma = gamma
        self.z_dim = self.policy.z_dim

        # Precompute useful variables
        self._off_diag = 1 - torch.eye(batch_size, batch_size, device=self.device)
        self._off_diag_sum = self._off_diag.sum()

    def init_storage(
        self,
        num_envs: int,
        episode_length_steps: int | torch.Tensor,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        capacity_scale: int,
        device: str,
    ) -> None:
        if isinstance(episode_length_steps, torch.Tensor):
            episode_length_steps = int(episode_length_steps.max().item())
        self.replay_buffer = ReplayBuffer(
            "rl",
            num_envs,
            capacity_scale * episode_length_steps,
            obs,
            actions_shape,
            self.z_dim,
            self.batch_size,
            device,
        )

    def test_mode(self) -> None:
        pass

    def train_mode(self) -> None:
        self.policy.train()

    def act(self, obs: TensorDict, z: torch.Tensor, dones: torch.Tensor | None) -> torch.Tensor:
        # compute the actions and values
        self.transition.actions = self.policy.act(obs, z).detach()
        # record obs and dones before env.step()
        self.transition.observations = obs
        self.transition.dones = dones
        self.transition.context = z
        return self.transition.actions

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        assert self.replay_buffer is not None, (
            "Replay buffer has not yet been initialized. You must call `init_storage` before training."
        )
        # Record the and next obs and next terminated (after env.step)
        # Terminated is all dones that are not time_outs (used to compute discount factor)
        self.transition.next_terminated = (dones * extras["time_outs"]).byte()
        self.transition.next_observations = obs

        # record the transition
        self.replay_buffer.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self) -> None:
        assert self.replay_buffer is not None, (
            "Replay buffer has not yet been initialized. You must call `init_storage` before training."
        )
        self.replay_buffer.compute_returns(self.gamma)

    def update_z(
        self,
        z: torch.Tensor | None,
        dones: torch.Tensor | None,
        num_envs: int,
    ) -> torch.Tensor:
        new_z = self.policy.sample_z(num_envs, device=self.device)
        if z is None or dones is None:
            return new_z
        return torch.where(dones.view(-1, 1), new_z, z)

    def update(self) -> tuple[dict[str, torch.Tensor], dict]:
        assert self.replay_buffer is not None, (
            "Replay buffer has not yet been initialized. You must call `init_storage` before training."
        )
        (
            obs_batch,
            actions_batch,
            context_batch,
            next_obs_batch,
            gammas_batch,
        ) = self.replay_buffer.sample_mini_batch(self.device)
        z = self.policy.sample_z(
            self.batch_size,
            goal_obs=next_obs_batch,
            goal_ratio=self.train_goal_ratio,
            device=self.device,
        )
        loss_dict = {}
        extras = {}

        fb_loss_dict, fb_extras = self._update_forward_backward(
            obs_batch,
            actions_batch,
            next_obs_batch,
            gammas_batch,
            z,
        )
        actor_loss_dict, actor_extras = self._update_actor(obs_batch, z)

        loss_dict.update(fb_loss_dict)
        loss_dict.update(actor_loss_dict)

        extras.update(fb_extras)
        extras.update(actor_extras)

        with torch.no_grad():
            self.policy.soft_update_targets(self.tau)

        return loss_dict, {"log": extras}

    """
    Helper functions
    """

    def broadcast_parameters(self) -> None:
        pass

    def reduce_parameters(self) -> None:
        pass

    def _update_forward_backward(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
        next_obs: TensorDict,
        gammas: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict]:
        # Forward-Backward loss
        with torch.no_grad():
            next_actions = self.policy.act(next_obs, z, clip=self.clip_actions)
            target_Fs = self.policy.F(next_obs, z, next_actions, use_target=True)
            target_B = self.policy.B(next_obs, use_target=True)
            target_Ms = torch.matmul(target_Fs, target_B.T)
            target_M = uncertainty_penalized_mean(target_Ms, self.forward_backward_pessimism)

        Fs = self.policy.F(obs, z, actions)  # num_parallel x batch_size x z_dim
        B = self.policy.B(next_obs)  # batch_size x z_dim
        Ms = torch.matmul(Fs, B.T)  # num_parallel x batch_size x batch_size

        diff = Ms - gammas * target_M
        fb_offdiag = 0.5 * (diff * self._off_diag).pow(2).sum() / self._off_diag_sum
        fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
        fb_loss = fb_offdiag + fb_diag

        # Orthonormality loss
        cov = torch.matmul(B, B.T)
        orth_offdiag = 0.5 * (cov * self._off_diag).pow(2).sum() / self._off_diag_sum
        orth_diag = -cov.diag().mean()
        orth_loss = self.ortho_loss_coef * (orth_offdiag + orth_diag)

        # Critic loss
        q_loss = torch.zeros(1, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            next_values = self.policy.evaluate(next_obs, z, next_actions, use_target=True)  # batch_size
            next_values = uncertainty_penalized_mean(next_values, self.forward_backward_pessimism)
            cov = torch.matmul(B.T, B) / B.shape[0]  # z_dim x z_dim
            implicit_reward = (torch.matmul(B, cov.inverse()) * z).sum(dim=-1)  # batch_size
            target_Q = implicit_reward.detach() + gammas.squeeze() * next_values  # batch_size
            target_Q = target_Q.expand(self.policy.critic_num_parallel, -1)  # num_parallel x batch_size
        Qs = self.policy.evaluate(obs, z, actions)  # num_parallel x batch_size
        q_loss = self.value_loss_coef * 0.5 * Qs.shape[0] * nn.functional.mse_loss(Qs, target_Q)

        loss = fb_loss + orth_loss + q_loss

        # optimize FB
        self.forward_optimizer.zero_grad()
        self.backward_optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy.forward_map.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.policy.backward_map.parameters(), self.max_grad_norm)
        self.forward_optimizer.step()
        self.backward_optimizer.step()

        loss_dict = {
            "forward_backward": fb_loss.detach(),
            "orthonormality": orth_loss.detach() / self.ortho_loss_coef,
            "value": q_loss.detach() / self.value_loss_coef,
        }

        extras = {
            "target_M": target_M.detach().mean(),
            "M1": Ms[0].detach().mean(),
            "F1": Fs[0].detach().mean(),
            "B": B.detach().mean(),
            "F1_norm": Fs[0].detach().norm(dim=-1).mean(),
            "B_norm": B.detach().norm(dim=-1).mean(),
            "z_norm": z.detach().norm(dim=-1).mean(),
            # "proj": (Fs @ B.T).norm(),
            # "residual": (Fs - (Fs @ B.T) @ B).norm(),
        }

        return loss_dict, extras

    def _update_actor(self, obs: TensorDict, z: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict]:
        actions = self.policy.act(obs, z, clip=self.clip_actions)
        Qs = self.policy.evaluate(obs, z, actions)
        Q = uncertainty_penalized_mean(Qs, self.actor_pessimism)
        actor_loss = -Q.mean()

        # optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        loss_dict = {"actor": actor_loss.detach()}

        extras = {"q": Q.detach().mean()}

        return loss_dict, extras
