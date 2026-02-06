from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from robot_rl.modules import ForwardBackward
from robot_rl.storage import ReplayBuffer, TrajectoryBuffer
from robot_rl.utils import compute_td_targets, reset_parameters


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
        discriminator_learning_rate: float = 1e-4,
        critic_learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        max_grad_norm: float | None = None,
        clip_actions: float = 0.2,
        gamma: float = 0.99,
        train_goal_ratio: float = 0.5,
        expert_asm_ratio: float = 0.0,
        forward_backward_pessimism: float = 0.0,
        actor_pessimism: float = 0.5,
        critic_pessimism: float = 0.0,
        value_loss_coef: float = 1.0,
        ortho_loss_coef: float = 1.0,
        grad_loss_coef: float = 1.0,
        critic_reward_eps: float = 1e-7,
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
        self.discriminator_optimizer = optim.Adam(
            self.policy.discriminator.parameters(),
            lr=discriminator_learning_rate,
            weight_decay=weight_decay,
        )
        self.critic_optimizer = optim.Adam(
            self.policy.critic.parameters(),
            lr=critic_learning_rate,
            weight_decay=weight_decay,
        )

        self.replay_buffer: ReplayBuffer | None = None
        self.transition = ReplayBuffer.Transition()
        self.expert_buffer: TrajectoryBuffer | None = None

        self.forward_backward_pessimism = forward_backward_pessimism
        self.actor_pessimism = actor_pessimism
        self.critic_pessimism = critic_pessimism
        self.value_loss_coef = value_loss_coef
        self.ortho_loss_coef = ortho_loss_coef
        self.grad_loss_coef = grad_loss_coef
        self.critic_reward_eps = critic_reward_eps
        self.batch_size = batch_size
        self.clip_actions = clip_actions
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.z_dim = self.policy.z_dim

        # Precompute useful variables
        self._off_diag = 1 - torch.eye(batch_size, batch_size, device=self.device)
        self._off_diag_sum = self._off_diag.sum()
        self._mixed_z_probs = torch.tensor(
            [train_goal_ratio, expert_asm_ratio, 1 - train_goal_ratio - expert_asm_ratio], device=self.device
        )

    def init_storage(
        self,
        num_envs: int,
        episode_length_steps: int | torch.Tensor,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        capacity_scale: int,
        device: str,
        motion_paths: list[str],
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
        self.expert_buffer = TrajectoryBuffer(motion_paths, self.batch_size, device)

    def test_mode(self) -> None:
        pass

    def train_mode(self) -> None:
        # initialize weights for training
        self.policy.apply(reset_parameters)
        self.policy.train(True, device=self.device)

    def act(
        self,
        obs: TensorDict,
        z: torch.Tensor,
        dones: torch.Tensor | None,
        random_sample: bool = False,
    ) -> torch.Tensor:
        # compute the actions and values
        self.transition.actions = self.policy.act(obs, z).detach()
        # uniformly sample from action space if specified
        if random_sample:
            self.transition.actions.uniform_(-1.0, 1.0).detach()
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

    def update_rollout_z(
        self,
        z: torch.Tensor | None,
        dones: torch.Tensor | None,
        num_envs: int,
    ) -> torch.Tensor:
        new_z = self._sample_random_z(num_envs)
        if z is None or dones is None:
            return new_z
        return torch.where(dones.view(-1, 1), new_z, z)

    def sample_mixed_z(self, goal_obs: TensorDict, expert_z: torch.Tensor) -> torch.Tensor:
        z = self._sample_random_z(self.batch_size)
        mixed_types = torch.multinomial(self._mixed_z_probs, self.batch_size, replacement=True).view(-1, 1)

        # z's from goal_obs
        perm = torch.randperm(self.batch_size, device=self.device)
        goal_z = self.policy.B(goal_obs[perm])
        goal_z = self.policy.z_normalizer(goal_z)
        z = torch.where(mixed_types == 0, goal_z, z)

        # expert z's
        perm = torch.randperm(self.batch_size, device=self.device)
        z = torch.where(mixed_types == 1, expert_z[perm], z)

        return z

    def update(self) -> tuple[dict[str, torch.Tensor], dict]:
        assert self.replay_buffer is not None and self.expert_buffer is not None, (
            "Buffers have not yet been initialized. You must call `init_storage` before training."
        )
        (
            obs_batch,
            actions_batch,
            z_batch,
            next_obs_batch,
            gammas_batch,
        ) = self.replay_buffer.sample_mini_batch(self.device)
        expert_obs_batch, expert_next_obs_batch = self.expert_buffer.sample()
        expert_z_batch = self.policy.goal_inference(expert_next_obs_batch)
        mixed_z = self.sample_mixed_z(next_obs_batch, expert_z_batch)
        loss_dict = {}
        extras = {}

        disc_loss_dict, disc_extras = self._update_discriminator(obs_batch, z_batch, expert_obs_batch, expert_z_batch)
        fb_loss_dict, fb_extras = self._update_forward_backward(
            obs_batch,
            actions_batch,
            next_obs_batch,
            gammas_batch,
            mixed_z,
        )
        critic_loss_dict, critic_extras = self._update_critic(
            obs_batch,
            mixed_z,
            actions_batch,
            next_obs_batch,
            gammas_batch,
        )
        actor_loss_dict, actor_extras = self._update_actor(obs_batch, mixed_z)

        loss_dict.update(disc_loss_dict)
        loss_dict.update(fb_loss_dict)
        loss_dict.update(critic_loss_dict)
        loss_dict.update(actor_loss_dict)

        extras.update(disc_extras)
        extras.update(fb_extras)
        extras.update(critic_extras)
        extras.update(actor_extras)

        with torch.no_grad():
            self.policy.soft_update_targets()

        return loss_dict, {"log": extras}

    """
    Helper functions
    """

    def broadcast_parameters(self) -> None:
        pass

    def reduce_parameters(self) -> None:
        pass

    def _sample_random_z(self, size: int) -> torch.Tensor:
        z = torch.randn((size, self.z_dim), dtype=torch.float32, device=self.device)
        z = self.policy.z_normalizer(z)
        return z

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
            target_M = compute_td_targets(target_Ms, self.forward_backward_pessimism)

        Fs = self.policy.F(obs, z, actions)  # num_parallel x batch_size x z_dim
        B = self.policy.B(next_obs)  # batch_size x z_dim
        Ms = torch.matmul(Fs, B.T)  # num_parallel x batch_size x batch_size

        # FB loss
        diff = Ms - gammas * target_M
        fb_offdiag = 0.5 * (diff * self._off_diag).pow(2).sum() / self._off_diag_sum
        fb_diag = -torch.diagonal(Ms, dim1=1, dim2=2).mean()
        fb_loss = fb_offdiag + fb_diag

        # Orthonormality loss
        cov = torch.matmul(B, B.T)
        orth_offdiag = 0.5 * (cov * self._off_diag).pow(2).sum() / self._off_diag_sum
        orth_diag = -cov.diag().mean()
        orth_loss = self.ortho_loss_coef * (orth_offdiag + orth_diag)

        # Fz regularization loss
        q_loss = torch.zeros(1, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            next_Qs = (target_Fs * z).sum(dim=-1)
            # next_Qs = self.policy.evaluate(next_obs, z, next_actions, use_target=True)  # batch_size
            next_Q = compute_td_targets(next_Qs, self.forward_backward_pessimism)
            cov = torch.matmul(B.T, B) / B.shape[0]  # z_dim x z_dim
            implicit_reward = (torch.matmul(B, cov.inverse()) * z).sum(dim=-1)  # batch_size
            target_Q = implicit_reward.detach() + gammas.squeeze() * next_Q  # batch_size
            target_Q = target_Q.expand(Fs.shape[0], -1)  # num_parallel x batch_size
        Qs = (Fs * z).sum(dim=-1)  # self.policy.evaluate(obs, z, actions)  # num_parallel x batch_size
        q_loss = self.value_loss_coef * 0.5 * Fs.shape[0] * nn.functional.mse_loss(Qs, target_Q)

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

        with torch.no_grad():
            loss_dict = {
                "FB/forward_backward": fb_loss.detach(),
                "FB/orthonormality": orth_loss.detach(),
                "FB/value_regularization": q_loss.detach(),
            }

            extras = {
                "FB/target_M": target_M.mean().detach(),
                "FB/M1": Ms[0].mean().detach(),
                "FB/F1": Fs[0].mean().detach(),
                "FB/B": B.mean().detach(),
                "FB/F1_norm": Fs[0].norm(dim=-1).mean().detach(),
                "FB/B_norm": B.norm(dim=-1).mean().detach(),
                "FB/z_norm": z.norm(dim=-1).mean().detach(),
            }

        return loss_dict, extras

    def _update_actor(self, obs: TensorDict, z: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict]:
        actions = self.policy.act(obs, z, clip=self.clip_actions)

        Qs_discriminator = self.policy.evaluate(obs, z, actions)
        Q_discriminator = compute_td_targets(Qs_discriminator, self.actor_pessimism)

        Fs = self.policy.F(obs, z, actions)
        Qs_fb = (Fs * z).sum(-1)
        Q_fb = compute_td_targets(Qs_fb, self.actor_pessimism)

        actor_loss = -Q_discriminator.mean() * Q_fb.abs().mean() - Q_fb.mean()

        # optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        with torch.no_grad():
            loss_dict = {
                "actor": actor_loss.detach(),
            }

            extras = {
                "actor/discriminator_value": Q_discriminator.mean().detach(),
                "actor/fb_value": Q_fb.mean().detach(),
            }

        return loss_dict, extras

    def _update_discriminator(
        self,
        obs: TensorDict,
        z: torch.Tensor,
        expert_obs: TensorDict,
        expert_z: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict]:
        train_logits = self.policy.D(obs, z)
        expert_logits = self.policy.D(expert_obs, expert_z)
        # binary cross entropy
        train_loss = nn.functional.softplus(train_logits)
        expert_loss = -nn.functional.logsigmoid(expert_logits)

        # compute gradient loss
        normalized_obs = self.policy.get_critic_obs(obs)
        normalized_obs = self.policy.critic_obs_normalizer(normalized_obs)
        normalized_expert_obs = self.policy.get_critic_obs(expert_obs)
        normalized_expert_obs = self.policy.critic_obs_normalizer(normalized_expert_obs)

        alpha = torch.rand(self.batch_size, 1, device=self.device)
        interpolates = torch.cat(
            [
                (alpha * normalized_obs + (1 - alpha) * normalized_expert_obs).requires_grad_(True),
                (alpha * z + (1 - alpha) * expert_z).requires_grad_(True),
            ],
            dim=1,
        )
        d_interpolates = self.policy.discriminator(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_loss = self.grad_loss_coef * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        loss = (train_loss + expert_loss).mean() + grad_loss

        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

        with torch.no_grad():
            loss_dict = {
                "discriminator/train": train_loss.mean().detach(),
                "discriminator/expert": expert_loss.mean().detach(),
                "discriminator/gradient": grad_loss.detach(),
            }

        return loss_dict, {}

    def _update_critic(
        self,
        obs: TensorDict,
        z: torch.Tensor,
        actions: torch.Tensor,
        next_obs: TensorDict,
        gammas: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        with torch.no_grad():
            logits = self.policy.D(obs, z).clamp(self.critic_reward_eps, 1 - self.critic_reward_eps)
            discriminator_reward = torch.log(logits / (1 - logits))
            next_actions = self.policy.act(next_obs, z, clip=self.clip_actions)
            next_Qs = self.policy.evaluate(next_obs, z, next_actions, use_target=True)
            target_Q = discriminator_reward + gammas * compute_td_targets(next_Qs, self.critic_pessimism)
            target_Q = target_Q.expand(self.policy.critic.num_parallel, -1, -1)

        Qs = self.policy.evaluate(obs, z, actions)
        loss = 0.5 * self.policy.critic.num_parallel * nn.functional.mse_loss(Qs, target_Q)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            loss_dict = {
                "critic": loss.detach(),
            }

            extras_dict = {
                "critic/target_Q": target_Q.mean().detach(),
                "critic/Q": Qs.mean().detach(),
                "critic/discriminator_reward": discriminator_reward.mean().detach(),
            }

        return loss_dict, extras_dict
