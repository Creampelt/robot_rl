from __future__ import annotations
from typing import cast

import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from robot_rl.networks import EMANormalization
from robot_rl.modules import ForwardBackward
from robot_rl.storage import ReplayBuffer, TrajectoryBuffer, ZBuffer
from robot_rl.utils import compute_td_targets, reset_parameters, compute_emd, forward_sliding_mean


class FbCpr:
    """FB-CPR algorithm (https://arxiv.org/pdf/2504.11054)."""

    policy: ForwardBackward
    """The actor critic module."""

    def __init__(
        self,
        policy: ForwardBackward,
        motion_dir: str,
        expert_bucket_size: int,
        expert_sequence_length: int,
        steps_per_z_update: int,
        actor_learning_rate: float = 1e-4,
        forward_learning_rate: float = 1e-4,
        backward_learning_rate: float = 1e-4,
        discriminator_learning_rate: float = 1e-4,
        disc_critic_learning_rate: float = 1e-4,
        aux_critic_learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        max_grad_norm: float | None = None,
        clip_actions: float = 0.2,
        gamma: float = 0.99,
        train_goal_ratio: float = 0.5,
        expert_asm_ratio: float = 0.0,
        expert_rollout_ratio: float = 0.5,
        expert_rollout_length: int = 250,
        z_relabel_ratio: float = 0.8,
        forward_backward_pessimism: float = 0.0,
        actor_pessimism: float = 0.5,
        disc_critic_pessimism: float = 0.5,
        aux_critic_pessimism: float = 0.5,
        discriminator_reg_coef: float = 1.0,
        aux_reg_coef: float = 1.0,
        value_loss_coef: float = 1.0,
        ortho_loss_coef: float = 1.0,
        grad_loss_coef: float = 1.0,
        discriminator_reward_eps: float = 1e-7,
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
        self.disc_critic_optimizer = optim.Adam(
            self.policy.disc_critic.parameters(),
            lr=disc_critic_learning_rate,
            weight_decay=weight_decay,
        )
        self.aux_critic_optimizer = optim.Adam(
            self.policy.aux_critic.parameters(),
            lr=aux_critic_learning_rate,
            weight_decay=weight_decay,
        )

        # exponential moving average normalization for aux rewards
        self.aux_reward_normalizer = EMANormalization(scale=True).to(self.device)

        self._z_buffer: ZBuffer | None = None
        self._expert_buffer: TrajectoryBuffer | None = None
        self._replay_buffer: ReplayBuffer | None = None
        self.transition = ReplayBuffer.Transition()

        # for expert rollout context
        self.expert_rollout_envs: torch.Tensor | None = None
        self.expert_rollout_z: torch.Tensor | None = None

        self.steps_per_z_update = steps_per_z_update
        self.train_goal_ratio = train_goal_ratio
        self.expert_asm_ratio = expert_asm_ratio
        self.expert_rollout_ratio = expert_rollout_ratio
        self.expert_rollout_length = expert_rollout_length
        self.z_relabel_ratio = z_relabel_ratio
        self.forward_backward_pessimism = forward_backward_pessimism
        self.actor_pessimism = actor_pessimism
        self.disc_critic_pessimism = disc_critic_pessimism
        self.aux_critic_pessimism = aux_critic_pessimism
        self.discriminator_reg_coef = discriminator_reg_coef
        self.aux_reg_coef = aux_reg_coef
        self.value_loss_coef = value_loss_coef
        self.ortho_loss_coef = ortho_loss_coef
        self.grad_loss_coef = grad_loss_coef
        self.discriminator_reward_eps = discriminator_reward_eps
        self.batch_size = batch_size
        self.clip_actions = clip_actions
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.z_dim = self.policy.z_dim

        motion_dir = os.path.abspath(motion_dir)
        self.motion_paths = glob.glob(os.path.join(motion_dir, "*.pt"))
        self.expert_bucket_size = expert_bucket_size
        self.expert_sequence_length = expert_sequence_length

        # Precompute useful variables
        self._off_diag = 1 - torch.eye(batch_size, batch_size, device=self.device)
        self._off_diag_sum = self._off_diag.sum()

    @property
    def replay_buffer(self) -> ReplayBuffer:
        assert self._replay_buffer is not None, "Replay buffer has not yet been initialized."
        return self._replay_buffer

    @property
    def expert_buffer(self) -> TrajectoryBuffer:
        assert self._expert_buffer is not None, "Expert buffer has not yet been initialized."
        return self._expert_buffer

    @property
    def z_buffer(self) -> ZBuffer:
        assert self._z_buffer is not None, "Z buffer has not yet been initialized."
        return self._z_buffer

    def init_storage(
        self,
        num_envs: int,
        episode_length_steps: int | torch.Tensor,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        capacity_scale: int,
        z_buffer_capacity: int,
        device: str,
    ) -> None:
        if isinstance(episode_length_steps, torch.Tensor):
            episode_length_steps = int(episode_length_steps.max().item())
        self._replay_buffer = ReplayBuffer(
            num_envs,
            capacity_scale * episode_length_steps,
            obs,
            actions_shape,
            self.z_dim,
            self.batch_size,
            device,
        )
        self._expert_buffer = TrajectoryBuffer(
            self.motion_paths,
            self.expert_bucket_size,
            self.policy.obs_groups["expert"],
            device,
        )
        self._z_buffer = ZBuffer(z_buffer_capacity, self.z_dim, device)

    def test_mode(self) -> None:
        self.policy.eval()

    def train_mode(self) -> None:
        self.policy.train()

    def prepare_for_training(self) -> None:
        # initialize weights for training
        self.policy.apply(reset_parameters)
        self.policy.init_targets(self.device)

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
        # dones is None if this is the first step
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
        # Record the rewards
        self.transition.rewards = rewards
        # Record the and next obs and next terminated (after env.step)
        # Terminated is all dones that are not time_outs (used to compute discount factor)
        self.transition.next_terminated = (dones * ~extras["time_outs"]).byte()
        self.transition.next_observations = obs

        # record the transition
        self.replay_buffer.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_gammas(self) -> None:
        self.replay_buffer.compute_gammas(self.gamma)

    def update_rollout_z(
        self,
        z: torch.Tensor | None,
        step: torch.Tensor,
        num_envs: int,
    ) -> torch.Tensor:
        step = step.long()
        # update from replay buffer
        if z is None:
            z = self._sample_random_z(num_envs)
        elif torch.any(
            step % self.steps_per_z_update == 0
        ):  # step tensor is all same value, but we do torch.any just to be safe
            z = self.z_buffer.sample(num_envs, device=self.device)

        # update from expert buffer
        rollout_idx = step % self.expert_rollout_length
        if torch.any(rollout_idx == 0) or self.expert_rollout_envs is None or self.expert_rollout_z is None:
            expert_env_mask = torch.rand(num_envs, device=self.device) > self.expert_rollout_ratio
            self.expert_rollout_envs = torch.argwhere(expert_env_mask).flatten()
            num_expert_updates = self.expert_rollout_envs.shape[0]

            _, expert_next_obs = self.expert_buffer.sample(
                num_expert_updates * self.expert_rollout_length,
                device=self.device,
            )
            expert_z = self.policy.B(expert_next_obs)  # num_expert_updates * expert_rollout_length, z_dim
            expert_z = expert_z.view(
                num_expert_updates, self.expert_rollout_length, -1
            )  # num_expert_updates, expert_rollout_length, z_dim
            expert_z = forward_sliding_mean(expert_z, self.expert_sequence_length, dim=1)
            self.expert_rollout_z = cast(
                torch.Tensor, self.policy.z_normalizer(expert_z)
            )  # num_expert_updates, expert_rollout_length, z_dim
        env_idxs = torch.arange(self.expert_rollout_envs.shape[0])
        z[self.expert_rollout_envs] = self.expert_rollout_z[env_idxs, rollout_idx[self.expert_rollout_envs]]

        return z

    @torch.no_grad()
    def sample_mixed_z(self, goal_obs: TensorDict, expert_z: torch.Tensor) -> torch.Tensor:
        z = self._sample_random_z(self.batch_size)

        mix_probs = torch.tensor(
            [self.train_goal_ratio, self.expert_asm_ratio, 1 - self.train_goal_ratio - self.expert_asm_ratio],
            device=self.device,
        )
        mix_indices = torch.multinomial(mix_probs, self.batch_size, replacement=True).view(-1, 1)

        # zs for encoded train goals
        perm = torch.randperm(self.batch_size, device=self.device)
        goal_z = self.policy.goal_inference(goal_obs)
        z = torch.where(mix_indices == 0, goal_z[perm], z)

        # zs from expert trajectories
        perm = torch.randperm(self.batch_size, device=self.device)
        z = torch.where(mix_indices == 1, expert_z[perm], z)

        return z

    @torch.no_grad()
    def encode_expert(self, obs: TensorDict) -> torch.Tensor:
        expert_B = self.policy.B(obs)  # batch_size, z_dim
        expert_B = expert_B.view(  # batch_size / seq_length, seq_length, z_dim
            -1,
            self.expert_sequence_length,
            *expert_B.shape[1:],
        )
        expert_z = expert_B.mean(dim=1)  # batch_size // seq_length, z_dim
        expert_z = self.policy.z_normalizer(expert_z)
        return torch.repeat_interleave(expert_z, self.expert_sequence_length, dim=0)  # batch_size, z_dim

    def update_priorities(
        self,
        qpos: torch.Tensor,
        expert_qpos: torch.Tensor,
        start_idx: int,
    ) -> dict[str, torch.Tensor]:
        assert self.expert_buffer is not None

        # compute priorities as 2^{max(0.5, min(2, emd)) * 4}
        mini_batch_size = qpos.shape[0]
        emds = torch.empty((mini_batch_size,), device=self.device)
        for i in range(mini_batch_size):
            emds[i] = compute_emd(qpos[i], expert_qpos[i])
        priorities = torch.pow(2, emds.clamp(min=0.5, max=2.0) * 4)

        # save to expert buffer
        self.expert_buffer.update_priorities(priorities, slice(start_idx, start_idx + priorities.shape[0]))

        return {
            "emd": emds.detach().cpu(),
        }

    def normalize_priorities(self) -> None:
        self.expert_buffer.normalize_priorities()

    def update(self) -> tuple[dict[str, torch.Tensor], dict]:
        obs, next_obs, actions, rewards, gammas, train_z = self.replay_buffer.sample_mini_batch(self.device)
        expert_obs, expert_next_obs = self.expert_buffer.sample(self.batch_size, self.device)

        # update parameters for all normalizers
        self.policy.update_normalization(obs)
        self.policy.update_normalization(next_obs)

        # encode expert z
        expert_z = self.encode_expert(expert_next_obs)

        # update discriminator
        disc_loss_dict, disc_extras = self._update_discriminator(obs, train_z, expert_obs, expert_z)

        # sample and store mixed z
        z = self.sample_mixed_z(next_obs, expert_z)
        self.z_buffer.add(z)

        # replace some train_zs with sampled zs
        relabel_mask = torch.rand(self.batch_size, 1, device=self.device) < self.z_relabel_ratio
        train_z = torch.where(relabel_mask, z, train_z)

        # normalize aux rewards
        rewards = self.aux_reward_normalizer(rewards)

        fb_loss_dict, fb_extras = self._update_forward_backward(obs, actions, next_obs, gammas, train_z)
        disc_critic_loss_dict, disc_critic_extras = self._update_disc_critic(obs, train_z, actions, next_obs, gammas)
        aux_critic_loss_dict, aux_critic_extras = self._update_aux_critic(
            obs,
            train_z,
            actions,
            next_obs,
            gammas,
            rewards,
        )
        actor_loss_dict, actor_extras = self._update_actor(obs, train_z, actions)

        # prepare logging dicts
        loss_dict = {}
        loss_dict.update(disc_loss_dict)
        loss_dict.update(fb_loss_dict)
        loss_dict.update(disc_critic_loss_dict)
        loss_dict.update(aux_critic_loss_dict)
        loss_dict.update(actor_loss_dict)

        extras = {}
        extras.update(disc_extras)
        extras.update(fb_extras)
        extras.update(disc_critic_extras)
        extras.update(aux_critic_extras)
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

    def _update_discriminator(
        self,
        obs: TensorDict,
        z: torch.Tensor,
        expert_obs: TensorDict,
        expert_z: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict]:
        expert_logits = self.policy.D(expert_obs, expert_z)
        unlabeled_logits = self.policy.D(obs, z)
        # binary cross entropy
        expert_loss = -nn.functional.logsigmoid(expert_logits)
        unlabeled_loss = nn.functional.softplus(unlabeled_logits)
        loss = torch.mean(expert_loss + unlabeled_loss)

        # compute gradient penalty loss
        normalized_obs = self.policy.get_discriminator_obs(obs)
        normalized_expert_obs = self.policy.get_discriminator_obs(expert_obs)

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

        grad_loss = self.grad_loss_coef * torch.mean(torch.square(gradients.norm(2, dim=1) - 1))

        loss += grad_loss

        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

        with torch.no_grad():
            loss_dict = {
                "Discriminator_Loss/total": loss.mean().detach(),
                "Discriminator_Loss/train": unlabeled_loss.mean().detach(),
                "Discriminator_Loss/expert": expert_loss.mean().detach(),
                "Discriminator_Loss/gradient": grad_loss.mean().detach(),
            }

        return loss_dict, {}

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
        fb_diag = -torch.diagonal(Ms, dim1=1, dim2=2).mean() * Ms.shape[0]
        fb_loss = fb_offdiag + fb_diag

        # Orthonormality loss
        Cov = torch.matmul(B, B.T)
        orth_offdiag = 0.5 * (Cov * self._off_diag).pow(2).sum() / self._off_diag_sum
        orth_diag = -Cov.diag().mean()
        orth_loss = self.ortho_loss_coef * (orth_offdiag + orth_diag)

        # Fz regularization loss
        q_loss = torch.zeros(1, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            next_Qs = (target_Fs * z).sum(dim=-1)  # batch_size
            next_Q = compute_td_targets(next_Qs, self.forward_backward_pessimism)
            cov = torch.matmul(B.T, B) / B.shape[0]  # z_dim x z_dim
            B_inv_cov = torch.linalg.solve(cov, B, left=False)
            implicit_reward = (B_inv_cov * z).sum(dim=-1)  # batch_size
            target_Q = implicit_reward.detach() + gammas.squeeze() * next_Q  # batch_size
            target_Q = target_Q.expand(Fs.shape[0], -1)  # num_parallel x batch_size
        Qs = (Fs * z).sum(dim=-1)  # num_parallel x batch_size
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
                "Forward_Backward_Loss/total": loss.mean().detach(),
                "Forward_Backward_Loss/fb": fb_loss.mean().detach(),
                "Forward_Backward_Loss/ortho": orth_loss.mean().detach(),
                "Forward_Backward_Loss/value_reg": q_loss.mean().detach(),
            }

            extras = {
                "fb_target_successor_measure": target_M.mean().detach(),
                "fb_successor_measure": Ms[0].mean().detach(),
                "fb_forward": Fs[0].mean().detach(),
                "fb_backward": B.mean().detach(),
                "fb_forward_norm": Fs[0].norm(dim=-1).mean().detach(),
            }

        return loss_dict, extras

    def _update_disc_critic(
        self,
        obs: TensorDict,
        z: torch.Tensor,
        actions: torch.Tensor,
        next_obs: TensorDict,
        gammas: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        num_parallel = self.policy.disc_critic.num_parallel
        with torch.no_grad():
            # compute discriminator reward
            logits = self.policy.D(obs, z).clamp(self.discriminator_reward_eps, 1 - self.discriminator_reward_eps)
            discriminator_reward = torch.log(logits / (1 - logits))
            # compute target value
            next_actions = self.policy.act(next_obs, z, clip=self.clip_actions)
            next_Qs = self.policy.evaluate_discriminator(next_obs, z, next_actions, use_target=True)
            target_Q = discriminator_reward + gammas * compute_td_targets(next_Qs, self.disc_critic_pessimism)
            target_Q = target_Q.expand(num_parallel, -1, -1)

        Qs = self.policy.evaluate_discriminator(obs, z, actions)
        loss = 0.5 * num_parallel * nn.functional.mse_loss(Qs, target_Q)

        self.disc_critic_optimizer.zero_grad()
        loss.backward()
        self.disc_critic_optimizer.step()

        with torch.no_grad():
            loss_dict = {
                "Critic_Loss/discriminator_critic": loss.mean().detach(),
            }

            extras_dict = {
                "discriminator_target_value": target_Q.mean().detach(),
                "discriminator_value": Qs.mean().detach(),
                "discriminator_reward": discriminator_reward.mean().detach(),
            }

        return loss_dict, extras_dict

    def _update_aux_critic(
        self,
        obs: TensorDict,
        z: torch.Tensor,
        actions: torch.Tensor,
        next_obs: TensorDict,
        gammas: torch.Tensor,
        rewards: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        num_parallel = self.policy.aux_critic.num_parallel
        with torch.no_grad():
            next_actions = self.policy.act(next_obs, z, clip=self.clip_actions)
            next_Qs = self.policy.evaluate_aux(next_obs, z, next_actions)
            target_Q = rewards.unsqueeze(1) + gammas * compute_td_targets(next_Qs, self.aux_critic_pessimism)
            target_Q = target_Q.expand(num_parallel, -1, -1)
        # compute critic loss
        Qs = self.policy.evaluate_aux(obs, z, actions)
        loss = 0.5 * num_parallel * nn.functional.mse_loss(Qs, target_Q)

        self.aux_critic_optimizer.zero_grad()
        loss.backward()
        self.aux_critic_optimizer.step()

        with torch.no_grad():
            loss_dict = {
                "Critic_Loss/auxiliary_critic": loss.mean().detach(),
            }

            extras_dict = {
                "aux_target_value": target_Q.mean().detach(),
                "aux_value": Qs.mean().detach(),
            }

        return loss_dict, extras_dict

    def _update_actor(
        self,
        obs: TensorDict,
        z: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict]:
        actions = self.policy.act(obs, z, clip=self.clip_actions)

        # compute discriminator reward loss
        Qs_discriminator = self.policy.evaluate_discriminator(obs, z, actions)
        Q_discriminator = (
            -self.discriminator_reg_coef * compute_td_targets(Qs_discriminator, self.actor_pessimism).mean()
        )

        # compute auxiliary reward loss
        Qs_aux = self.policy.evaluate_aux(obs, z, actions)
        Q_aux = -self.aux_reg_coef * compute_td_targets(Qs_aux, self.actor_pessimism).mean()

        Fs = self.policy.F(obs, z, actions)
        Qs_fb = (Fs * z).sum(-1)
        Q_fb = compute_td_targets(Qs_fb, self.actor_pessimism)
        reg_weight = Q_fb.abs().mean().detach()
        Q_fb = -Q_fb.mean()

        actor_loss = Q_fb + Q_discriminator * reg_weight + Q_aux * reg_weight

        # optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        with torch.no_grad():
            loss_dict = {
                "Actor_Loss/total": actor_loss.mean().detach(),
                "Actor_Loss/discriminator": Q_discriminator.mean().detach(),
                "Actor_Loss/aux": Q_aux.mean().detach(),
                "Actor_Loss/fb": Q_fb.mean().detach(),
            }

            extras_dict = {
                "actor_reg_weight": reg_weight.mean().detach(),
            }

        return loss_dict, extras_dict
