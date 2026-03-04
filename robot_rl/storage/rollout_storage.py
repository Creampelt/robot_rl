# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import torch
from tensordict import TensorDict

from robot_rl.utils import split_and_pad_trajectories


class RolloutStorage:
    @dataclass
    class Transition:
        observations: TensorDict | None = None
        actions: torch.Tensor | None = None
        privileged_actions: torch.Tensor | None = None
        rewards: torch.Tensor | None = None
        dones: torch.Tensor | None = None
        values: torch.Tensor | None = None
        actions_log_prob: torch.Tensor | None = None
        action_mean: torch.Tensor | None = None
        action_sigma: torch.Tensor | None = None
        hidden_states: tuple | None = None
        next_observations: TensorDict | None = None
        last_observations: TensorDict | None = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        training_type: Literal["rl", "distillation"],
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
        use_last_obs: bool = False,
    ):
        # store inputs
        self.training_type: Literal["rl", "distillation"] = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.actions_shape = actions_shape

        # Core
        self.observations = TensorDict(
            {key: torch.zeros(num_transitions_per_env, *value.shape, device=device) for key, value in obs.items()},
            batch_size=[num_transitions_per_env, num_envs],
            device=self.device,
        )
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # for distillation
        if training_type == "distillation":
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        # for reinforcement learning
        elif training_type == "rl":
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # For last observation (for estimation)
        self.last_obs = self.observations.clone() if use_last_obs else None

        # For RNN networks
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        # counter for the number of transitions stored
        self.step = 0

    def add_transitions(self, transition: Transition):
        # check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations.to(self.device))
        self.actions[self.step].copy_(transition.actions.to(self.device))
        self.rewards[self.step].copy_(transition.rewards.to(self.device).view(-1, 1))
        self.dones[self.step].copy_(transition.dones.to(self.device).view(-1, 1))

        # for distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions.to(self.device))
        # for reinforcement learning
        elif self.training_type == "rl":
            self.values[self.step].copy_(transition.values.to(self.device))
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.to(self.device).view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean.to(self.device))
            self.sigma[self.step].copy_(transition.action_sigma.to(self.device))

        # For last observation (for estimation)
        if self.last_obs is not None:
            self.last_obs[self.step].copy_(transition.last_observations.to(self.device))

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # increment the counter
        self.step += 1

    def _save_hidden_states(
        self,
        hidden_states: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None | tuple[None, None],
    ) -> None:
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden states to match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
        if self.saved_hidden_states_c is None:
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i].to(self.device))
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i].to(self.device))

    def clear(self) -> None:
        self.step = 0

    def compute_returns(
        self,
        last_values: torch.Tensor,
        gamma: float,
        lam: float,
        normalize_advantage: bool = True,
    ) -> None:
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            # if we are at the last step, bootstrap the return value
            if step == self.num_transitions_per_env - 1:
                next_values = last_values.to(self.device)
            else:
                next_values = self.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            self.returns[step] = advantage + self.values[step]

        # Compute the advantages
        self.advantages = self.returns - self.values
        # Normalize the advantages if flag is set
        # This is to prevent double normalization (i.e. if per minibatch normalization is used)
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    # for distillation
    def generator(
        self, device: str | None = None
    ) -> Iterator[tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            yield (
                self.observations[i].to(device),
                self.actions[i].to(device),
                self.privileged_actions[i].to(device),
                self.dones[i].to(device),
            )

    # for reinforcement learning with feedforward networks
    def mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8, device: str | None = None
    ) -> Iterator[
        tuple[
            TensorDict,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            tuple[None, None],
            None,
            TensorDict | None,
        ]
    ]:
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Core
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        # For PPO
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        # For last observation (for estimation)
        last_obs = self.last_obs.flatten(0, 1) if self.last_obs is not None else None

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # Create the mini-batch
                # -- Core
                obs_batch = observations[batch_idx]
                actions_batch = actions[batch_idx]

                # -- For PPO
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                # -- For last observation (for estimation)
                last_obs_batch = last_obs[batch_idx].to(device) if last_obs is not None else None

                # yield the mini-batch
                yield (
                    obs_batch.to(device),
                    actions_batch.to(device),
                    target_values_batch.to(device),
                    advantages_batch.to(device),
                    returns_batch.to(device),
                    old_actions_log_prob_batch.to(device),
                    old_mu_batch.to(device),
                    old_sigma_batch.to(device),
                    (None, None),
                    None,
                    last_obs_batch,
                )

    # for reinforcement learning with recurrent networks
    def recurrent_mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8, device: str | None = None
    ) -> Iterator[
        tuple[
            TensorDict,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            tuple[torch.Tensor | list[torch.Tensor], torch.Tensor | list[torch.Tensor]],
            torch.Tensor,
            TensorDict | None,
        ]
    ]:
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        assert self.saved_hidden_states_a is not None and self.saved_hidden_states_c is not None

        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)

        if self.last_obs is not None:
            padded_last_obs_trajectories, _ = split_and_pad_trajectories(self.last_obs, self.dones)
        else:
            padded_last_obs_trajectories = None

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]

                last_obs_batch = (
                    padded_last_obs_trajectories[:, first_traj:last_traj].to(device)
                    if padded_last_obs_trajectories is not None
                    else None
                )

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    .to(device)
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    .to(device)
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                yield (
                    obs_batch.to(device),
                    actions_batch.to(device),
                    values_batch.to(device),
                    advantages_batch.to(device),
                    returns_batch.to(device),
                    old_actions_log_prob_batch.to(device),
                    old_mu_batch.to(device),
                    old_sigma_batch.to(device),
                    (hid_a_batch, hid_c_batch),
                    masks_batch.to(device),
                    last_obs_batch,
                )

                first_traj = last_traj
