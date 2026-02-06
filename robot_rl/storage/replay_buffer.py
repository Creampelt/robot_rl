# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import Literal
from dataclasses import dataclass


class ReplayBuffer:
    @dataclass
    class Transition:
        observations: TensorDict | None = None
        actions: torch.Tensor | None = None
        dones: torch.Tensor | None = None
        context: torch.Tensor | None = None
        next_observations: TensorDict | None = None
        next_terminated: torch.Tensor | None = None

        def clear(self) -> None:
            self.__init__()

        def to_full(self) -> ReplayBuffer.FullTransition:
            # Note: dones may still be None (if obs is first obs)
            assert (
                self.observations is not None
                and self.actions is not None
                and self.context is not None
                and self.next_observations is not None
                and self.next_terminated is not None
            ), "All transition values must be filled."
            if self.dones is not None:
                dones = self.dones.view(-1, 1)
            else:
                dones = None
            return ReplayBuffer.FullTransition(
                self.observations,
                self.actions,
                dones,
                self.context,
                self.next_observations,
                self.next_terminated.view(-1, 1),
            )

    @dataclass(frozen=True)
    class FullTransition:
        observations: TensorDict
        actions: torch.Tensor
        dones: torch.Tensor | None
        context: torch.Tensor
        next_observations: TensorDict
        next_terminated: torch.Tensor

    def __init__(
        self,
        training_type: Literal["rl"],
        num_envs: int,
        capacity_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        z_dim: int,
        batch_size: int,
        device: str = "cpu",
    ):
        # store inputs
        self.training_type: Literal["rl"] = training_type
        self.num_envs = num_envs
        self.capacity = capacity_per_env * num_envs
        self.device = device

        # Core
        # We only take value.shape[1:] to ignore num_envs dimension
        self.observations = TensorDict(
            {key: torch.zeros(self.capacity, *value.shape[1:], device=device) for key, value in obs.items()},
            batch_size=self.capacity,
            device=self.device,
        )
        self.actions = torch.zeros(self.capacity, *actions_shape, device=self.device)
        self.context = torch.zeros(self.capacity, z_dim, device=self.device)
        self.next_observations: TensorDict = self.observations.clone()
        self.next_terminated = torch.zeros(self.capacity, 1, device=self.device).byte()
        self.gammas = torch.zeros(self.capacity, 1, device=self.device)

        # counter for the number of transitions stored
        self._curr_idx = 0
        self._is_full = False

        # tensor for selecting mini batches
        self._indices = torch.zeros(batch_size, device=self.device, dtype=torch.long)

    def __len__(self) -> int:
        return self.capacity if self._is_full else self._curr_idx

    def add_transitions(self, transition: Transition):
        # ensure all fields are full
        full_transition = transition.to_full()
        # only include transitions that haven't terminated (otherwise next_obs is state after reset)
        # if dones is None, all are valid
        if full_transition.dones is None:
            valid_idxs = torch.arange(self.num_envs, device=self.device)
        else:
            valid_idxs = torch.argwhere(~full_transition.dones.view(-1)).flatten()
        n_valid = len(valid_idxs)
        if n_valid == 0:
            return
        buf_idxs = (torch.arange(0, n_valid, device=self.device) + self._curr_idx) % self.capacity

        assert n_valid < self.capacity, (
            f"Cannot store {valid_idxs.shape[0]} transitions in replay buffer of size {self.capacity}. "
            "You may need to increase buffer capacity or decrease num_envs."
        )

        self.observations.update_at_(full_transition.observations[valid_idxs].to(self.device), buf_idxs)
        self.next_observations.update_at_(full_transition.next_observations[valid_idxs].to(self.device), buf_idxs)
        self.actions.index_copy_(0, buf_idxs, full_transition.actions[valid_idxs].to(self.device))
        self.context.index_copy_(0, buf_idxs, full_transition.context[valid_idxs].to(self.device))
        self.next_terminated.index_copy_(0, buf_idxs, full_transition.next_terminated[valid_idxs].to(self.device))

        # increment the counter
        self._curr_idx += n_valid
        if self._curr_idx >= self.capacity:
            self._is_full = True
            self._curr_idx -= self.capacity

    def compute_returns(self, gamma: float) -> None:
        self.gammas = gamma * (1 - self.next_terminated).float()

    def sample_mini_batch(self, device: str) -> tuple[TensorDict, torch.Tensor, torch.Tensor, TensorDict, torch.Tensor]:
        # sample indices from uniform distribution
        self._indices.random_(0, len(self))

        return (
            self.observations[self._indices].to(device),
            self.actions[self._indices].to(device),
            self.context[self._indices].to(device),
            self.next_observations[self._indices].to(device),
            self.gammas[self._indices].to(device),
        )
