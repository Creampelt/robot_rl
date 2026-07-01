# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict


class ReplayBuffer:
    """Storage for the data collected across rollouts (for off-policy RL).

    The replay storage is populated by adding transitions during the rollout phase.
    """

    class Transition:
        """Storage for a single state transition.

        This class is populated incrementally during the rollout phase and then passed to
        :meth:`ReplayBuffer.add_transition` to record the data.
        """

        def __init__(self) -> None:
            """Initialize an empty transition container."""
            self.observations: TensorDict | None = None
            """Observations at the current step."""

            self.actions: torch.Tensor | None = None
            """Actions taken at the current step."""

            self.rewards: torch.Tensor | None = None
            """Rewards received after the action."""

            self.dones: torch.Tensor | None = None
            """Done flags indicating episode termination or timeout at the current step."""

            self.context: torch.Tensor | None = None
            """Latent context (z) vectors at the current step."""

            self.next_observations: TensorDict | None = None
            """Observations after the current step."""

            self.next_terminated: torch.Tensor | None = None
            """Done flags indicating episode termination after the current step."""

        def clear(self) -> None:
            """Reset all transition fields to None."""
            self.__init__()

    class Batch:
        """A batch of data yielded by the replay buffer.

        This class provides named access to mini-batch fields.
        """

        def __init__(
            self,
            observations: TensorDict,
            next_observations: TensorDict,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            gammas: torch.Tensor,
            context: torch.Tensor,
            next_terminated: torch.Tensor | None = None,
        ) -> None:
            """Initialize a batch container over rollout data."""
            self.observations: TensorDict = observations
            """Batch of observations."""

            self.next_observations: TensorDict = next_observations
            """Batch of next observations."""

            self.actions: torch.Tensor = actions
            """Batch of actions."""

            self.rewards: torch.Tensor = rewards
            """Batch of rewards."""

            self.gammas: torch.Tensor = gammas
            """Batch of gammas."""

            self.context: torch.Tensor = context
            """Batch of latent context (z) vectors."""

            self.next_terminated: torch.Tensor | None = next_terminated
            """Batch of terminated flags after the step (true termination only; used by SAC to gate the bootstrap)."""

    def __init__(
        self,
        num_envs: int,
        capacity_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        z_dim: int,
        batch_size: int,
        device: str = "cpu",
        keep_terminal: bool = False,
    ) -> None:
        """Initialize the buffer storage.

        Args:
            num_envs: Number of parallel environments feeding the buffer.
            capacity_per_env: Stored transitions per environment (total capacity = ``capacity_per_env * num_envs``).
            obs: A representative observation TensorDict used to size the obs/next-obs storage.
            actions_shape: Shape of a single environment's action.
            z_dim: Latent-context dimension (use ``0`` for algorithms without a latent, e.g. SAC).
            batch_size: Mini-batch size returned by :meth:`sample_mini_batch`.
            device: Storage device.
            keep_terminal: If ``False`` (default; FbCpr) transitions whose ``dones`` is set are dropped, since the
                stored next-obs would be a post-reset state. If ``True`` (SAC) *all* transitions are kept -- the
                caller must set ``next_observations`` to the true pre-reset next-obs (via ``time_outs_obs``) and
                ``next_terminated`` to gate the bootstrap.
        """
        # store inputs
        self.num_envs = num_envs
        self.capacity = capacity_per_env * num_envs
        self.device = device
        self.keep_terminal = keep_terminal

        # Core
        # We only take value.shape[1:] to ignore num_envs dimension
        self.observations = TensorDict(
            {
                key: torch.zeros(self.capacity, *value.shape[1:], dtype=value.dtype, device=device)
                for key, value in obs.items()
            },
            batch_size=self.capacity,
            device=self.device,
        )
        self.actions = torch.zeros(self.capacity, *actions_shape, device=self.device)
        self.rewards = torch.zeros(self.capacity, device=self.device)
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
        """Get the total number of transitions currently stored in the buffer."""
        return self.capacity if self._is_full else self._curr_idx

    def add_transitions(self, transition: Transition) -> None:
        """Add a transition to the buffer."""
        # Drop transitions whose next_obs is a post-reset state (dones set), unless keep_terminal is set (SAC,
        # where next_obs is the true pre-reset obs). If dones is None, all are valid.
        if transition.dones is None or self.keep_terminal:
            valid_idxs = torch.arange(self.num_envs, device=self.device)
        else:
            valid_idxs = torch.argwhere(~transition.dones.view(-1)).flatten()
        num_valid = len(valid_idxs)
        # Exit if no transitions are valid to store
        if num_valid == 0:
            return
        # Raise error if buffer is too small to store transitions
        if num_valid >= self.capacity:
            raise RuntimeError(
                f"Cannot store {num_valid} transitions in replay buffer of size {self.capacity}. "
                "You may need to increase buffer capacity or decrease num_envs."
            )

        buf_idxs = (torch.arange(0, num_valid, device=self.device) + self._curr_idx) % self.capacity
        self.observations.update_at_(transition.observations[valid_idxs].to(self.device), buf_idxs)  # type: ignore
        self.next_observations.update_at_(transition.next_observations[valid_idxs].to(self.device), buf_idxs)  # type: ignore
        self.actions.index_copy_(0, buf_idxs, transition.actions[valid_idxs].to(self.device))  # type: ignore
        self.rewards.index_copy_(0, buf_idxs, transition.rewards[valid_idxs].to(self.device))  # type: ignore
        self.context.index_copy_(0, buf_idxs, transition.context[valid_idxs].to(self.device))  # type: ignore
        self.next_terminated.index_copy_(
            0, buf_idxs, transition.next_terminated[valid_idxs].to(self.device).unsqueeze(-1)
        )  # type: ignore

        # increment the counter
        self._curr_idx += num_valid
        if self._curr_idx >= self.capacity:
            self._is_full = True
            self._curr_idx -= self.capacity

    def sample_mini_batch(self, device: str | None = None) -> Batch:
        """Randomly sample a mini-batch from the replay buffer."""
        # Sample indices from uniform distribution
        self._indices.random_(0, len(self))

        return ReplayBuffer.Batch(
            self.observations[self._indices].to(device),
            self.next_observations[self._indices].to(device),
            self.actions[self._indices].to(device),
            self.rewards[self._indices].to(device),
            self.gammas[self._indices].to(device),
            self.context[self._indices].to(device),
            self.next_terminated[self._indices].to(device),
        )
