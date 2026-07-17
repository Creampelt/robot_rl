# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict


class ReplayBuffer:
    """Storage for the data collected across rollouts.

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
            effective_n_steps: torch.Tensor | None = None,
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
            """Batch of terminated flags after the step (true termination only)."""

            self.effective_n_steps: torch.Tensor | None = effective_n_steps
            """Per-sample number of steps actually aggregated (<= n_steps; 1-step or capped at an episode end)."""

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
        n_steps: int = 1,
        gamma: float = 0.99,
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
            n_steps: Number of steps for n-step returns. ``1`` (default) is single-step. ``>1`` requires
                ``keep_terminal=True`` (n-step needs the per-env temporal sequence that only the keep-all layout
                preserves -- every ``add_transitions`` then writes exactly ``num_envs`` transitions, so a stored
                index ``i`` maps to env ``i % num_envs``, row ``i // num_envs``, and env ``e``'s next step is at
                ``i + num_envs``). ``sample_mini_batch`` then aggregates the discounted return, stopping at
                episode ends, and returns ``effective_n_steps`` per sample.
            gamma: Discount factor used for the n-step return (unused for ``n_steps == 1``).
        """
        # store inputs
        self.num_envs = num_envs
        self.capacity_per_env = capacity_per_env
        self.capacity = capacity_per_env * num_envs
        self.device = device
        self.keep_terminal = keep_terminal
        self.n_steps = n_steps
        self.gamma = gamma
        if n_steps > 1 and not keep_terminal:
            raise ValueError("n_steps > 1 requires keep_terminal=True (n-step needs the per-env time sequence).")

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
        self.dones = torch.zeros(self.capacity, 1, device=self.device).byte()  # episode ends (n-step boundaries)
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
        # Episode-end flags (for n-step boundaries); zeros when the caller doesn't provide dones.
        dones_src = transition.dones if transition.dones is not None else torch.zeros(self.num_envs, device=self.device)
        self.dones.index_copy_(0, buf_idxs, dones_src.view(-1)[valid_idxs].byte().to(self.device).unsqueeze(-1))

        # increment the counter
        self._curr_idx += num_valid
        if self._curr_idx >= self.capacity:
            self._is_full = True
            self._curr_idx -= self.capacity

    def sample_mini_batch(self, device: str | None = None) -> Batch:
        """Randomly sample a mini-batch from the replay buffer (with n-step aggregation when ``n_steps > 1``)."""
        if self.n_steps > 1:
            nstep = self._sample_nstep(device)
            if nstep is not None:
                return nstep
            # not enough consecutive data for a full n-step yet -> fall through to single-step sampling

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

    def _sample_nstep(self, device: str | None = None) -> Batch | None:
        """Sample a mini-batch with n-step returns over the per-env time sequence (keep_terminal layout).

        A stored index is ``row * num_envs + env``, so env ``e``'s consecutive steps are ``num_envs`` apart. Start
        indices whose n-step window would cross the circular write head are excluded; the discounted return is
        summed until the first episode end, and ``effective_n_steps`` records how many steps were aggregated.
        Returns ``None`` if there is not yet a full n-step window anywhere in the buffer.
        """
        batch_size = self._indices.shape[0]
        cap_rows = self.capacity_per_env
        filled_rows = cap_rows if self._is_full else self._curr_idx // self.num_envs
        write_row = self._curr_idx // self.num_envs
        max_offset = self.n_steps - 1

        # Valid start rows: their n-step window must not cross the write head (same mask for every env).
        rows = torch.arange(filled_rows, device=self.device)
        if self._is_full:
            before = rows < write_row
            safe = torch.where(before, (rows + max_offset) < write_row, (rows + max_offset) < (cap_rows + write_row))
        else:
            safe = (rows + max_offset) < filled_rows
        valid_rows = rows[safe]
        if valid_rows.numel() == 0:
            return None

        # Sample (start row, env) pairs and build the n consecutive flat indices per sample.
        start_rows = valid_rows[torch.randint(valid_rows.numel(), (batch_size,), device=self.device)]
        envs = torch.randint(self.num_envs, (batch_size,), device=self.device)
        start_flat = start_rows * self.num_envs + envs
        offsets = torch.arange(self.n_steps, device=self.device)
        step_rows = (start_rows.unsqueeze(-1) + offsets) % cap_rows  # [B, n]
        step_flat = step_rows * self.num_envs + envs.unsqueeze(-1)  # [B, n]

        all_rewards = self.rewards[step_flat]  # [B, n]
        all_dones = self.dones[step_flat].squeeze(-1).float()  # [B, n]

        # Mask that is 1 up to and including the first episode end, 0 afterwards; discounted n-step return.
        dones_shifted = torch.cat([torch.zeros_like(all_dones[..., :1]), all_dones[..., :-1]], dim=-1)
        done_masks = torch.cumprod(1.0 - dones_shifted, dim=-1)
        discounts = torch.pow(torch.tensor(self.gamma, device=self.device), offsets.float())
        n_step_rewards = (all_rewards * done_masks * discounts.view(1, -1)).sum(dim=-1)  # [B]

        # First episode end within the window -> effective horizon + the index whose next-state we bootstrap from.
        first_done = torch.argmax((all_dones > 0).float(), dim=-1)
        no_dones = all_dones.sum(dim=-1) == 0
        first_done = torch.where(no_dones, torch.full_like(first_done, self.n_steps - 1), first_done)
        effective_n = (first_done + 1).unsqueeze(-1).long()  # [B, 1]
        final_flat = step_flat.gather(1, first_done.unsqueeze(-1)).squeeze(-1)  # [B]

        return ReplayBuffer.Batch(
            self.observations[start_flat].to(device),
            self.next_observations[final_flat].to(device),
            self.actions[start_flat].to(device),
            n_step_rewards.to(device),
            self.gammas[start_flat].to(device),
            self.context[start_flat].to(device),
            self.next_terminated[final_flat].to(device),
            effective_n.to(device),
        )
