from __future__ import annotations

import torch
from tensordict import TensorDict

from .replay_buffer import ReplayBuffer


class OfflineTransitionDataset:
    """A fixed set of transitions read from disk, sampled like a replay buffer.

    Expects a file holding a dict with ``observations`` and ``next_observations`` (each a TensorDict of
    observation groups, first dimension the transition index), ``actions``, and optionally ``rewards`` and
    ``terminated``. Latent context is not stored: forward-backward relabels it every step, so it is
    allocated as zeros and overwritten by the algorithm.
    """

    def __init__(
        self,
        path: str,
        z_dim: int,
        batch_size: int,
        storage_device: str = "cpu",
        gamma: float = 0.99,
        device: str | None = None,
    ) -> None:
        """Load the dataset and allocate the sampling scratch.

        Args:
            path: File written by the dataset builder.
            z_dim: Latent-context width to allocate.
            batch_size: Mini-batch size returned by :meth:`sample_mini_batch`.
            storage_device: Device holding the transitions; ``"cpu"`` keeps a large set out of VRAM.
            gamma: Discount written into every sampled transition that did not terminate.
            device: Unused; accepted so the constructor matches the buffer's call signature.
        """
        del device
        data = torch.load(path, weights_only=False)
        self.observations: TensorDict = data["observations"].to(storage_device)
        self.next_observations: TensorDict = data["next_observations"].to(storage_device)
        self.actions: torch.Tensor = data["actions"].to(storage_device)
        num = self.actions.shape[0]
        if self.observations.shape[0] != num or self.next_observations.shape[0] != num:
            raise ValueError(
                f"Transition counts disagree: actions {num}, observations {self.observations.shape[0]},"
                f" next_observations {self.next_observations.shape[0]}."
            )
        self.rewards = data.get("rewards", torch.zeros(num, 1)).to(storage_device).view(num, 1)
        terminated = data.get("terminated", torch.zeros(num, 1, dtype=torch.bool))
        self.next_terminated = terminated.to(storage_device).view(num, 1).bool()
        self.gammas = torch.where(self.next_terminated, 0.0, gamma).to(storage_device).float()
        self.context = torch.zeros(num, z_dim, device=storage_device)
        self.batch_size = batch_size
        self.obs_groups = sorted(self.observations.keys())
        self._indices = torch.zeros(batch_size, dtype=torch.long, device=storage_device)
        print(f"[INFO] Loaded {num} offline transitions with groups {self.obs_groups}.")

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return self.actions.shape[0]

    def sample_mini_batch(self, device: str | None = None) -> ReplayBuffer.Batch:
        """Draw a uniform mini-batch, moved to ``device``."""
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
