from typing import Iterator
import torch
import tensordict
from tensordict import TensorDict, cat as td_cat
from random import randint

SAFE_GLOBALS = [tensordict]


class TrajectoryBuffer:
    def __init__(self, motion_paths: list[str], batch_size: int, sequence_length: int, device: str = "cpu") -> None:
        self.device = device
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self.motions = self._parse_motion_paths(motion_paths, device)  # num_motions x sequence_length
        self.num_motions = self.motions.shape[0]
        self.priorities = torch.zeros((self.motions.shape[0],), device=device)

    def sample(self, device: str | None = None) -> tuple[TensorDict, TensorDict]:
        """Sample expert_obs and expert_next_obs with sequences weighted by priorities (see `update_priorities`).
        Observations within each sequence are sampled uniformly. Returns (obs, next_obs)."""
        ep_indices = torch.multinomial(self.priorities, self.batch_size, replacement=True)
        # subtract 1 from sequence_length so there is always a next_obs
        seq_indices = torch.randint(0, self.sequence_length - 1, (self.batch_size,))
        return self.motions[ep_indices, seq_indices].to(device), self.motions[ep_indices, seq_indices + 1].to(device)

    def get_batch_motions(self, mini_batch_size: int, device: str | None = None) -> Iterator[tuple[int, TensorDict]]:
        """Sample entire motion trajectories in mini batches. Returns generator containing obs sequence as TensorDict
        with batch shape (mini_batch_size, sequence_length), except for the last mini-batch (which may be truncated)."""
        for idx in range(0, self.num_motions, mini_batch_size):
            yield idx, self.motions[idx : idx + mini_batch_size].to(device)

    def update_priorities(self, priorities: torch.Tensor, indices: torch.Tensor | slice) -> None:
        self.priorities[indices] = priorities.to(self.device)
        self.priorities /= self.priorities.sum()

    def _parse_motion_paths(self, motion_paths: list[str], device: str) -> TensorDict:
        motions: list[TensorDict] = []
        motion_lengths = torch.zeros(len(motion_paths), dtype=torch.int)
        for i, motion_path in enumerate(motion_paths):
            # disable weights so we can load tensordict
            motion = torch.load(motion_path, weights_only=False)
            assert isinstance(motion, TensorDict), f"Invalid obs type {type(motion)}, expected TensorDict"
            # if motion is not evenly divisible by sequence length, we take the largest divisible section and randomly
            # align the window
            remainder = motion.shape[0] % self.sequence_length
            start_idx = randint(0, remainder)
            end_idx = start_idx + motion.shape[0] - remainder
            motions.append(motion[start_idx:end_idx].reshape(-1, self.sequence_length))
            motion_lengths[i] = motion.shape[0]
        return td_cat(motions, dim=0).to(device)  # type: ignore
