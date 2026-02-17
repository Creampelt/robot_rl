from typing import Iterator
from random import randint

import torch
from tensordict import TensorDict, cat as td_cat

from robot_rl.utils import get_obs
from .expert_buffer import ExpertBuffer


class TrajectoryBuffer(ExpertBuffer):
    def __init__(
        self,
        motion_paths: list[str],
        bucket_size: int,
        expert_obs_groups: list[str],
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.bucket_size = bucket_size
        self.obs_groups = expert_obs_groups

        self.motions = self._parse_motion_paths(motion_paths, device)  # num_motions x bucket_size
        self.num_motions = self.motions.shape[0]
        self.priorities = torch.ones((self.motions.shape[0],), device=device)

    def sample(self, batch_size: int, device: str | None = None) -> tuple[TensorDict, TensorDict]:
        """Sample current and next expert observations from multinomial distribution weighted by priorities (see
        `update_priorities`).

        Args:
            batch_size: The batch size to sample.
            device: The device to move the output to. Defaults to None, which keeps the observations on the buffer's
                device.

        Returns:
            A tuple containing the expert obs and next obs as TensorDicts. Shape is (batch_size).
        """
        # sample episodes according to priorities
        ep_indices = torch.multinomial(self.priorities, batch_size, replacement=True)
        # uniformly sample from sequence
        seq_indices = torch.randint(0, self.bucket_size - 1, (batch_size,), device=self.device)
        return (
            self.motions[ep_indices, seq_indices].to(device),
            self.motions[ep_indices, seq_indices + 1].to(device),
        )

    def sample_states(self, num_envs: int) -> torch.Tensor:
        """Sample states (concatenated root_pose, root_vel, joint_pos, joint_vel) for a vectorized environment.
        Returns a batched tensor of shape (num_envs, state_dim)."""
        ep_indices = torch.multinomial(self.priorities, num_envs, replacement=True)
        motion_indices = torch.randint(0, self.bucket_size, (num_envs,), device=self.device)
        motions = self.motions[ep_indices, motion_indices]
        return self.get_expert_obs(motions)

    def get_batch_motions(
        self,
        mini_batch_size: int,
        device: str | None = None,
    ) -> Iterator[TensorDict]:
        """Sample entire motion trajectories in mini batches. Returns iterator containing batched observations as
        TensorDict with shape (mini_batch_size, bucket_size, *obs_size). Note that the final batch may be truncated."""
        for idx in range(0, self.num_motions, mini_batch_size):
            motions = self.motions[idx : idx + mini_batch_size].to(device)
            yield motions

    def update_priorities(self, priorities: torch.Tensor, indices: torch.Tensor | slice) -> None:
        self.priorities[indices] = priorities.to(self.device)

    def normalize_priorities(self) -> None:
        self.priorities /= self.priorities.sum()

    def get_expert_obs(self, obs: TensorDict) -> torch.Tensor:
        return get_obs(obs, self.obs_groups)

    def _parse_motion_paths(self, motion_paths: list[str], device: str) -> TensorDict:
        motions: list[TensorDict] = []
        motion_lengths = torch.zeros(len(motion_paths), dtype=torch.int)
        for i, motion_path in enumerate(motion_paths):
            # disable weights so we can load tensordict
            motion = torch.load(motion_path, weights_only=False)
            assert isinstance(motion, TensorDict), f"Invalid obs type {type(motion)}, expected TensorDict"
            # if motion is not evenly divisible by sequence length, we take the largest divisible section and randomly
            # align the window
            remainder = motion.shape[0] % self.bucket_size
            start_idx = randint(0, remainder)
            end_idx = start_idx + motion.shape[0] - remainder
            motions.append(motion[start_idx:end_idx].reshape(-1, self.bucket_size))
            motion_lengths[i] = motion.shape[0]
        return td_cat(motions, dim=0).to(device)  # type: ignore
