from collections.abc import Iterator

import torch
from tensordict import TensorDict

from robot_rl.utils import get_obs

from .expert_buffer import ExpertBuffer


class TrajectoryBuffer(ExpertBuffer):
    def __init__(
        self,
        motion_path: str,
        expert_obs_groups: list[str],
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.obs_groups = expert_obs_groups

        # motions file should be obs tensordict with batch shape (num_motions, bucket_size)
        self.motions = torch.load(motion_path, weights_only=False).to(device)
        assert len(self.motions.shape) == 2
        self.num_motions, self.bucket_size = self.motions.shape
        self.priorities = torch.ones((self.motions.shape[0],), device=device)

        self._eval_order = torch.arange(0, self.num_motions, device=self.device)

        print(f"[INFO] Successfully loaded {self.num_motions} motions with length {self.bucket_size}.")

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
        # uniformly sample from sequence (exclude last so there will always be a next obs)
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
        # randomize order since rigid body DR is fixed per-environment
        self._eval_order = torch.randperm(self.num_motions, device=self.device)
        for idx in range(0, self.num_motions, mini_batch_size):
            eval_idxs = self._eval_order[idx : idx + mini_batch_size]
            motions = self.motions[eval_idxs].to(device)
            yield motions

    def update_priorities(self, priorities: torch.Tensor, indices: torch.Tensor | slice) -> None:
        actual_indices = self._eval_order[indices]
        self.priorities[actual_indices] = priorities.to(self.device)

    def normalize_priorities(self) -> None:
        self.priorities /= self.priorities.sum()

    def get_expert_obs(self, obs: TensorDict) -> torch.Tensor:
        return get_obs(obs, self.obs_groups)
