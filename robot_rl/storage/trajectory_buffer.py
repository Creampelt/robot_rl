import torch
from tensordict import TensorDict, cat as td_cat


class TrajectoryBuffer:
    def __init__(self, motion_paths: list[str], batch_size: int, device: str = "cpu") -> None:
        self.device = device
        self.batch_size = batch_size
        self.num_motions = len(motion_paths)

        self.motion_lengths, self.motions = self._parse_motion_paths(motion_paths, device)
        self.motion_starts = torch.cumsum(self.motion_lengths, dim=0) - self.motion_lengths[0]
        self.priorities = torch.zeros((self.num_motions,), device=device)

    def sample(self) -> tuple[TensorDict, TensorDict]:
        ep_indices = torch.multinomial(self.priorities, self.batch_size, replacement=True)
        ep_starts = self.motion_starts[ep_indices]
        ep_lengths = self.motion_lengths[ep_indices] - 1  # subtract 1 from motion_lengths so there is always a next_obs
        obs_indices = torch.floor(torch.rand_like(ep_lengths.float()) * ep_lengths.float() + ep_starts).long()
        return self.motions[obs_indices], self.motions[obs_indices + 1]

    def update_priorities(self, priorities: torch.Tensor, indices: torch.Tensor) -> None:
        self.priorities[indices] = priorities
        self.priorities /= self.priorities.sum()

    def _parse_motion_paths(self, motion_paths: list[str], device: str) -> tuple[torch.Tensor, TensorDict]:
        motions: list[TensorDict] = []
        motion_lengths = torch.zeros(len(motion_paths), dtype=torch.int)
        for i, motion_path in enumerate(motion_paths):
            motion = torch.load(motion_path)
            assert isinstance(motion, TensorDict), f"Invalid obs type {type(motion)}, expected TensorDict"
            motions.append(motion)
            motion_lengths[i] = motion.shape[0]
        return motion_lengths.to(device), td_cat(motions, dim=0).to(device)  # type: ignore
