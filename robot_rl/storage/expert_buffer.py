from abc import ABC, abstractmethod
from typing import Iterator
import torch


class ExpertBuffer(ABC):
    """Abstract expert buffer class that can be sampled from at runtime."""

    @abstractmethod
    def sample_states(self, num_envs: int) -> torch.Tensor:
        """Sample states (concatenated root_pose, root_vel, joint_pos, joint_vel) for a vectorized environment.
        Returns a batched tensor of shape (num_envs, state_dim)."""
        raise NotImplementedError
