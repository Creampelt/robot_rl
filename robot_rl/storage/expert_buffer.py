import torch
from abc import ABC, abstractmethod


class ExpertBuffer(ABC):
    """Abstract expert buffer class that can be sampled from at runtime."""

    @abstractmethod
    def sample_states(self, num_envs: int) -> dict[str, torch.Tensor]:
        """Sample states for a vectorized environment. Returns a state dictionary."""
        raise NotImplementedError
