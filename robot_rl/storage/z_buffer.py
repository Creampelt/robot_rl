import torch


class ZBuffer:
    def __init__(
        self,
        capacity: int,
        z_dim: int,
        device: str = "cpu",
    ):
        # store inputs
        self.capacity = capacity
        self.device = device

        # storage tensor
        self.z = torch.zeros(self.capacity, z_dim, device=self.device)

        # counter for the number of zs stored
        self._curr_idx = 0
        self._is_full = False

    def __len__(self) -> int:
        return self.capacity if self._is_full else self._curr_idx

    def add(self, z: torch.Tensor) -> None:
        buf_idxs = (torch.arange(0, z.shape[0], device=self.device) + self._curr_idx) % self.capacity
        self.z.index_copy_(0, buf_idxs, z.to(self.device))

        # increment the counter
        self._curr_idx += z.shape[0]
        if self._curr_idx >= self.capacity:
            self._is_full = True
            self._curr_idx -= self.capacity

    def sample(self, num_envs: int, device: str | None = None) -> torch.Tensor:
        indices = torch.randint(0, len(self), (num_envs,), device=self.device)
        return self.z[indices].to(device)
