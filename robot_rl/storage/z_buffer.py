import torch


class ZBuffer:
    """Buffer to store latent z vectors."""

    def __init__(self, capacity: int, z_dim: int, device: str = "cpu") -> None:
        """Initialize the buffer."""
        # store inputs
        self.capacity = capacity
        self.device = device

        # storage tensor
        self.z = torch.zeros(self.capacity, z_dim, device=self.device)

        # counter for the number of zs stored
        self._curr_idx = 0
        self._is_full = False

    def __len__(self) -> int:
        """Get the number of elements in the buffer."""
        return self.capacity if self._is_full else self._curr_idx

    def add(self, z: torch.Tensor) -> None:
        """Add a batch of z values to the buffer."""
        buf_idxs = (torch.arange(0, z.shape[0], device=self.device) + self._curr_idx) % self.capacity
        self.z.index_copy_(0, buf_idxs, z.to(self.device))

        # increment the counter
        self._curr_idx += z.shape[0]
        if self._curr_idx >= self.capacity:
            self._is_full = True
            self._curr_idx -= self.capacity

    def sample(self, num_envs: int, device: str | None = None) -> torch.Tensor:
        """Sample z values from the buffer for all environments."""
        indices = torch.randint(0, len(self), (num_envs,), device=self.device)
        return self.z[indices].to(device)

    def state_dict(self) -> dict:
        """Return the buffer contents and fill state for checkpointing."""
        return {"z": self.z, "curr_idx": self._curr_idx, "is_full": self._is_full}

    def load_state_dict(self, state: dict) -> None:
        """Restore the buffer contents and fill state from a checkpoint."""
        self.z = state["z"].to(self.device)
        self.capacity = self.z.shape[0]
        self._curr_idx = int(state["curr_idx"])
        self._is_full = bool(state["is_full"])
