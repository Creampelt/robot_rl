import torch


class ZBuffer:
    def __init__(self, capacity: int, dim: int, device: str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        self._storage = torch.zeros((capacity, dim), device=device, dtype=dtype)
        self._idx: int = 0
        self._is_full: bool = False
        self.capacity = capacity
        self.device = device

    def __len__(self) -> int:
        return self.capacity if self._is_full else self._idx

    def empty(self) -> bool:
        return self._idx == 0 and not self._is_full

    def add(self, data: torch.Tensor) -> None:
        N = data.shape[0]
        if self._idx + N >= self.capacity:
            free = self.capacity - self._idx
            self._storage[self._idx :] = data[:free]
            self._storage[: N - free] = data[free:]
            self._is_full = True
        else:
            self._storage[self._idx : self._idx + N] = data
        self._idx = (self._idx + data.shape[0]) % self.capacity

    def sample(self, num: int, device: str | None = None) -> torch.Tensor:
        if device is None:
            device = self.device
        idx = torch.randint(0, len(self), (num,))
        return self._storage[idx].clone().to(device)
