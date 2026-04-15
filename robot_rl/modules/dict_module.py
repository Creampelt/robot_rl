from __future__ import annotations

import torch.nn as nn
from tensordict import TensorDict
from typing import Generic, TypeVar

T = TypeVar("T", bound=nn.Module)


class DictModule(nn.Module, Generic[T]):
    """Applies independent module instances to each key in a TensorDict.

    Each key gets its own module instance, stored in an ``nn.ModuleDict``. Keys present in the input TensorDict
    but not in the module dict are passed through unchanged.

    Example::

        normalizer: DictModule[nn.BatchNorm1d] = DictModule({
            "obs": nn.BatchNorm1d(64, affine=False),
            "state": nn.BatchNorm1d(32, affine=False),
        })
        normalized_obs = normalizer(obs_tensordict)
    """

    modules_dict: nn.ModuleDict

    def __init__(self, modules: dict[str, T]) -> None:
        """Initialize the module dict with one sub-module per key."""
        super().__init__()
        self.keys = list(modules.keys())
        self.modules_dict = nn.ModuleDict(modules)

    def forward(self, obs: TensorDict) -> TensorDict:
        """Apply each sub-module to its corresponding key in ``obs`` and return the result."""
        result = obs.clone()
        for key in self.keys:
            if key in result:
                result[key] = self.modules_dict[key](result[key])
        return result
