from typing import Sequence

import torch
import torch.nn as nn

from robot_rl.utils import resolve_nn_activation


class Discriminator(nn.Module):
    def __init__(self, obs_dim: int, z_dim: int, hidden_layers: Sequence[int], activation: str = "relu") -> None:
        assert hidden_layers, "Must have at least one hidden layer."
        super().__init__()

        layers = [nn.Linear(obs_dim + z_dim, hidden_layers[0]), nn.LayerNorm(hidden_layers[0]), nn.Tanh()]
        for i, dim in enumerate(hidden_layers[:-1]):
            layers.append(nn.Linear(dim, hidden_layers[i + 1]))
            layers.append(resolve_nn_activation(activation))
        layers.append(nn.Linear(hidden_layers[-1], 1))
        self.trunk = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        s = self.compute_logits(obs, z)
        return torch.sigmoid(s)

    def compute_logits(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, obs], dim=1)
        return self.trunk(x)

    def compute_reward(self, obs: torch.Tensor, z: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        s = self.forward(obs, z)
        s = torch.clamp(s, eps, 1 - eps)
        return s.log() - (1 - s).log()
