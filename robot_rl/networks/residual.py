import torch
import torch.nn as nn

from robot_rl.utils import resolve_nn_activation

from .parallel import ParallelLinear, ParallelLayerNorm


class Block(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_parallel: int = 1, activation: str = "mish") -> None:
        super().__init__()
        layers = [
            ParallelLayerNorm(input_dim, num_parallel),
            ParallelLinear(input_dim, output_dim, num_parallel),
            resolve_nn_activation(activation),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.model(x)


class ResidualBlock(Block):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.model(x)
