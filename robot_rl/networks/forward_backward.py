from typing import Sequence
import torch
import torch.nn as nn

from robot_rl.utils import resolve_nn_activation
from .parallel import ParallelLinear, ParallelLayerNorm
from .residual import Block, ResidualBlock


class Embedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = [],
        num_parallel: int = 1,
        activation: str | None = None,
        first_activation: str | None = None,
    ) -> None:
        super().__init__()
        layers = self._make_layers(input_dim, output_dim, hidden_dims, num_parallel, activation, first_activation)
        self.model = nn.Sequential(*layers)

    def _make_layers(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        num_parallel: int,
        activation: str | None,
        first_activation: str | None,
    ) -> list[nn.Module]:
        # set defaults
        if activation is None:
            activation = "relu"
        if first_activation is None:
            first_activation = "tanh"

        # construct layers for model
        layers = [
            ParallelLinear(input_dim, hidden_dims[0], num_parallel),
            ParallelLayerNorm(hidden_dims[0], num_parallel),
            resolve_nn_activation(first_activation),
        ]
        layer_dims = list(hidden_dims) + [output_dim]
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(ParallelLinear(in_dim, out_dim, num_parallel))
            layers.append(resolve_nn_activation(activation))
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResidualEmbedding(Embedding):
    def _make_layers(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        num_parallel: int,
        activation: str | None,
        first_activation: str | None,
    ) -> list[nn.Module]:
        # set defaults
        if activation is None:
            activation = "mish"
        if first_activation is None:
            first_activation = "mish"

        # construct layers for model
        layers: list[nn.Module] = [Block(input_dim, hidden_dims[0], num_parallel, activation)]
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(ResidualBlock(in_dim, out_dim, num_parallel, activation))
        layers.append(Block(hidden_dims[-1], output_dim, num_parallel, activation))
        return layers


class EmbeddedNet(nn.Module):
    def __init__(
        self,
        input_dim: Sequence[int],
        output_dim: int,
        embedding_dims: Sequence[int],
        hidden_dims: Sequence[int],
        num_parallel: int = 1,
        activation: str | None = None,
        first_activation: str | None = None,
        last_activation: str | nn.Module = "identity",
    ) -> None:
        super().__init__()
        assert hidden_dims and embedding_dims, "hidden_dims and embedding_dims must both have at least one layer."
        self.num_parallel = num_parallel
        self.num_embeddings = len(input_dim)

        embedding_layers, layers = self._make_layers(
            input_dim,
            output_dim,
            embedding_dims,
            hidden_dims,
            num_parallel,
            activation,
            first_activation,
        )
        if isinstance(last_activation, str):
            layers.append(resolve_nn_activation(last_activation))
        else:
            layers.append(last_activation)

        self.embeddings = nn.ModuleList(embedding_layers)
        self.model = nn.Sequential(*layers)

    def _make_layers(
        self,
        input_dim: Sequence[int],
        output_dim: int,
        embedding_dims: Sequence[int],
        hidden_dims: Sequence[int],
        num_parallel: int,
        activation: str | None,
        first_activation: str | None,
    ) -> tuple[list[nn.Module], list[nn.Module]]:
        # set defaults
        if activation is None:
            activation = "relu"
        if first_activation is None:
            first_activation = "tanh"

        # construct modules for each embedding term
        embedding_layers: list[nn.Module] = [
            Embedding(
                dim,
                hidden_dims[0] // self.num_embeddings,
                embedding_dims,
                num_parallel,
                activation,
                first_activation,
            )
            for dim in input_dim
        ]

        # construct layers for main model
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(ParallelLinear(in_dim, out_dim, num_parallel))
            layers.append(resolve_nn_activation(activation))
        layers.append(ParallelLinear(hidden_dims[-1], output_dim, num_parallel))

        return embedding_layers, layers

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        assert len(args) == self.num_embeddings, (
            f"Invalid number of inputs received. Expected {self.num_embeddings}, but received {len(args)}"
        )
        if self.num_parallel > 1:
            x = tuple([x_i.expand(self.num_parallel, -1, -1) for x_i in args])
        else:
            x = args

        embed_out = [e(x_i) for e, x_i in zip(self.embeddings, x)]
        x = torch.concat(embed_out, dim=-1)
        return self.model(x)


class EmbeddedResNet(EmbeddedNet):
    def _make_layers(
        self,
        input_dim: Sequence[int],
        output_dim: int,
        embedding_dims: Sequence[int],
        hidden_dims: Sequence[int],
        num_parallel: int,
        activation: str | None,
        first_activation: str | None,
    ) -> tuple[list[nn.Module], list[nn.Module]]:
        # set defaults
        if activation is None:
            activation = "mish"
        if first_activation is None:
            first_activation = "mish"

        # construct modules for each embedding term
        embedding_layers: list[nn.Module] = [
            ResidualEmbedding(
                dim,
                hidden_dims[0] // self.num_embeddings,
                embedding_dims,
                num_parallel,
                activation,
                first_activation,
            )
            for dim in input_dim
        ]

        # construct layers for main model
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(ResidualBlock(in_dim, out_dim, num_parallel, activation))
        layers.append(Block(hidden_dims[-1], output_dim, num_parallel))

        return embedding_layers, layers
