# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from collections.abc import Sequence
from functools import reduce

from robot_rl.utils import get_param, resolve_layer_norm, resolve_linear, resolve_nn_activation

from .parallel import ParallelLinear


class ResMLP(nn.Sequential):
    """Residual MLP model.

    The residual network is a sequence of residual blocks consisting of a layer norm, linear layer, and activation
    function, with a non-residual last layer. The first block may optionally be non-residual. The remaining interface
    is similar to :class:`MLP`.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | Sequence[int],
        hidden_dims: Sequence[int],
        num_parallel: int = 1,
        activation: str = "mish",
        last_activation: str | None = None,
        first_residual: bool = True,
    ) -> None:
        """Initialize the model.

        Args:
            input_dim: Dimension of the input.
            output_dim: Dimension of the output.
            hidden_dims: Dimensions of the hidden layers. A value of ``-1`` indicates that the dimension should be
                inferred from the input dimension.
            num_parallel: Number of parallel networks. Defaults to 1.
            activation: Activation function.
            last_activation: Activation function of the last layer. None results in a linear last layer.
            first_residual: Whether the first block should be residual. Defaults to True.
        """
        super().__init__()

        # Store the activation function
        self.activation = activation
        # Resolve number of hidden dims if they are -1
        hidden_dims_processed = [input_dim if dim == -1 else dim for dim in hidden_dims]

        # Create layers sequentially
        layers = []
        first_block = _ResidualBlock if first_residual else _Block
        layers.append(first_block(input_dim, hidden_dims_processed[0], num_parallel, activation))
        layers.extend([
            _ResidualBlock(
                hidden_dims_processed[layer_index], hidden_dims_processed[layer_index + 1], num_parallel, activation
            )
            for layer_index in range(len(hidden_dims_processed) - 1)
        ])

        # Add last layer
        if isinstance(output_dim, int):
            layers.append(_Block(hidden_dims_processed[-1], output_dim, num_parallel, last_activation))
        else:
            # Compute the total output dimension
            total_out_dim = reduce(lambda x, y: x * y, output_dim)
            # Add a layer to reshape the output to the desired shape
            layers.append(_Block(hidden_dims_processed[-1], total_out_dim, num_parallel, last_activation))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))

        # Register the layers
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    def init_weights(self, scales: float | tuple[float]) -> None:
        """Initialize the weights of the MLP.

        Args:
            scales: Scale factor for the weights.
        """
        idx = 0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=get_param(scales, idx))
                nn.init.zeros_(module.bias)
                idx += 1
            elif isinstance(module, ParallelLinear):
                weight = module.weight.data
                n_parallel = weight.size(0)
                rows = weight.size(1)
                cols = weight.numel() // n_parallel // rows
                flattened = weight.new(n_parallel, rows, cols).normal_(0, 1)
                qs = []
                for flat_tensor in torch.unbind(flattened, dim=0):
                    if rows < cols:
                        flat_tensor.t_()
                    # Compute the qr factorization
                    q, r = torch.linalg.qr(flat_tensor)
                    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
                    d = torch.diag(r, 0)
                    ph = d.sign()
                    q *= ph
                    if rows < cols:
                        q.t_()
                    qs.append(q)
                qs = torch.stack(qs, dim=0)
                with torch.no_grad():
                    weight.view_as(qs).copy_(qs)
                    # Apply ReLU gain on parallel layers to match BFM-Zero's parallel-orthogonal init.
                    weight.mul_(get_param(scales, idx) * nn.init.calculate_gain("relu"))
                module.bias.data.fill_(0.0)
                idx += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the residual MLP."""
        for layer in self:
            x = layer(x)
        return x


class _Block(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int, num_parallel: int, activation: str | None) -> None:
        super().__init__()
        layers = []
        layers.append(resolve_layer_norm(input_dim, num_parallel))
        layers.append(resolve_linear(input_dim, output_dim, num_parallel))
        # Add the activation if specified
        if activation is not None:
            layers.append(resolve_nn_activation(activation))

        # Register the layers
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)


class _ResidualBlock(_Block):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res_x = x
        for layer in self:
            res_x = layer(res_x)
        return x + res_x
