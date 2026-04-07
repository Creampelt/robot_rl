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


class MLP(nn.Sequential):
    """Multi-Layer Perceptron.

    The MLP network is a sequence of linear layers and activation functions. The last layer is a linear layer that
    outputs the desired dimension unless the last activation function is specified.

    It provides additional conveniences:
    - If the hidden dimensions have a value of ``-1``, the dimension is inferred from the input dimension.
    - If the output dimension is a tuple, the output is reshaped to the desired shape.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | Sequence[int],
        hidden_dims: Sequence[int],
        num_parallel: int = 1,
        activation: str = "elu",
        first_activation: str | None = None,
        last_activation: str | None = None,
        normalize_input: bool = False,
    ) -> None:
        """Initialize the MLP.

        Args:
            input_dim: Dimension of the input.
            output_dim: Dimension of the output.
            hidden_dims: Dimensions of the hidden layers. A value of ``-1`` indicates that the dimension should be
                inferred from the input dimension.
            num_parallel: Number of parallel networks. Defaults to 1.
            activation: Activation function.
            first_activation: Activation function of the first layer. None uses the default model activation.
            last_activation: Activation function of the last layer. None results in a linear last layer.
            normalize_input: Whether to normalize the input with LayerNorm.
        """
        super().__init__()

        # Store the activation function
        self.activation = activation
        # Resolve activation functions
        activation_mod = resolve_nn_activation(activation)
        first_activation_mod = (
            resolve_nn_activation(first_activation) if first_activation is not None else activation_mod
        )
        last_activation_mod = resolve_nn_activation(last_activation) if last_activation is not None else None
        # Resolve number of hidden dims if they are -1
        hidden_dims_processed = [input_dim if dim == -1 else dim for dim in hidden_dims]

        # Create layers sequentially
        layers = []
        if normalize_input:
            layers.append(resolve_layer_norm(input_dim, num_parallel))
        layers.append(resolve_linear(input_dim, hidden_dims_processed[0], num_parallel))
        layers.append(first_activation_mod)

        for layer_index in range(len(hidden_dims_processed) - 1):
            layers.append(
                resolve_linear(hidden_dims_processed[layer_index], hidden_dims_processed[layer_index + 1], num_parallel)
            )
            layers.append(activation_mod)

        # Add last layer
        if isinstance(output_dim, int):
            layers.append(resolve_linear(hidden_dims_processed[-1], output_dim, num_parallel))
        else:
            # Compute the total output dimension
            total_out_dim = reduce(lambda x, y: x * y, output_dim)
            # Add a layer to reshape the output to the desired shape
            layers.append(resolve_linear(hidden_dims_processed[-1], total_out_dim, num_parallel))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))

        # Add last activation function if specified
        if last_activation_mod is not None:
            layers.append(last_activation_mod)

        # Register the layers
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    def init_weights(self, scales: float | tuple[float]) -> None:
        """Initialize the weights of the MLP.

        Args:
            scales: Scale factor for the weights.
        """
        for idx, module in enumerate(self):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=get_param(scales, idx))
                nn.init.zeros_(module.bias)
            elif isinstance(module, ParallelLinear):
                # gain = nn.init.calculate_gain(self.activation)
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
                    weight.mul_(get_param(scales, idx))
                module.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        for layer in self:
            x = layer(x)
        return x
