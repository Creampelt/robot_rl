# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce

import torch
import torch.nn as nn

from robot_rl.utils import resolve_nn_activation


class MLP(nn.Sequential):
    """Multi-layer perceptron.

    The MLP network is a sequence of linear layers and activation functions. The
    last layer is a linear layer that outputs the desired dimension unless the
    last activation function is specified.

    It provides additional conveniences:

    - If the hidden dimensions have a value of ``-1``, the dimension is inferred
      from the input dimension.
    - If the output dimension is a tuple, the output is reshaped to the desired
      shape.

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | Sequence[int],
        hidden_dims: Sequence[int],
        activation: str = "elu",
        last_activation: str | nn.Module | None = None,
        first_activation: str | nn.Module | list[str | nn.Module] | None = None,
    ):
        """Initialize the MLP.

        Args:
            input_dim: Dimension of the input.
            output_dim: Dimension of the output.
            hidden_dims: Dimensions of the hidden layers. A value of ``-1`` indicates
                that the dimension should be inferred from the input dimension.
            activation: Activation function. Defaults to "elu".
            last_activation: Activation function of the last layer. Defaults to None,
                in which case the last layer is linear.
            first_activation: Activation function of the first layer. Defaults to None,
                in which case `activation` is used. Can pass an activation name, nn.Module, or
                list of names/nn.Modules (which will be applied in order).
        """
        super().__init__()

        self.activation = activation

        # resolve activation functions
        activation_mod = resolve_nn_activation(activation)
        if isinstance(last_activation, str):
            last_activation_mod = resolve_nn_activation(last_activation)
        elif last_activation is not None:
            last_activation_mod = last_activation
        else:
            last_activation_mod = None
        # resolve first activation(s)
        if first_activation is None:
            first_activation = activation
        if not isinstance(first_activation, list):
            first_activation = [first_activation]
        first_activation_mod: list[nn.Module] = [
            resolve_nn_activation(act) if isinstance(act, str) else act for act in first_activation
        ]

        # resolve number of hidden dims if they are -1
        hidden_dims_processed = [input_dim if dim == -1 else dim for dim in hidden_dims]

        # create layers sequentially
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims_processed[0]))
        layers.extend(first_activation_mod)

        for layer_index in range(len(hidden_dims_processed) - 1):
            layers.append(nn.Linear(hidden_dims_processed[layer_index], hidden_dims_processed[layer_index + 1]))
            layers.append(activation_mod)

        # add last layer
        if isinstance(output_dim, int):
            layers.append(nn.Linear(hidden_dims_processed[-1], output_dim))
        else:
            # compute the total output dimension
            total_out_dim = reduce(lambda x, y: x * y, output_dim)
            # add a layer to reshape the output to the desired shape
            layers.append(nn.Linear(hidden_dims_processed[-1], total_out_dim))
            layers.append(nn.Unflatten(-1, tuple(output_dim)))

        # add last activation function if specified
        if last_activation_mod is not None:
            layers.append(last_activation_mod)

        # register the layers
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    def init_weights(self, scales: float | tuple[float] | None = None):
        """Initialize the weights of the MLP.

        Args:
            scales: Scale factor for the weights.
        """

        def get_scale(idx) -> float:
            """Get the scale factor for the weights of the MLP.

            Args:
                idx: Index of the layer.
            """
            if isinstance(scales, (list, tuple)):
                return scales[idx]
            elif isinstance(scales, float):
                return scales
            else:
                return nn.init.calculate_gain(self.activation)

        # initialize the weights
        for idx, module in enumerate(self):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=get_scale(idx))
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP.

        Args:
            x: Input tensor.
        """
        for layer in self:
            x = layer(x)
        return x

    def reset(self, dones=None, hidden_states=None):
        pass

    def detach_hidden_states(self, dones=None):
        pass
