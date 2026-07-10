# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inference-policy adapters: exportable modules that chain a shared upstream module into actor inference."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from .mlp_model import MLPModel


class SharedMemoryInferencePolicy(nn.Module):
    """Adapter that chains a shared memory module into actor inference."""

    is_recurrent: bool = True

    def __init__(self, memory: nn.Module, actor: MLPModel) -> None:
        """Wrap the memory module and the actor head it feeds."""
        super().__init__()
        self.memory = memory
        self.actor = actor

    @property
    def output_mean(self) -> torch.Tensor:
        """Return the mean of the current output distribution."""
        return self.actor.output_mean

    @property
    def output_std(self) -> torch.Tensor:
        """Return the standard deviation of the current output distribution."""
        return self.actor.output_std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Return the entropy of the current output distribution."""
        return self.actor.output_entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Return raw parameters of the current output distribution."""
        return self.actor.output_distribution_params

    def forward(self, obs: TensorDict, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Advance the memory and run the actor head on its latent."""
        latent = self.memory(obs)
        return self.actor.forward_from_latent(latent, *args, **kwargs)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Reset the memory (and actor) recurrent state."""
        self.memory.reset(dones)
        self.actor.reset(dones)


class EncoderInferencePolicy(nn.Module):
    """Adapter that chains a shared observation encoder into actor inference (latent as an extra input)."""

    is_recurrent: bool = False

    def __init__(self, encoder: nn.Module, actor: MLPModel, latent_first: bool = True) -> None:
        """Wrap the encoder module and the actor consuming its latent.

        Args:
            encoder: The shared observation encoder.
            actor: The actor model taking the encoder latent as an extra input.
            latent_first: Whether the latent precedes the other extra inputs (PPO/SAC convention) or trails
                them (FB-CPR convention: ``actor(obs, z, c)``).
        """
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.latent_first = latent_first

    @property
    def output_mean(self) -> torch.Tensor:
        """Return the mean of the current output distribution."""
        return self.actor.output_mean

    @property
    def output_std(self) -> torch.Tensor:
        """Return the standard deviation of the current output distribution."""
        return self.actor.output_std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Return the entropy of the current output distribution."""
        return self.actor.output_entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Return raw parameters of the current output distribution."""
        return self.actor.output_distribution_params

    def forward(self, obs: TensorDict, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Run the actor with the encoder latent as an extra input."""
        latent = self.encoder(obs)
        inputs = (latent, *args) if self.latent_first else (*args, latent)
        return self.actor(obs, *inputs, **kwargs)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Reset the actor state (the encoder is stateless)."""
        self.actor.reset(dones)
