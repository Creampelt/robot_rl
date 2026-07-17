# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from typing import Any


class TargetNetwork(nn.Module):
    """A slow-moving target copy of an online model, for bootstrapped value estimates.

    The online model is held as a plain reference (not a registered submodule), so this wrapper's own
    parameters/state are *only* the target's -- the frozen target params are naturally excluded from any
    ``requires_grad``-filtered optimizer. Calling the wrapper evaluates the target under ``torch.no_grad``.
    """

    def __init__(self, online: nn.Module, tau: float = 0.005) -> None:
        """Initialize the target network.

        Args:
            online: The online model to track. Kept as a reference; not registered as a submodule.
            tau: Default Polyak coefficient for :meth:`update` (``target <- (1 - tau) * target + tau * online``).
        """
        super().__init__()
        # Wrap the online net in a tuple so nn.Module does not register it (would double-count its params).
        self._online_ref = (online,)
        self.target = copy.deepcopy(online)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.tau = float(tau)

    @property
    def online(self) -> nn.Module:
        """The tracked online model."""
        return self._online_ref[0]

    @torch.no_grad()
    def update(self, tau: float | None = None) -> None:
        """Polyak soft-update the target toward the online model.

        Args:
            tau: Override for the interpolation coefficient; defaults to the value set at construction.
        """
        from robot_rl.utils import soft_update_params

        t = self.tau if tau is None else float(tau)
        # Shared foreach mul+add: a per-param lerp_ would round differently than soft_update_params.
        soft_update_params(tuple(self.online.parameters()), tuple(self.target.parameters()), t)
        # Buffers (e.g. normalization running stats) are not gradient-updated; hard-copy them each step.
        for tb, ob in zip(self.target.buffers(), self.online.buffers(), strict=True):
            tb.copy_(ob)

    @torch.no_grad()
    def hard_sync(self) -> None:
        """Copy the online parameters and buffers into the target exactly (equivalent to ``update(tau=1)``)."""
        for tp, op in zip(self.target.parameters(), self.online.parameters(), strict=True):
            tp.copy_(op)
        for tb, ob in zip(self.target.buffers(), self.online.buffers(), strict=True):
            tb.copy_(ob)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the target model under ``no_grad``."""
        with torch.no_grad():
            return self.target(*args, **kwargs)
