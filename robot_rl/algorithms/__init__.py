# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learning algorithms."""

from .distillation import Distillation
from .fb_cpr import FbCpr
from .ppo import PPO
from .sac import SAC
from .terrain_fb_cpr import TerrainFbCpr

__all__ = ["PPO", "SAC", "Distillation", "FbCpr", "TerrainFbCpr"]
