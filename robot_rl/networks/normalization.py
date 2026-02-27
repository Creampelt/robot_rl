# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2020 Preferred Networks, Inc.

from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, eps=1e-2, until=None):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the module learns input values until the sum of batch sizes
            exceeds it.

        Note: The normalization parameters are computed over the whole batch, not for each environment separately.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x):
        """Normalize mean and variance of values based on empirical values."""

        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        """Learn input values without computing the output values of them"""

        if not self.training:
            return
        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count
        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)

    # @torch.jit.unused
    def inverse(self, y):
        """De-normalize values based on empirical values."""

        return y * (self._std + self.eps) + self._mean


class EmpiricalDiscountedVariationNormalization(nn.Module):
    """Reward normalization from Pathak's large scale study on PPO.

    Reward normalization. Since the reward function is non-stationary, it is useful to normalize
    the scale of the rewards so that the value function can learn quickly. We did this by dividing
    the rewards by a running estimate of the standard deviation of the sum of discounted rewards.
    """

    def __init__(self, shape, eps=1e-2, gamma=0.99, until=None):
        super().__init__()

        self.emp_norm = EmpiricalNormalization(shape, eps, until)
        self.disc_avg = _DiscountedAverage(gamma)

    def forward(self, rew):
        if self.training:
            # update discounted rewards
            avg = self.disc_avg.update(rew)
            # update moments from discounted rewards
            self.emp_norm.update(avg)

        # normalize rewards with the empirical std
        if self.emp_norm._std > 0:
            return rew / self.emp_norm._std
        else:
            return rew


class EMANormalization(nn.Module):
    """Exponential moving average."""

    def __init__(
        self,
        tau: float = 0.99,
        epsilon: float = 1e-8,
        shape: tuple[int, ...] = (1,),
        translate: bool = False,
        scale: bool = False,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.epsilon = epsilon
        self.translate = translate
        self.scale = scale
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("mean_square", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("counter", torch.LongTensor([0]))

    def forward(self, x):
        m = x.mean()
        sm = x.pow(2).mean()
        self.mean.data = self.tau * self.mean + (1 - self.tau) * m  # type: ignore
        self.mean_square.data = self.tau * self.mean_square + (1 - self.tau) * sm  # type: ignore
        self.counter += 1  # type: ignore
        norm = 1 - self.tau**self.counter
        ema_mean = self.mean / norm  # type: ignore
        ema_mean_square = self.mean_square / norm  # type: ignore
        var = torch.clamp(ema_mean_square - ema_mean**2, min=self.epsilon)

        translate_mean = ema_mean if self.translate else 0
        scale_std = torch.sqrt(var) if self.scale else 1
        return (x - translate_mean) / scale_std

    @property
    def S(self) -> torch.Tensor:
        norm = 1 - self.tau**self.counter
        ema_mean = self.mean / norm  # type: ignore
        ema_mean_square = self.mean_square / norm  # type: ignore
        var = torch.clamp(ema_mean_square - ema_mean**2, self.epsilon)
        return var

    @property
    def M(self) -> torch.Tensor:
        norm = 1 - self.tau**self.counter
        ema_mean = self.mean / norm  # type: ignore
        return ema_mean


"""
Helper class.
"""


class _DiscountedAverage:
    r"""Discounted average of rewards.

    The discounted average is defined as:

    .. math::

        \bar{R}_t = \gamma \bar{R}_{t-1} + r_t

    Args:
        gamma (float): Discount factor.
    """

    def __init__(self, gamma):
        self.avg = None
        self.gamma = gamma

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        if self.avg is None:
            self.avg = rew
        else:
            self.avg = self.avg * self.gamma + rew
        return self.avg
