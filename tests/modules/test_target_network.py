# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the TargetNetwork wrapper."""

import torch
import torch.nn as nn

from robot_rl.modules import TargetNetwork


def _make_net() -> nn.Module:
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    # Give it a buffer to exercise buffer copying (mimics a normalization running stat).
    net.register_buffer("running_stat", torch.zeros(4))
    return net


class TestTargetNetwork:
    """Tests for ``TargetNetwork``."""

    def test_init_matches_online_and_is_frozen(self) -> None:
        """Target starts as an exact copy with all parameters frozen; the wrapper exposes only target params."""
        online = _make_net()
        tn = TargetNetwork(online, tau=0.01)
        for tp, op in zip(tn.target.parameters(), online.parameters()):
            assert torch.equal(tp, op)
            assert tp.requires_grad is False
        # The online net is a plain reference, not a registered submodule -> wrapper params == target params only.
        assert all(not p.requires_grad for p in tn.parameters())
        assert sum(p.numel() for p in tn.parameters()) == sum(p.numel() for p in tn.target.parameters())
        assert tn.online is online

    def test_soft_update_polyak(self) -> None:
        """update(tau) should interpolate: target <- (1 - tau) * target + tau * online."""
        online = _make_net()
        tn = TargetNetwork(online, tau=0.25)
        # Perturb the online weights so target and online differ.
        with torch.no_grad():
            for p in online.parameters():
                p.add_(torch.randn_like(p))
        before = [tp.clone() for tp in tn.target.parameters()]
        tn.update()  # tau=0.25
        for b, tp, op in zip(before, tn.target.parameters(), online.parameters()):
            expected = 0.75 * b + 0.25 * op
            assert torch.allclose(tp, expected, atol=1e-6)

    def test_hard_sync_and_tau_one_equivalent(self) -> None:
        """hard_sync() and update(tau=1.0) should both copy the online params exactly."""
        for use_hard in (True, False):
            online = _make_net()
            tn = TargetNetwork(online, tau=0.01)
            with torch.no_grad():
                for p in online.parameters():
                    p.add_(torch.randn_like(p))
                online.running_stat.copy_(torch.arange(4, dtype=torch.float))
            tn.hard_sync() if use_hard else tn.update(tau=1.0)
            for tp, op in zip(tn.target.parameters(), online.parameters()):
                assert torch.allclose(tp, op, atol=1e-7)
            assert torch.allclose(tn.target.get_buffer("running_stat"), online.get_buffer("running_stat"))

    def test_buffers_copied_on_update(self) -> None:
        """Buffers (non-gradient stats) are hard-copied from online to target on every update."""
        online = _make_net()
        tn = TargetNetwork(online, tau=0.1)
        with torch.no_grad():
            online.running_stat.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        tn.update()
        assert torch.allclose(tn.target.get_buffer("running_stat"), torch.tensor([1.0, 2.0, 3.0, 4.0]))

    def test_forward_is_target_and_detached(self) -> None:
        """Calling the wrapper evaluates the target and returns a tensor with no gradient graph."""
        online = _make_net()
        tn = TargetNetwork(online, tau=0.01)
        x = torch.randn(5, 4)
        out = tn(x)
        assert torch.equal(out, tn.target(x))
        assert out.requires_grad is False and out.grad_fn is None

    def test_online_grad_unaffected_by_wrapper(self) -> None:
        """Optimizing the online net is unaffected: online params still require grad and get gradients."""
        online = _make_net()
        TargetNetwork(online, tau=0.01)  # wrapping must not freeze the online net
        x = torch.randn(3, 4)
        online(x).sum().backward()
        assert all(p.requires_grad for p in online.parameters())
        assert all(p.grad is not None for p in online.parameters())
