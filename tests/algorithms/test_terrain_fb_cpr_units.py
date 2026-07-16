# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for TerrainFbCpr's standalone pieces (the full algorithm needs an env + Isaac)."""

from __future__ import annotations

import torch

from robot_rl.algorithms.terrain_fb_cpr import (
    NUM_FAMILIES,
    TRAIN_TILE_CLIP_FAMILIES,
    _BilinearResidualHead,
    _NormedEncoder,
    _RunningStd,
)


class TestBilinearResidualHead:
    """No additive (s, a)-only path to the output."""

    def test_c_zero_is_exactly_zero(self) -> None:
        """With c = 0 the residual must vanish EXACTLY, so the prediction IS the baseline."""
        head = _BilinearResidualHead(in_dim=8, c_dim=4, out_dim=6)
        sa = torch.randn(16, 8)
        out = head(sa, torch.zeros(16, 4))
        assert torch.equal(out, torch.zeros(16, 6))

    def test_grad_reaches_c_at_c_zero(self) -> None:
        """dL/dc = W(s,a)^T r is nonzero AT c = 0 -- the dead basin is structurally removed."""
        head = _BilinearResidualHead(in_dim=8, c_dim=4, out_dim=6)
        sa = torch.randn(16, 8)
        c = torch.zeros(16, 4, requires_grad=True)
        head(sa, c).sum().backward()
        assert c.grad is not None and float(c.grad.abs().sum()) > 0.0


class TestRunningStd:
    """The residual re-standardizer: update() folds targets only; scale() is read-only."""

    def test_update_tracks_target_variance(self) -> None:
        """The EMA converges toward the batch variance of what update() sees."""
        rs = _RunningStd(3, momentum=0.5)
        x = torch.randn(4096, 3) * torch.tensor([1.0, 2.0, 4.0])
        for _ in range(16):
            rs.update(x)
        assert torch.allclose(rs.var.sqrt(), torch.tensor([1.0, 2.0, 4.0]), rtol=0.15)

    def test_scale_never_updates(self) -> None:
        """scale() on predictions must not fold prediction statistics into the EMA."""
        rs = _RunningStd(3)
        before = rs.var.clone()
        rs.scale(torch.randn(64, 3) * 100)
        assert torch.equal(rs.var, before)

    def test_floor_bounds_amplification(self) -> None:
        """A near-zero-variance dim must not blow the scaled value up unboundedly."""
        rs = _RunningStd(1, floor=0.05)
        for _ in range(200):
            rs.update(torch.zeros(64, 1))
        assert float(rs.scale(torch.ones(1, 1))) <= 1.0 / 0.05 + 1e-6


class TestTrainCompatMap:
    """The train-time tile->clip compat map (WIDER than eval routing on purpose)."""

    def test_every_tile_family_has_a_source(self) -> None:
        """Every generated tile family must be able to draw SOME clip family."""
        assert set(TRAIN_TILE_CLIP_FAMILIES) == set(range(NUM_FAMILIES))
        assert all(len(v) > 0 for v in TRAIN_TILE_CLIP_FAMILIES.values())

    def test_flat_serves_the_perception_blind_tiles(self) -> None:
        """Flat clips serve flat + rough + slope in TRAINING (wider than the eval routing map)."""
        for tile in (0, 1, 2):
            assert TRAIN_TILE_CLIP_FAMILIES[tile] == (0,)

    def test_terrain_families_are_diagonal(self) -> None:
        """stairs/boxes/edge tiles draw only their own family (feasibility is the point of coning)."""
        for tile in (3, 4, 5):
            assert TRAIN_TILE_CLIP_FAMILIES[tile] == (tile,)


class TestNormedEncoder:
    """The inference wrapper: always running stats, regardless of batch size or module mode."""

    def test_batch_one_and_no_stat_update(self) -> None:
        """One deployment frame must normalize like a batch and leave the running stats untouched.

        Train-mode BatchNorm would both crash on batch-1 variance and drift the EMA.
        """
        import torch.nn as nn

        norm = nn.BatchNorm1d(4, momentum=0.01, affine=False)
        norm(torch.randn(256, 4) * 3 + 1)  # seed the running stats in train mode
        mean, var = norm.running_mean.clone(), norm.running_var.clone()
        enc = _NormedEncoder(nn.Linear(4, 4), norm)
        x = torch.randn(1, 4)
        out_single = enc(x)
        out_in_batch = enc(x.repeat(8, 1))[0]
        assert torch.allclose(out_single[0], out_in_batch, atol=1e-6)
        assert torch.equal(norm.running_mean, mean) and torch.equal(norm.running_var, var)
