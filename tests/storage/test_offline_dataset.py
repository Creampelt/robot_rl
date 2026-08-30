# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the offline transition dataset."""

from __future__ import annotations

import torch
from pathlib import Path
from tensordict import TensorDict

import pytest

from robot_rl.storage import OfflineTransitionDataset

NUM = 64
ACT = 5
Z = 8


def _write(
    tmp_path: Path,
    num: int = NUM,
    terminated: torch.Tensor | None = None,
    rewards: torch.Tensor | None = None,
    obs_dim: int = 7,
    next_num: int | None = None,
) -> Path:
    """Write a minimal dataset file and return its path."""
    obs = TensorDict({"state": torch.randn(num, obs_dim)}, batch_size=[num])
    n = num if next_num is None else next_num
    nxt = TensorDict({"state": torch.randn(n, obs_dim)}, batch_size=[n])
    payload = {"observations": obs, "next_observations": nxt, "actions": torch.randn(num, ACT)}
    if terminated is not None:
        payload["terminated"] = terminated
    if rewards is not None:
        payload["rewards"] = rewards
    path = tmp_path / "data.pt"
    torch.save(payload, path)
    return path


class TestOfflineTransitionDataset:
    """Tests for loading and sampling stored transitions."""

    def test_sampled_batch_has_the_replay_buffer_shapes(self, tmp_path: Path) -> None:
        """A sampled batch must match what the algorithms already consume."""
        ds = OfflineTransitionDataset(_write(tmp_path), z_dim=Z, batch_size=16)
        batch = ds.sample_mini_batch()
        assert len(ds) == NUM
        assert batch.actions.shape == (16, ACT)
        assert batch.context.shape == (16, Z)
        assert batch.observations["state"].shape == (16, 7)
        assert batch.next_observations["state"].shape == (16, 7)

    def test_terminated_transitions_get_zero_discount(self, tmp_path: Path) -> None:
        """Gamma must vanish exactly where the episode ended, so no value bootstraps past it."""
        term = torch.zeros(NUM, 1, dtype=torch.bool)
        term[::2] = True
        ds = OfflineTransitionDataset(_write(tmp_path, terminated=term), z_dim=Z, batch_size=8, gamma=0.97)
        assert torch.all(ds.gammas[term.squeeze()] == 0.0)
        assert torch.allclose(ds.gammas[~term.squeeze()], torch.full((1,), 0.97))

    def test_missing_reward_and_terminated_default_to_zero(self, tmp_path: Path) -> None:
        """Forward-backward is reward-free, so a dataset need not carry rewards."""
        ds = OfflineTransitionDataset(_write(tmp_path), z_dim=Z, batch_size=8, gamma=0.99)
        assert torch.all(ds.rewards == 0.0)
        assert not ds.next_terminated.any()
        assert torch.allclose(ds.gammas, torch.full((1,), 0.99))

    def test_mismatched_transition_counts_are_rejected(self, tmp_path: Path) -> None:
        """A truncated next_observations set would silently misalign every transition."""
        with pytest.raises(ValueError, match="Transition counts disagree"):
            OfflineTransitionDataset(_write(tmp_path, next_num=NUM - 1), z_dim=Z, batch_size=8)

    def test_obs_groups_are_reported_for_the_contract_check(self, tmp_path: Path) -> None:
        """construct_algorithm refuses a dataset missing a group its consumers read."""
        ds = OfflineTransitionDataset(_write(tmp_path), z_dim=Z, batch_size=8)
        assert ds.obs_groups == ["state"]
