# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for offline Forward-Backward (dummy VecEnv and a stored dataset, no Isaac Sim)."""

from __future__ import annotations

import copy
import torch
from pathlib import Path
from tensordict import TensorDict

import pytest

from robot_rl.algorithms import Fb

OBS_DIM, ACT_DIM, NUM_ENVS, Z_DIM, BATCH = 6, 3, 8, 4, 16


class _DummyVecEnv:
    """A minimal VecEnv supplying only the shapes offline training needs; never stepped."""

    def __init__(self) -> None:
        self.num_envs = NUM_ENVS
        self.num_actions = ACT_DIM
        self.cfg = None

    def get_observations(self) -> TensorDict:
        """Return a representative observation."""
        return TensorDict({"policy": torch.randn(NUM_ENVS, OBS_DIM)}, batch_size=NUM_ENVS)


def _write_dataset(tmp_path: Path, num: int = 128) -> Path:
    """Write a small transition dataset and return its path."""
    obs = TensorDict({"policy": torch.randn(num, OBS_DIM)}, batch_size=[num])
    nxt = TensorDict({"policy": torch.randn(num, OBS_DIM)}, batch_size=[num])
    path = tmp_path / "offline.pt"
    torch.save(
        {
            "observations": obs,
            "next_observations": nxt,
            "actions": torch.randn(num, ACT_DIM).clamp(-1.0, 1.0),
            "terminated": torch.rand(num, 1) < 0.05,
        },
        path,
    )
    return path


def _make_cfg(dataset_path: Path, behavior_reg_coef: float = 0.0) -> dict:
    return {
        "clip_actions": 1.0,
        "storage_device": "cpu",
        "obs_groups": {"actor": ["policy"], "critic": ["policy"], "backward": ["policy"]},
        "actor": {
            "class_name": "ResidualFuseModel",
            "embedding_dims": [16],
            "hidden_dims": [16, 16],
            "distribution_cfg": {"class_name": "TruncatedGaussianDistribution", "init_std": 1.0},
        },
        "algorithm": {
            "class_name": "Fb",
            "dataset_path": str(dataset_path),
            "z_dim": Z_DIM,
            "batch_size": BATCH,
            # the FB loss reduces over an ensemble dimension, so the forward map must be parallel
            # mirrors the shipped FbCpr configuration at small scale: a parallel residual forward map
            # and a ball-normalized backward map, which the orthonormality loss assumes
            "forward_map": {
                "class_name": "ResidualFuseModel",
                "embedding_dims": [16],
                "hidden_dims": [16, 16],
                "num_parallel": 2,
            },
            "backward_map": {
                "class_name": "MLPModel",
                "hidden_dims": [16],
                "last_activation": "ball_norm",
                "normalize_first_layer": True,
            },
            "actor_learning_rate": 1e-3,
            "forward_learning_rate": 1e-3,
            "backward_learning_rate": 1e-3,
            "gamma": 0.99,
            "fb_tau": 0.1,
            "ortho_loss_coef": 1.0,
            "behavior_reg_coef": behavior_reg_coef,
            "max_grad_norm": 1.0,
            "compile_mode": None,
        },
    }


def _build(tmp_path: Path, behavior_reg_coef: float = 0.0) -> Fb:
    env = _DummyVecEnv()
    return Fb.construct_algorithm(
        env.get_observations(), env, _make_cfg(_write_dataset(tmp_path), behavior_reg_coef), device="cpu"
    )


class TestFb:
    """End-to-end CPU behavior of the offline Forward-Backward algorithm."""

    def test_update_produces_finite_losses(self, tmp_path: Path) -> None:
        """Every reported loss term must be finite after a real gradient step."""
        alg = _build(tmp_path)
        alg.train_mode()
        loss_dict, extras = alg.update()
        assert loss_dict, "update reported no losses"
        for name, value in {**loss_dict, **extras}.items():
            assert torch.isfinite(value).all(), f"{name} is not finite"

    def test_update_changes_the_learned_parameters(self, tmp_path: Path) -> None:
        """A step that leaves every model untouched would pass a finiteness check but learn nothing."""
        alg = _build(tmp_path)
        alg.train_mode()
        before = [copy.deepcopy(m.state_dict()) for m in alg.models]
        for _ in range(3):
            alg.update()
        moved = [
            any(not torch.equal(b[k], m.state_dict()[k]) for k in b) for b, m in zip(before, alg.models, strict=False)
        ]
        assert all(moved), f"some models did not change: {moved}"

    def test_targets_track_the_live_networks(self, tmp_path: Path) -> None:
        """The forward target must follow its online network, or the TD target never improves."""
        alg = _build(tmp_path)
        alg.train_mode()
        before = copy.deepcopy(alg.target_forward_map.state_dict())
        for _ in range(3):
            alg.update()
        after = alg.target_forward_map.state_dict()
        assert any(not torch.equal(before[k], after[k]) for k in before)

    def test_behavior_regularizer_is_reported_only_when_enabled(self, tmp_path: Path) -> None:
        """At coefficient 0 this is the plain offline FB objective; the term must contribute nothing."""
        off = _build(tmp_path, behavior_reg_coef=0.0)
        off.train_mode()
        assert off.update()[0]["Actor_Loss/behavior_loss"] == 0.0

        on = _build(tmp_path, behavior_reg_coef=1.0)
        on.train_mode()
        assert on.update()[0]["Actor_Loss/behavior_loss"] > 0.0

    def test_required_groups_cover_every_model_that_reads_the_dataset(self) -> None:
        """The forward map is built on the critic set, so omitting it would let a short dataset through."""
        groups = {"actor": ["a"], "critic": ["c"], "backward": ["b"]}
        assert Fb.expert_bundle_groups(groups) == ["a", "b", "c"]

    def test_dataset_missing_a_consumed_group_is_rejected(self, tmp_path: Path) -> None:
        """A dataset lacking a group the backward map reads would train quietly on less than configured."""
        cfg = _make_cfg(_write_dataset(tmp_path))
        cfg["obs_groups"]["backward"] = ["absent"]
        env = _DummyVecEnv()
        with pytest.raises((ValueError, KeyError)):
            Fb.construct_algorithm(env.get_observations(), env, cfg, device="cpu")

    def test_eval_is_a_no_op_without_motions(self, tmp_path: Path) -> None:
        """Training must not require an eval bundle; scoring is optional."""
        alg = _build(tmp_path)
        assert alg.eval_buffer is None
        assert alg.eval(_DummyVecEnv()) == []

    def test_policy_state_keys_exclude_optimizers(self, tmp_path: Path) -> None:
        """Slimmed checkpoints keep only what inference needs."""
        alg = _build(tmp_path)
        saved = alg.save()
        for key in Fb.policy_state_keys():
            assert key in saved
        assert not any("optimizer" in key for key in Fb.policy_state_keys())
