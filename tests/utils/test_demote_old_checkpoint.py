# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that checkpoint demotion strips only resume-only state, and only outside the keep-full window."""

import pathlib
import torch

from robot_rl.utils import demote_old_checkpoint

POLICY_KEYS = ("actor_state_dict", "backward_map_state_dict", "obs_normalizer_state_dict")
RESUME_KEYS = ("critic_state_dict", "actor_optimizer_state_dict", "expert_buffer_state_dict")
SAVE_INTERVAL = 10


class _Alg:
    """Minimal stand-in exposing the one method demotion depends on."""

    @staticmethod
    def policy_state_keys() -> tuple[str, ...]:
        """Return the keys a slim checkpoint must retain."""
        return POLICY_KEYS


def _write(log_dir: pathlib.Path, it: int) -> None:
    """Write a full checkpoint carrying both policy and resume-only state."""
    ckpt = {k: torch.zeros(4) for k in POLICY_KEYS + RESUME_KEYS}
    ckpt |= {"iter": it, "env_step": it * 100, "infos": {"note": "x"}}
    torch.save(ckpt, log_dir / f"model_{it}.pt")


class TestDemoteOldCheckpoint:
    """Behavior of the rolling policy-only demotion."""

    def test_demotes_the_checkpoint_leaving_the_window(self, tmp_path: pathlib.Path) -> None:
        """The checkpoint that just left the window loses resume state; the newest keeps it."""
        for it in (0, SAVE_INTERVAL, 2 * SAVE_INTERVAL):
            _write(tmp_path, it)
        demoted = demote_old_checkpoint(_Alg(), str(tmp_path), 2 * SAVE_INTERVAL, 1, SAVE_INTERVAL)
        assert demoted == SAVE_INTERVAL

        slim = torch.load(tmp_path / f"model_{SAVE_INTERVAL}.pt", weights_only=False)
        assert slim["_policy_only"] is True
        assert set(slim) == set(POLICY_KEYS) | {"iter", "env_step", "infos", "_policy_only"}
        assert slim["iter"] == SAVE_INTERVAL, "iteration metadata must survive so the file stays identifiable"

        # keep=1 means exactly one resumable checkpoint: the newest is untouched
        newest = torch.load(tmp_path / f"model_{2 * SAVE_INTERVAL}.pt", weights_only=False)
        assert set(RESUME_KEYS) <= set(newest)

    def test_is_idempotent_and_disabled_by_a_falsy_keep(self, tmp_path: pathlib.Path) -> None:
        """Re-running does not re-demote, and keep of None/0 turns demotion off."""
        for it in (0, SAVE_INTERVAL):
            _write(tmp_path, it)
        assert demote_old_checkpoint(_Alg(), str(tmp_path), SAVE_INTERVAL, 1, SAVE_INTERVAL) == 0
        # re-running must not re-demote (the caller re-uploads on a non-None return)
        assert demote_old_checkpoint(_Alg(), str(tmp_path), SAVE_INTERVAL, 1, SAVE_INTERVAL) is None
        for keep in (None, 0):
            assert demote_old_checkpoint(_Alg(), str(tmp_path), SAVE_INTERVAL, keep, SAVE_INTERVAL) is None

    def test_no_op_without_policy_state_keys_or_a_missing_file(self, tmp_path: pathlib.Path) -> None:
        """An algorithm without policy_state_keys, or a window not yet left, demotes nothing."""
        _write(tmp_path, SAVE_INTERVAL)
        assert demote_old_checkpoint(object(), str(tmp_path), SAVE_INTERVAL, 1, SAVE_INTERVAL) is None
        # window has not been left yet, so nothing to demote
        assert demote_old_checkpoint(_Alg(), str(tmp_path), 0, 1, SAVE_INTERVAL) is None
