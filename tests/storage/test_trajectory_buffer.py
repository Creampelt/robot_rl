# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that trajectory windows are stitched from real frames only, never hold-final-frame padding."""

import torch

from robot_rl.storage.trajectory_buffer import NUM_STITCH_SEGMENTS, _get_idxs

BUCKET = 500
NUM_SLICES = 512
PADDED_ROWS = [11, 50, 127, 300, BUCKET] * 16
FULL_ROWS = [BUCKET] * 32


def _draw(valid_lengths: list[int], seq_length: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Draw windows for the given per-row valid lengths, reshaped to (slice, frame)."""
    valid = torch.tensor(valid_lengths)
    idxs = _get_idxs(torch.ones(len(valid)), valid, NUM_SLICES, seq_length, BUCKET, NUM_STITCH_SEGMENTS)
    return valid, [t.view(NUM_SLICES, seq_length) for t in idxs]


class TestTrajectoryWindowStitching:
    """Window construction over bundles whose rows are padded out to a common bucket size."""

    def test_padded_rows_are_never_sampled(self) -> None:
        """Every current and next index stays inside its own row's valid length."""
        for rows, seq_length in ((FULL_ROWS, 250), (PADDED_ROWS, 250), (PADDED_ROWS, 8), ([5] * 16, 250)):
            valid, (ep, frame, next_ep, next_frame) = _draw(rows, seq_length)
            assert (frame >= 0).all() and (frame < valid[ep]).all()
            assert (next_frame >= 0).all() and (next_frame < valid[next_ep]).all()

    def test_degenerate_rows_shorter_than_one_segment(self) -> None:
        """Rows too short to fill even one segment wrap within real frames instead of padding."""
        valid, (ep, frame, next_ep, next_frame) = _draw([1] * 16, 100)
        assert (frame < valid[ep]).all()
        assert (next_frame < valid[next_ep]).all()

    def test_next_index_is_the_windows_following_frame(self) -> None:
        """A frame + 1 lookup cannot cross a seam, so the successor is emitted explicitly."""
        for seq_length in (8, 150, 250):
            _, (ep, frame, next_ep, next_frame) = _draw(PADDED_ROWS, seq_length)
            assert torch.equal(next_ep[:, :-1], ep[:, 1:])
            assert torch.equal(next_frame[:, :-1], frame[:, 1:])

    def test_rows_covering_the_window_stay_unbroken(self) -> None:
        """Un-padded bundles keep single-clip windows, so stitching cannot perturb existing runs."""
        for seq_length in (8, 250):
            _, (ep, frame, _, _) = _draw(FULL_ROWS, seq_length)
            assert (ep == ep[:, :1]).all()
            assert torch.equal(frame, frame[:, :1] + torch.arange(seq_length))
