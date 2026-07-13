# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the TrajectoryBuffer's sampling weights, held-out split and caller-chosen rows.

The failure these guard against is silent: ``eval()`` overwrites ``priorities`` wholesale every
``eval_interval`` and renormalizes, so a family-softening or held-out weight baked into that tensor
survives until the first eval and then disappears -- leaving priorities that are pure EMD difficulty,
which preferentially hammer whatever the policy is worst at. The symptom looks like a mis-set alpha
rather than like alpha having been erased.
"""

from __future__ import annotations

import torch
from pathlib import Path
from tensordict import TensorDict

import pytest

from robot_rl.storage import TrajectoryBuffer

NUM_MOTIONS = 12
BUCKET = 20
OBS_DIM = 5


@pytest.fixture
def buffer(tmp_path: Path) -> TrajectoryBuffer:
    """Build a buffer over a synthetic bundle: 3 families of sizes 8 / 3 / 1, with 2 rows held out."""
    families = torch.tensor([0] * 8 + [1] * 3 + [2] * 1, dtype=torch.float32)
    heldout = torch.zeros(NUM_MOTIONS)
    heldout[[3, 9]] = 1.0
    # scalar-per-frame keys carry NO trailing feature dim in a real bundle: (num_motions, bucket_size)
    motions = TensorDict(
        {
            "expert": torch.randn(NUM_MOTIONS, BUCKET, OBS_DIM),
            "length": torch.full((NUM_MOTIONS, BUCKET), float(BUCKET)),
            "family": families.view(NUM_MOTIONS, 1).expand(NUM_MOTIONS, BUCKET).clone(),
            "heldout": heldout.view(NUM_MOTIONS, 1).expand(NUM_MOTIONS, BUCKET).clone(),
        },
        batch_size=[NUM_MOTIONS, BUCKET],
    )
    path = tmp_path / "motions.pt"
    torch.save(motions, path)
    return TrajectoryBuffer(str(path), ["expert"], device="cpu")


class TestHeldOutSplit:
    """The held-out rows must be unreachable by BOTH samplers, and survive eval() and a resume."""

    def test_read_from_the_bundle(self, buffer: TrajectoryBuffer) -> None:
        """The split is pinned in the bundle, not redrawn at construction."""
        assert buffer.heldout_indices.tolist() == [3, 9]

    def test_zero_weight_in_both_samplers(self, buffer: TrajectoryBuffer) -> None:
        """sample() (D's expert batch + the rollout z) and sample_states() (RSI) share one distribution."""
        assert buffer.sample_weights[[3, 9]].sum() == 0.0
        assert (buffer.sample_weights[[0, 1, 2, 4]] > 0).all()

    def test_eval_cannot_resurrect_them(self, buffer: TrajectoryBuffer) -> None:
        """eval() writes a LARGE priority to every row it scores; the held-out rows must stay at zero."""
        buffer.update_priorities(torch.full((NUM_MOTIONS,), 16.0), slice(0, NUM_MOTIONS))
        buffer.normalize_priorities()
        assert buffer.sample_weights[[3, 9]].sum() == 0.0
        drawn = torch.multinomial(buffer.sample_weights, 500, replacement=True)
        assert not torch.isin(drawn, torch.tensor([3, 9])).any(), "a held-out clip reached the sampler"

    def test_survives_a_save_load_roundtrip(self, buffer: TrajectoryBuffer, tmp_path: Path) -> None:
        """A resume that redraws the split would train on rows it then evaluates as held out."""
        buffer.set_family_softening(alpha=0.5)
        state = buffer.state_dict()
        torch.save(state, tmp_path / "ckpt.pt")

        buffer.train_mask = torch.ones(NUM_MOTIONS)  # clobber, as a fresh construction would
        buffer.base_weights = torch.ones(NUM_MOTIONS)
        buffer.load_state_dict(torch.load(tmp_path / "ckpt.pt", weights_only=False))

        assert buffer.heldout_indices.tolist() == [3, 9]
        assert torch.equal(buffer.train_mask, state["train_mask"])
        assert torch.equal(buffer.base_weights, state["base_weights"])


class TestFamilySoftening:
    """alpha reweights per clip by its family's size; it must NOT live in `priorities`."""

    def test_alpha_one_is_the_natural_corpus(self, buffer: TrajectoryBuffer) -> None:
        """Alpha = 1 -> every clip equal -> the box-dominated corpus."""
        buffer.set_family_softening(alpha=1.0)
        assert torch.allclose(buffer.base_weights, torch.ones(NUM_MOTIONS))

    def test_alpha_zero_equalizes_family_mass(self, buffer: TrajectoryBuffer) -> None:
        """Alpha = 0 -> weight ~ 1/n_f -> each family gets the same total mass."""
        buffer.set_family_softening(alpha=0.0, weight_clip=1e9)
        w = buffer.base_weights
        mass = [w[:8].sum(), w[8:11].sum(), w[11:].sum()]
        assert torch.allclose(torch.stack(mass), torch.full((3,), mass[0]), rtol=1e-5)

    def test_rare_families_are_upweighted_and_clipped(self, buffer: TrajectoryBuffer) -> None:
        """The lone family-2 clip is boosted, but the clip bounds how far."""
        buffer.set_family_softening(alpha=0.5, weight_clip=2.0)
        assert buffer.base_weights[11] > buffer.base_weights[0]
        assert buffer.base_weights.max() <= 2.0

    def test_alpha_is_not_destroyed_by_eval(self, buffer: TrajectoryBuffer) -> None:
        """The whole point of the separate tensor: eval() sweeping priorities must not erase alpha."""
        buffer.set_family_softening(alpha=0.5)
        before = buffer.base_weights.clone()
        buffer.update_priorities(torch.rand(NUM_MOTIONS) + 1.0, slice(0, NUM_MOTIONS))
        buffer.normalize_priorities()
        assert torch.equal(buffer.base_weights, before)


class TestCallerChosenRows:
    """sample(ep_indices=) bypasses the weighted draw -- the seam the terrain cone plugs into."""

    def test_uses_exactly_the_given_rows(self, buffer: TrajectoryBuffer) -> None:
        """Every returned window must come from the requested motion."""
        rows = torch.tensor([5, 5, 7])
        seq = 4
        obs, next_obs = buffer.sample(len(rows) * seq, seq_length=seq, ep_indices=rows)
        assert obs["expert"].shape == (len(rows) * seq, OBS_DIM)

        # each window's frames must match the source row's frames somewhere in the bucket
        got = obs["expert"].view(len(rows), seq, OBS_DIM)
        for i, row in enumerate(rows.tolist()):
            src = buffer.motions["expert"][row]
            assert any(torch.allclose(got[i], src[s : s + seq]) for s in range(BUCKET - seq))
        assert next_obs["expert"].shape == (len(rows) * seq, OBS_DIM)

    def test_row_count_must_match_the_window_count(self, buffer: TrajectoryBuffer) -> None:
        """A miscounted cone draw must fail loudly, not silently truncate."""
        with pytest.raises(ValueError, match="ep_indices"):
            buffer.sample(12, seq_length=4, ep_indices=torch.tensor([1, 2]))

    def test_none_falls_back_to_the_weighted_draw(self, buffer: TrajectoryBuffer) -> None:
        """The default path is unchanged, and still respects the held-out mask."""
        obs, _ = buffer.sample(64, seq_length=1)
        assert obs["expert"].shape == (64, OBS_DIM)
