# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the FuseModel, and for the FB-CPR [z; c] fusion contract built on it.

The fusion passes ``cat([z, c])`` as ONE input rather than giving c its own
parallel embedding branch, because the successor measure is a joint function of task x terrain. The
tests that matter here are the two that would fail silently otherwise: the branch/arity contract, and
that an ENCODER-LESS build is unchanged by the fusion.
"""

from __future__ import annotations

import tempfile
import torch
from tensordict import TensorDict

import onnx
import pytest

from robot_rl.models import FuseModel
from robot_rl.utils.export import save_jit, save_onnx
from tests.conftest import make_obs

NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 4
Z_DIM = 16
C_DIM = 6
TRUNK = 24  # hidden_dims[0]: the width the branches must split between them
OBS_GROUPS = {"actor": ["policy"], "critic": ["policy"]}


def _make(input_dims: tuple[int, ...], output_dim: int, obs_set: str = "actor") -> tuple[FuseModel, TensorDict]:
    obs = make_obs(NUM_ENVS, OBS_DIM)
    model = FuseModel(
        obs,
        OBS_GROUPS,
        obs_set,
        input_dims,
        output_dim,
        embedding_dims=[16],
        hidden_dims=[TRUNK, TRUNK],
        activation="elu",
    )
    return model, obs


class TestBranchContract:
    """One embedding branch per input_dims slot -- INCLUDING zeros -- and arity counts nonzero slots."""

    def test_zero_slot_is_a_real_branch(self) -> None:
        """A 0 in input_dims builds a bare-obs branch; arity still counts only nonzero slots."""
        model, _ = _make((Z_DIM, 0), NUM_ACTIONS)
        assert len(model.embeddings) == 2, "the trailing 0 is a bare-obs branch, not an arg placeholder"
        assert model.num_inputs == 1, "arity counts only NONZERO slots"

    def test_branch_widths_split_the_trunk(self) -> None:
        """Branch widths must sum to the trunk input width (the last absorbs the remainder)."""
        model, _ = _make((Z_DIM, 0), NUM_ACTIONS)
        widths = [emb[-2].out_features for emb in model.embeddings]
        assert sum(widths) == TRUNK, "concatenated embeddings must total the trunk input width"
        assert widths == [TRUNK // 2, TRUNK - TRUNK // 2]

    def test_fusing_c_into_z_keeps_two_branches(self) -> None:
        """The whole point: [z; c] is ONE input, so the actor stays at 2 branches, not 3."""
        fused, _unused_obs = _make((Z_DIM + C_DIM, 0), NUM_ACTIONS)
        assert len(fused.embeddings) == 2
        assert fused.num_inputs == 1
        # the fused branch's first layer must accept obs + z + c
        assert fused.embeddings[0][0].in_features == OBS_DIM + Z_DIM + C_DIM

    def test_arity_error_when_a_call_site_is_missed(self) -> None:
        """A missed fusion site passes a bare z where [z; c] is expected -- must FAIL, not degrade."""
        model, obs = _make((Z_DIM + C_DIM, 0), NUM_ACTIONS)
        with pytest.raises(RuntimeError):  # shape mismatch inside the embedding
            model(obs, torch.randn(NUM_ENVS, Z_DIM))
        with pytest.raises(ValueError, match="Invalid number of inputs"):  # passed z and c separately
            model(obs, torch.randn(NUM_ENVS, Z_DIM), torch.randn(NUM_ENVS, C_DIM))


class TestEncoderlessBuildIsUnchanged:
    """The regression gate: with no encoder, c_dim == 0 and the build must be IDENTICAL to pre-fusion."""

    @pytest.mark.parametrize(
        ("pre_fusion", "post_fusion"),
        [
            ((Z_DIM, 0), (Z_DIM + 0, 0)),  # actor:   (z_dim, 0, *c_dims) with c_dims=() -> (z_dim+0, 0)
            ((Z_DIM, NUM_ACTIONS), (Z_DIM + 0, NUM_ACTIONS)),  # critics / forward map
        ],
    )
    def test_shapes_identical(self, pre_fusion: tuple[int, ...], post_fusion: tuple[int, ...]) -> None:
        """With c_dim == 0 the fused build has the same slots, keys, shapes and weights as before."""
        torch.manual_seed(0)
        before, _obs = _make(pre_fusion, NUM_ACTIONS)
        torch.manual_seed(0)
        after, _ = _make(post_fusion, NUM_ACTIONS)

        assert tuple(before.input_dims) == tuple(after.input_dims)
        assert before.num_inputs == after.num_inputs
        sd_b, sd_a = before.state_dict(), after.state_dict()
        assert sd_b.keys() == sd_a.keys()
        for k in sd_b:
            assert sd_b[k].shape == sd_a[k].shape
            assert torch.equal(sd_b[k], sd_a[k]), f"same seed must give the same weights: {k}"


class TestFusedForwardAndExport:
    """Forward and export behavior of a model whose z slot carries [z; c]."""

    def test_forward_shapes(self) -> None:
        """A fused actor and a fused critic both accept one [z; c] tensor."""
        actor, obs = _make((Z_DIM + C_DIM, 0), NUM_ACTIONS)
        out = actor(obs, torch.randn(NUM_ENVS, Z_DIM + C_DIM))
        assert out.shape == (NUM_ENVS, NUM_ACTIONS)

        critic, obs = _make((Z_DIM + C_DIM, NUM_ACTIONS), 1, obs_set="critic")
        q = critic(obs, torch.randn(NUM_ENVS, Z_DIM + C_DIM), torch.randn(NUM_ENVS, NUM_ACTIONS))
        assert q.shape == (NUM_ENVS, 1)

    def test_export_signature_collapses_to_one_extra_input(self) -> None:
        """Exported actor goes from (obs, z, c) to (obs, zc): input_1 disappears. Deployment concatenates."""
        actor, _ = _make((Z_DIM + C_DIM, 0), NUM_ACTIONS)
        exported = actor.as_onnx()
        assert exported.other_dims == [Z_DIM + C_DIM]
        assert exported.input_names == ["obs", "input_0"]
        dummies = exported.get_dummy_inputs()
        assert len(dummies) == 2
        assert dummies[1].shape == (1, Z_DIM + C_DIM)

    def test_jit_and_onnx_roundtrip(self) -> None:
        """The fused actor traces and exports cleanly."""
        actor, _ = _make((Z_DIM + C_DIM, 0), NUM_ACTIONS)
        with tempfile.TemporaryDirectory() as tmp:
            jit_path = save_jit(actor.as_jit(), tmp, "policy.pt")
            loaded = torch.jit.load(jit_path)
            out = loaded(torch.randn(1, OBS_DIM), torch.randn(1, Z_DIM + C_DIM))
            assert out.shape == (1, NUM_ACTIONS)

            onnx_path = save_onnx(actor.as_onnx(), tmp, "policy.onnx")
            onnx.checker.check_model(onnx.load(onnx_path))
