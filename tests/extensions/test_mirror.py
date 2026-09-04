"""Tests for the left-right observation mirror."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from robot_rl.extensions.mirror import MirrorSpec, ObsMirror, _pair_permutation

SPEC = MirrorSpec(pair_patterns=[("Left", "Right"), ("AL", "AR")], negate_joints=r"Roll|Yaw")


def _mirror() -> ObsMirror:
    """A two-group mirror: 4 joints (pitch/roll pairs) and 2 paired bodies with position + angular velocity."""
    joints = ["Left_Pitch", "Left_Roll", "Right_Pitch", "Right_Roll"]
    jperm = _pair_permutation(joints, SPEC)
    jsign = torch.tensor([1.0, -1.0, 1.0, -1.0])
    pos_perm, pos_sign = ObsMirror._geometric("vector", 6, ["AL1", "AR1"], SPEC)
    ang_perm, ang_sign = ObsMirror._geometric("pseudovector", 6, ["AL1", "AR1"], SPEC)
    state_perm = torch.cat([pos_perm, 6 + ang_perm])
    state_sign = torch.cat([pos_sign, ang_sign])
    return ObsMirror({"joints": jperm, "state": state_perm}, {"joints": jsign, "state": state_sign}, jperm, jsign)


class TestPairing:
    def test_partner_lookup_swaps_sides_and_keeps_midline(self) -> None:
        perm = _pair_permutation(["Trunk", "Left_Hip", "Right_Hip", "AL1", "AR1"], SPEC)
        assert perm.tolist() == [0, 2, 1, 4, 3]

    def test_missing_partner_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="not in the list"):
            _pair_permutation(["Left_Hip"], SPEC)


class TestApply:
    def test_is_an_involution(self) -> None:
        m = _mirror()
        obs = TensorDict({"joints": torch.randn(5, 4), "state": torch.randn(5, 12)}, batch_size=[5])
        twice = m.apply(m.apply(obs))
        for k in obs.keys():
            assert torch.equal(twice[k], obs[k])

    def test_joint_rule_swaps_sides_and_negates_roll(self) -> None:
        m = _mirror()
        obs = TensorDict({"joints": torch.tensor([[1.0, 2.0, 3.0, 4.0]])}, batch_size=[1])
        assert m.apply(obs)["joints"].tolist() == [[3.0, -4.0, 1.0, -2.0]]

    def test_vector_and_pseudovector_signs(self) -> None:
        m = _mirror()
        pos = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])  # AL1 then AR1 positions
        ang = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        out = m.apply(TensorDict({"state": torch.cat([pos, ang], -1)}, batch_size=[1]))["state"][0]
        assert out[:6].tolist() == [4.0, -5.0, 6.0, 1.0, -2.0, 3.0]  # bodies swapped, y negated
        assert out[6:].tolist() == [-4.0, 5.0, -6.0, -1.0, 2.0, -3.0]  # bodies swapped, x and z negated

    def test_mask_selects_rows(self) -> None:
        m = _mirror()
        obs = TensorDict({"joints": torch.randn(4, 4), "state": torch.randn(4, 12)}, batch_size=[4])
        mask = torch.tensor([True, False, True, False])
        out = m.apply(obs, mask)
        assert torch.equal(out["joints"][1], obs["joints"][1]) and not torch.equal(out["joints"][0], obs["joints"][0])

    def test_unknown_groups_pass_through(self) -> None:
        m = _mirror()
        obs = TensorDict({"joints": torch.randn(2, 4), "other": torch.randn(2, 3)}, batch_size=[2])
        assert torch.equal(m.apply(obs)["other"], obs["other"])

    def test_augment_stacks_original_then_mirror(self) -> None:
        m = _mirror()
        obs = TensorDict({"joints": torch.randn(3, 4)}, batch_size=[3])
        actions = torch.randn(3, 4)
        obs_out, act_out = m.augment(None, obs, actions)
        assert obs_out.batch_size[0] == 6 and act_out.shape == (6, 4)
        assert torch.equal(obs_out["joints"][:3], obs["joints"])
        assert torch.equal(act_out[3:], m.flip_actions(actions))


class TestGeometric:
    def test_quaternion_rule(self) -> None:
        perm, sign = ObsMirror._geometric("quaternion", 4, None, SPEC)
        assert perm.tolist() == [0, 1, 2, 3] and sign.tolist() == [1.0, -1.0, 1.0, -1.0]

    def test_width_must_match_block(self) -> None:
        with pytest.raises(ValueError, match="multiple of 3"):
            ObsMirror._geometric("vector", 7, None, SPEC)
