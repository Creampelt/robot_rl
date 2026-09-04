"""Left-right mirroring of observation groups as signed permutations, built from the env's own term layout."""

from __future__ import annotations

import re
import torch
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from tensordict import TensorDict

# How each observation term function mirrors under a reflection across the robot's sagittal plane
# (y -> -y). Vectors keep x and z; pseudovectors (angular velocities) keep only y; quaternions
# (w, x, y, z) become (w, -x, y, -z); joint terms permute left<->right and negate roll/yaw axes.
_VECTOR = ("projected_gravity", "root_pos_w", "root_lin_vel_w", "base_lin_vel", "body_pos_h", "body_lin_vel_h",
           "frame_pos", "frame_lin_vel_b", "root_lin_vel_b")
_PSEUDOVECTOR = ("base_ang_vel", "root_ang_vel_w", "root_ang_vel_b", "body_ang_vel_h")
_QUATERNION = ("root_quat_w",)
_ROTATION = ("body_rot_h", "frame_rot")
_JOINT = ("joint_pos", "joint_pos_rel", "joint_vel", "joint_vel_rel", "last_action")
_SCALAR = ("base_pos_z",)
_BODY_TERMS = ("body_pos_h", "body_rot_h", "body_lin_vel_h", "body_ang_vel_h")


@dataclass
class MirrorSpec:
    """How to pair and sign-flip names for one robot.

    Args:
        pair_patterns: Substring pairs whose swap maps a left name to its right partner (and back).
        negate_joints: Regex over joint names whose angle changes sign under the reflection (roll and
            yaw axes, i.e. axes lying in the sagittal plane).
        term_rules: Optional per-term overrides keyed by ``"<group>/<term>"`` with a rule name from
            ``ObsMirror.RULES``; needed only for term functions the table above does not know.
    """

    pair_patterns: Sequence[tuple[str, str]] = (("Left", "Right"), ("left", "right"))
    negate_joints: str = r"Roll|Yaw|yaw|Waist"
    term_rules: dict[str, str] = field(default_factory=dict)

    def partner(self, name: str) -> str:
        for a, b in self.pair_patterns:
            if a in name:
                return name.replace(a, b)
            if b in name:
                return name.replace(b, a)
        return name


def _pair_permutation(names: Sequence[str], spec: MirrorSpec) -> torch.Tensor:
    """Index of each name's mirror partner (itself for midline names); a missing partner is an error."""
    index = {n: i for i, n in enumerate(names)}
    perm = []
    for n in names:
        p = spec.partner(n)
        if p not in index:
            raise ValueError(f"{n!r} mirrors to {p!r}, which is not in the list; add a pair pattern or the body")
        perm.append(index[p])
    return torch.tensor(perm, dtype=torch.long)


class ObsMirror:
    """Mirror observation groups (and actions) across the sagittal plane.

    Every group is a concatenation of terms, and every term mirrors as a signed permutation of its
    entries, so the whole group does too: ``mirrored = obs[..., perm] * sign``. The permutation is
    derived from the env's observation manager (term order, sizes, history length, body subsets) and
    the articulation's joint/body names, so it cannot drift from what a stored bundle contains.
    """

    RULES = ("vector", "pseudovector", "quaternion", "rotation", "joint", "scalar")

    def __init__(self, group_perm: dict[str, torch.Tensor], group_sign: dict[str, torch.Tensor],
                 action_perm: torch.Tensor, action_sign: torch.Tensor) -> None:
        self.group_perm = group_perm
        self.group_sign = group_sign
        self.action_perm = action_perm
        self.action_sign = action_sign

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_env(cls, env, spec: MirrorSpec) -> ObsMirror:
        """Build the mirror from a wrapped Isaac Lab env (anything exposing ``.unwrapped``)."""
        base = env.unwrapped
        obs_mgr = base.observation_manager
        robot = base.scene["robot"]
        joint_perm = _pair_permutation(robot.joint_names, spec)
        joint_sign = torch.tensor(
            [-1.0 if re.search(spec.negate_joints, n) else 1.0 for n in robot.joint_names]
        )
        group_perm, group_sign = {}, {}
        for group, names in obs_mgr.active_terms.items():
            dims = obs_mgr.group_obs_term_dim[group]
            cfgs = obs_mgr._group_obs_term_cfgs[group]
            perms, signs, offset = [], [], 0
            for name, dim, cfg in zip(names, dims, cfgs):
                total = int(torch.tensor(dim).prod().item())
                rule = spec.term_rules.get(f"{group}/{name}") or cls._rule_for(cfg)
                per_frame = total // max(cfg.history_length, 1)
                if rule == "joint":
                    p, s = joint_perm, joint_sign
                elif rule == "scalar":
                    p, s = torch.arange(per_frame), torch.ones(per_frame)
                else:
                    bodies = cls._term_bodies(cfg, robot.body_names)
                    p, s = cls._geometric(rule, per_frame, bodies, spec)
                if p.numel() != per_frame:
                    raise ValueError(f"{group}/{name}: rule {rule!r} produced {p.numel()} entries for a {per_frame}-wide term")
                reps = max(cfg.history_length, 1)  # history is flattened time-major within the term
                perms.append(torch.cat([offset + k * per_frame + p for k in range(reps)]))
                signs.append(s.repeat(reps))
                offset += total
            group_perm[group] = torch.cat(perms)
            group_sign[group] = torch.cat(signs)
        return cls(group_perm, group_sign, joint_perm, joint_sign)

    @staticmethod
    def _rule_for(cfg) -> str:
        fn = cfg.func.__name__ if callable(cfg.func) else str(cfg.func).rsplit(".", 1)[-1]
        if fn in _JOINT:
            return "joint"
        if fn in _SCALAR:
            return "scalar"
        if fn in _QUATERNION:
            return "quaternion"
        if fn in _ROTATION:
            fmt = cfg.params.get("format", "quat")
            return {"tan_normal": "rotation", "quat": "quaternion"}.get(fmt) or _unsupported(fn, fmt)
        if fn in _PSEUDOVECTOR:
            return "pseudovector"
        if fn in _VECTOR:
            return "vector"
        raise ValueError(f"no mirror rule for observation term function {fn!r}; add one to MirrorSpec.term_rules")

    @staticmethod
    def _term_bodies(cfg, all_bodies: Sequence[str]) -> list[str] | None:
        """Body names a per-body term iterates over, in its own order, or None for single-frame terms."""
        fn = cfg.func.__name__ if callable(cfg.func) else str(cfg.func).rsplit(".", 1)[-1]
        if fn not in _BODY_TERMS:
            return None
        asset_cfg = cfg.params.get("asset_cfg")
        if asset_cfg is None or asset_cfg.body_ids is None or isinstance(asset_cfg.body_ids, slice):
            return list(all_bodies)
        return [all_bodies[i] for i in asset_cfg.body_ids]

    @staticmethod
    def _geometric(rule: str, width: int, bodies: list[str] | None, spec: MirrorSpec) -> tuple[torch.Tensor, torch.Tensor]:
        block_sign = {
            "vector": [1.0, -1.0, 1.0],
            "pseudovector": [-1.0, 1.0, -1.0],
            "quaternion": [1.0, -1.0, 1.0, -1.0],
            "rotation": [1.0, -1.0, 1.0, 1.0, -1.0, 1.0],  # tangent (x column) then normal (z column)
        }[rule]
        k = len(block_sign)
        n_blocks = width // k
        if n_blocks * k != width:
            raise ValueError(f"rule {rule!r} needs a multiple of {k} entries, got {width}")
        body_perm = _pair_permutation(bodies, spec) if bodies is not None and len(bodies) == n_blocks else torch.arange(n_blocks)
        if bodies is not None and len(bodies) != n_blocks:
            raise ValueError(f"term lists {len(bodies)} bodies but holds {n_blocks} blocks of {k}")
        perm = (body_perm[:, None] * k + torch.arange(k)[None, :]).reshape(-1)
        sign = torch.tensor(block_sign).repeat(n_blocks)
        return perm, sign

    # ------------------------------------------------------------------ application

    def to(self, device) -> ObsMirror:
        self.group_perm = {k: v.to(device) for k, v in self.group_perm.items()}
        self.group_sign = {k: v.to(device) for k, v in self.group_sign.items()}
        self.action_perm, self.action_sign = self.action_perm.to(device), self.action_sign.to(device)
        return self

    def flip_group(self, group: str, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.group_perm[group]] * self.group_sign[group]

    def flip_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions[..., self.action_perm] * self.action_sign

    def apply(self, obs: TensorDict, mask: torch.Tensor | None = None) -> TensorDict:
        """Mirror the groups this mirror knows; ``mask`` (bool, batch-shaped) selects rows, None = all."""
        out = obs.clone()
        for group in obs.keys():
            if group not in self.group_perm:
                continue
            flipped = self.flip_group(group, obs[group])
            if mask is not None:
                m = mask.reshape(*mask.shape, *([1] * (flipped.ndim - mask.ndim)))
                flipped = torch.where(m, flipped, obs[group])
            out[group] = flipped
        return out

    def augment(self, env=None, obs: TensorDict | None = None, actions: torch.Tensor | None = None):
        """PPO-style data augmentation: original samples followed by their mirror images."""
        obs_out = torch.cat([obs, self.apply(obs)], dim=0) if obs is not None else None
        act_out = torch.cat([actions, self.flip_actions(actions)], dim=0) if actions is not None else None
        return obs_out, act_out


def _unsupported(fn: str, fmt: str):
    raise ValueError(f"no mirror rule for {fn!r} in format {fmt!r}")


def make_augmentation_func(env, spec: MirrorSpec) -> Callable:
    """A ``Symmetry.data_augmentation_func`` for the PPO extension, from the same mirror."""
    mirror = ObsMirror.from_env(env, spec)

    def augment(env=None, obs=None, actions=None):
        dev = obs.device if obs is not None else actions.device
        return mirror.to(dev).augment(env, obs, actions)

    return augment
