"""Rebuild trained policies from a checkpoint + train cfg (no env needed) and export them."""

from __future__ import annotations

import copy
import os
import re
import torch
import torch.nn as nn
from typing import TYPE_CHECKING

from robot_rl.modules.dict_module import DictModule
from robot_rl.modules.mlp import MLP
from robot_rl.utils.utils import resolve_callable

if TYPE_CHECKING:
    from robot_rl.algorithms import FbCpr
    from robot_rl.models import MLPModel


class _ExportWrapperMixin:
    """Gives export wrapper modules the same ``as_jit()``/``as_onnx()`` interface as MLPModel.

    The wrappers are already export-friendly, so both return ``self``; ``jit_trace`` tells
    :func:`save_jit` to trace (FuseModel submodules don't script).
    """

    jit_trace = True

    def as_jit(self) -> nn.Module:
        return self  # type: ignore[return-value]

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return self  # type: ignore[return-value]


class _BfmZeroPolicyExport(_ExportWrapperMixin, nn.Module):
    """FB-CPR actor export.

    The (external) actor-group normalizer baked in front of the actor's own export module
    (``actor.as_jit()``).
    """

    def __init__(self, alg: FbCpr) -> None:
        super().__init__()
        actor = alg.get_policy()
        if len(actor.obs_groups) != 1:
            raise NotImplementedError(
                f"BFM-Zero policy export supports a single actor obs group, got {actor.obs_groups}."
            )
        self.ext_normalizer = copy.deepcopy(alg.obs_normalizer.modules_dict[actor.obs_groups[0]])
        self.policy = actor.as_jit()
        self.obs_dim = self.policy.obs_dim
        self.z_dim = self.policy.other_dims[0]

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return the deterministic joint action for observation ``obs`` and task latent ``z``."""
        return self.policy(self.ext_normalizer(obs), z)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(1, self.obs_dim), torch.zeros(1, self.z_dim))

    @property
    def input_names(self) -> list[str]:
        return ["obs", "z"]

    @property
    def output_names(self) -> list[str]:
        return ["action"]


class _BackwardMapExport(_ExportWrapperMixin, nn.Module):
    """``(state, obs) -> z``: per-group BatchNorm (eval) -> concat -> backward MLP (ball-normed)."""

    def __init__(self, bn_state: nn.BatchNorm1d, bn_obs: nn.BatchNorm1d, mlp: MLP) -> None:
        super().__init__()
        self.bn_state = bn_state
        self.bn_obs = bn_obs
        self.mlp = mlp

    def forward(self, state: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat((self.bn_state(state), self.bn_obs(obs)), dim=-1))

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(1, self.bn_state.num_features), torch.zeros(1, self.bn_obs.num_features))

    @property
    def input_names(self) -> list[str]:
        return ["state", "obs"]

    @property
    def output_names(self) -> list[str]:
        return ["z"]


def _load_bn(nsd: dict[str, torch.Tensor], key: str) -> nn.BatchNorm1d:
    """Rebuild one per-group BatchNorm1d normalizer from the checkpoint normalizer state dict."""
    prefix = f"modules_dict.{key}" if f"modules_dict.{key}.running_mean" in nsd else key
    bn = nn.BatchNorm1d(nsd[f"{prefix}.running_mean"].shape[0], momentum=0.01, affine=False)
    bn.load_state_dict({
        "running_mean": nsd[f"{prefix}.running_mean"],
        "running_var": nsd[f"{prefix}.running_var"],
        "num_batches_tracked": nsd[f"{prefix}.num_batches_tracked"],
    })
    return bn


def _group_dims(nsd: dict[str, torch.Tensor]) -> dict[str, int]:
    """Obs-group name -> flat dimension, read from the normalizer running stats."""
    dims = {}
    for k, v in nsd.items():
        if k.endswith(".running_mean"):
            dims[k.removeprefix("modules_dict.").removesuffix(".running_mean")] = v.shape[0]
    return dims


def _num_actions(actor_sd: dict[str, torch.Tensor]) -> int:
    """Infer the action dimension from the distribution's per-action parameter vector."""
    for k, v in actor_sd.items():
        if "distribution" in k and isinstance(v, torch.Tensor) and v.ndim == 1:
            return v.shape[0]
    raise ValueError("Cannot infer num_actions from the actor state dict (no 1-D distribution parameter).")


def rebuild_ppo_policy(train_cfg: dict, ckpt: dict) -> MLPModel:
    """Rebuild a PPO actor from its checkpoint + train cfg.

    A W&B-logged train cfg has ``class_name`` keys popped by construction; defaults are substituted.
    """
    if "memory_state_dict" in ckpt:
        raise NotImplementedError("Export of recurrent (memory-bearing) policies is not supported.")
    asd = ckpt["actor_state_dict"]
    cfg = copy.deepcopy(train_cfg)

    actor_cfg = dict(cfg["actor"])
    actor_class = resolve_callable(actor_cfg.pop("class_name", "MLPModel"))
    dist_cfg = actor_cfg.get("distribution_cfg")
    if dist_cfg is not None:
        dist_cfg.setdefault("class_name", "GaussianDistribution")

    # total obs dim from the first MLP layer (no auxiliary inputs feed a PPO actor)
    first_idx = min(int(m.group(1)) for k in asd if (m := re.match(r"mlp\.(\d+)\.weight", k)))
    obs_dim = asd[f"mlp.{first_idx}.weight"].shape[1]

    groups = cfg["obs_groups"]["actor"]
    # only the concatenated dim matters for layer sizes; put it all on the first group
    obs = {g: torch.zeros(1, obs_dim if i == 0 else 0) for i, g in enumerate(groups)}
    actor = actor_class(obs, {"actor": groups}, "actor", _num_actions(asd), **actor_cfg)
    actor.load_state_dict(asd, strict=True)
    return actor.eval()


def rebuild_bfmzero_actor(train_cfg: dict, ckpt: dict) -> _BfmZeroPolicyExport:
    """Rebuild the FB-CPR actor + its normalizer from the checkpoint and wrap it for export."""
    import types

    cfg = copy.deepcopy(train_cfg)
    nsd = ckpt["obs_normalizer_state_dict"]
    asd = ckpt["actor_state_dict"]
    obs = {group: torch.zeros(1, dim) for group, dim in _group_dims(nsd).items()}
    obs_groups = {"actor": cfg["obs_groups"]["actor"]}
    z_dim = cfg["algorithm"]["z_dim"]

    actor_cfg = dict(cfg["actor"])
    actor_class = resolve_callable(actor_cfg.pop("class_name", "ResidualFuseModel"))
    dist_cfg = actor_cfg.get("distribution_cfg")
    if dist_cfg is not None:
        dist_cfg.setdefault("class_name", "TruncatedGaussianDistribution")
        dist_cfg.setdefault("low", -cfg["clip_actions"])
        dist_cfg.setdefault("high", cfg["clip_actions"])
    actor = actor_class(obs, obs_groups, "actor", (z_dim, 0), _num_actions(asd), **actor_cfg)
    actor.load_state_dict(asd, strict=True)

    normalizer = DictModule({g: _load_bn(nsd, g) for g in obs_groups["actor"]})
    alg_shim = types.SimpleNamespace(get_policy=lambda: actor, obs_normalizer=normalizer)
    return _BfmZeroPolicyExport(alg_shim).eval()  # type: ignore[arg-type]


def rebuild_bfmzero_backward(train_cfg: dict, ckpt: dict) -> _BackwardMapExport:
    """Rebuild the FB-CPR backward map + its normalizers from the checkpoint."""
    nsd = ckpt["obs_normalizer_state_dict"]
    bsd = ckpt["backward_map_state_dict"]
    bn_state, bn_obs = _load_bn(nsd, "state"), _load_bn(nsd, "obs")

    mlp_sd = {k.removeprefix("mlp."): v for k, v in bsd.items() if k.startswith("mlp.")}
    z_dim = mlp_sd[sorted(k for k in mlp_sd if k.endswith(".weight"))[-1]].shape[0]
    bw_keys = (
        "hidden_dims",
        "num_parallel",
        "activation",
        "first_activation",
        "last_activation",
        "normalize_first_layer",
    )
    bw_cfg = {k: v for k, v in train_cfg["backward_map"].items() if k in bw_keys}
    mlp = MLP(bn_state.num_features + bn_obs.num_features, z_dim, **bw_cfg)
    mlp.load_state_dict(mlp_sd, strict=True)

    module = _BackwardMapExport(bn_state, bn_obs, mlp).eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


class _FuseModelExport(_ExportWrapperMixin, nn.Module):
    """Export wrapper for a model with EXTERNAL per-group normalization.

    ``forward(obs_concat, *others)`` splits the obs by group, normalizes each slice, and runs the
    model's own export module (``model.as_jit()``; MLPModel-based exports take one concatenated
    input).
    """

    def __init__(self, model: nn.Module, normalizers: list[nn.BatchNorm1d], other_dims: list[int]) -> None:
        super().__init__()
        self.inner = model.as_jit()
        self.single_input = not hasattr(self.inner, "other_dims")
        self.normalizers = nn.ModuleList(normalizers)
        self.group_dims = [bn.num_features for bn in normalizers]
        self.other_dims = list(other_dims)

    def forward(self, obs: torch.Tensor, *others: torch.Tensor) -> torch.Tensor:
        slices = torch.split(obs, self.group_dims, dim=-1)
        normed = torch.cat([bn(x) for bn, x in zip(self.normalizers, slices, strict=True)], dim=-1)
        if self.single_input:
            return self.inner(torch.cat([normed, *others], dim=-1))
        return self.inner(normed, *others)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        return (torch.zeros(1, sum(self.group_dims)), *(torch.zeros(1, d) for d in self.other_dims))

    @property
    def input_names(self) -> list[str]:
        return ["obs", *[f"input_{i}" for i in range(len(self.other_dims))]]

    @property
    def output_names(self) -> list[str]:
        return ["output"]


# FB-CPR auxiliary models: cfg key -> (obs set, default class, output spec, other-input spec)
_FBCPR_AUX_MODELS = {
    "forward_map": ("critic", "ResidualFuseModel", "z_dim", ("z_dim", "num_actions")),
    "disc_critic": ("critic", "ResidualFuseModel", 1, ("z_dim", "num_actions")),
    "aux_critic": ("critic", "ResidualFuseModel", 1, ("z_dim", "num_actions")),
    "discriminator": ("discriminator", "DiscriminatorModel", 1, ("z_dim",)),
}


def rebuild_fbcpr_aux(train_cfg: dict, ckpt: dict, name: str) -> _FuseModelExport:
    """Rebuild one of the FB-CPR auxiliary models (forward map, critics, discriminator) for export."""
    obs_set, default_class, out_spec, other_spec = _FBCPR_AUX_MODELS[name]
    cfg = copy.deepcopy(train_cfg)
    nsd = ckpt["obs_normalizer_state_dict"]
    z_dim = cfg["algorithm"]["z_dim"]
    dims = {"z_dim": z_dim, "num_actions": _num_actions(ckpt["actor_state_dict"])}
    other_dims = [dims[k] for k in other_spec]
    out_dim = dims[out_spec] if isinstance(out_spec, str) else out_spec

    obs = {group: torch.zeros(1, dim) for group, dim in _group_dims(nsd).items()}
    model_cfg = dict(cfg[name])
    model_class = resolve_callable(model_cfg.pop("class_name", default_class))
    if name == "discriminator":
        model = model_class(obs, cfg["obs_groups"], obs_set, out_dim, other_input_dims=tuple(other_dims), **model_cfg)
    else:
        model = model_class(obs, cfg["obs_groups"], obs_set, tuple(other_dims), out_dim, **model_cfg)
    model.load_state_dict(ckpt[f"{name}_state_dict"], strict=True)

    groups = cfg["obs_groups"][obs_set]
    normalizers = [_load_bn(nsd, g) for g in groups]
    module = _FuseModelExport(model.eval(), normalizers, other_dims).eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def rebuild_ppo_critic(train_cfg: dict, ckpt: dict) -> nn.Module:
    """Rebuild a PPO critic from its checkpoint + train cfg."""
    csd = ckpt["critic_state_dict"]
    cfg = copy.deepcopy(train_cfg)
    critic_cfg = dict(cfg["critic"])
    critic_class = resolve_callable(critic_cfg.pop("class_name", "MLPModel"))
    first_idx = min(int(m.group(1)) for k in csd if (m := re.match(r"mlp\.(\d+)\.weight", k)))
    obs_dim = csd[f"mlp.{first_idx}.weight"].shape[1]
    groups = cfg["obs_groups"]["critic"]
    obs = {g: torch.zeros(1, obs_dim if i == 0 else 0) for i, g in enumerate(groups)}
    critic = critic_class(obs, {"critic": groups}, "critic", 1, **critic_cfg)
    critic.load_state_dict(csd, strict=True)
    return critic.eval()


def save_jit(module: nn.Module, path: str, filename: str) -> str:
    """Save a TorchScript export (modules with ``jit_trace`` set are traced; FuseModels don't script)."""
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    with torch.no_grad():
        if getattr(module, "jit_trace", False):
            scripted = torch.jit.trace(module, module.get_dummy_inputs())
        else:
            scripted = torch.jit.script(module)
    scripted.save(save_path)
    return save_path


def save_onnx(module: nn.Module, path: str, filename: str, verbose: bool = False) -> str:
    """Save an ONNX export; the module must provide dummy inputs and input/output names."""
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    torch.onnx.export(
        module,
        module.get_dummy_inputs(),
        save_path,
        export_params=True,
        opset_version=18,
        verbose=verbose,
        input_names=module.input_names,
        output_names=module.output_names,
    )
    return save_path
