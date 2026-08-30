from __future__ import annotations

import math
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from robot_rl.env import URLVecEnv
from robot_rl.models import FuseModel, MLPModel
from robot_rl.modules import DictModule, TargetNetwork
from robot_rl.storage import OfflineTransitionDataset, ReplayBuffer
from robot_rl.utils import (
    compute_td_targets,
    resolve_callable,
    resolve_dtype,
    resolve_obs_groups,
    resolve_optimizer,
)


class Fb:
    r"""Forward-Backward representations trained offline from a fixed transition dataset.

    Learns :math:`F(s, a, z)` and :math:`B(s')` whose inner product approximates the successor measure, so
    that a reward :math:`r` is served zero-shot by :math:`z = \mathbb{E}[r(s) B(s)]` and the policy
    :math:`\pi_z(s) = \arg\max_a F(s, a, z)^\top z`. Unlike :class:`FbCpr` there is no environment
    interaction, no discriminator and no auxiliary critic: every gradient step reads a stored transition.

    The dataset's action distribution is the only behaviour the actor ever sees, so
    :attr:`behavior_reg_coef` is the lever against action-distribution shift; at 0 this is the published
    offline FB objective and relies on ensemble pessimism alone.
    """

    ACTOR_KEY = "actor"

    def __init__(
        self,
        actor: FuseModel,
        forward_map: FuseModel,
        backward_map: MLPModel,
        obs_normalizer: DictModule[nn.BatchNorm1d],
        dataset: ReplayBuffer,
        z_dim: int,
        actor_learning_rate: float = 1e-4,
        forward_learning_rate: float = 1e-4,
        backward_learning_rate: float = 1e-4,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        max_grad_norm: float | None = None,
        clip_actor_std: float = 0.2,
        train_goal_ratio: float = 0.5,
        forward_backward_pessimism: float = 0.0,
        actor_pessimism: float = 0.5,
        behavior_reg_coef: float = 0.0,
        value_loss_coef: float = 0.0,
        ortho_loss_coef: float = 1.0,
        fb_tau: float = 0.01,
        batch_size: int = 1024,
        device: str = "cpu",
        dtype: str = "float32",
        compile_mode: str | None = None,
        clip_actions: float | None = None,
        multi_gpu_cfg: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the algorithm with models, dataset, and optimization settings."""
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.dtype = resolve_dtype(dtype)
        self._autocast_enabled = self.dtype in (torch.float16, torch.bfloat16)
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.clip_actor_std = clip_actor_std
        self.train_goal_ratio = train_goal_ratio
        self.forward_backward_pessimism = forward_backward_pessimism
        self.actor_pessimism = actor_pessimism
        self.behavior_reg_coef = behavior_reg_coef
        self.value_loss_coef = value_loss_coef
        self.ortho_loss_coef = ortho_loss_coef
        self.fb_tau = fb_tau
        self.clip_actions = clip_actions

        self.actor = actor.to(device)
        self.forward_map = forward_map.to(device)
        self.backward_map = backward_map.to(device)
        self.obs_normalizer = obs_normalizer.to(device)

        # ParallelLinear allocates its weights uninitialized, so the models must be initialized here
        for model in self.models:
            model.init_weights()

        self.target_forward_map = TargetNetwork(self.forward_map, tau=fb_tau).to(device)
        self.target_backward_map = TargetNetwork(self.backward_map, tau=fb_tau).to(device)
        self.dataset = dataset

        optimizer_class = resolve_optimizer(optimizer)
        self.actor_optimizer = optimizer_class(
            self.actor.parameters(), lr=actor_learning_rate, weight_decay=weight_decay
        )
        self.forward_optimizer = optimizer_class(
            self.forward_map.parameters(), lr=forward_learning_rate, weight_decay=weight_decay
        )
        self.backward_optimizer = optimizer_class(
            self.backward_map.parameters(), lr=backward_learning_rate, weight_decay=weight_decay
        )

        # off-diagonal mask for the FB and orthonormality losses; built once at the training batch size
        off_diag = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        self._off_diag = off_diag.unsqueeze(0)
        self._off_diag_sum = off_diag.sum()

        if compile_mode is not None:
            self._update_forward_backward = torch.compile(self._update_forward_backward, mode=compile_mode)
            self._update_actor = torch.compile(self._update_actor, mode=compile_mode)

    @property
    def models(self) -> list[MLPModel]:
        """The learnable models, for train/eval mode switching and checkpointing."""
        return [self.actor, self.forward_map, self.backward_map]

    def project_z(self, z: torch.Tensor) -> torch.Tensor:
        """Project ``z`` onto the sphere of radius ``sqrt(z_dim)``."""
        return math.sqrt(z.shape[-1]) * nn.functional.normalize(z, dim=-1)

    def _sample_random_z(self, size: int) -> torch.Tensor:
        """Draw ``size`` latents uniformly on the sphere."""
        return self.project_z(torch.randn((size, self.z_dim), dtype=torch.float32, device=self.device))

    @torch.no_grad()
    def _sample_z(self, goal_obs: TensorDict) -> torch.Tensor:
        """Mix random latents with latents encoding states drawn from the dataset.

        Args:
            goal_obs: Observations whose encodings serve as goal-reaching latents.

        Returns:
            Latents of shape ``(batch_size, z_dim)``.
        """
        with torch.autocast(device_type=self.device, dtype=self.dtype, enabled=self._autocast_enabled):
            z = self._sample_random_z(self.batch_size)
            perm = torch.randperm(self.batch_size, device=self.device)
            goal_z = self.project_z(self.backward_map(goal_obs[perm]))
            take_goal = torch.rand(self.batch_size, 1, device=self.device) < self.train_goal_ratio
        return torch.where(take_goal, goal_z, z)

    def _soft_update_targets(self) -> None:
        """Blend the forward and backward targets toward the live networks."""
        self.target_forward_map.update()
        self.target_backward_map.update()

    def update(self) -> tuple[dict[str, torch.Tensor], dict]:
        """Run one offline update over a sampled mini-batch and return its losses and extras."""
        batch = self.dataset.sample_mini_batch(self.device)
        with torch.no_grad(), torch.autocast(device_type=self.device, dtype=self.dtype, enabled=self._autocast_enabled):
            batch.observations = self.obs_normalizer(batch.observations)
            batch.next_observations = self.obs_normalizer(batch.next_observations)
            # Every latent is drawn here: a stored transition carries no z to preserve, and a zero
            # context is off the sphere and yields a degenerate value.
            batch.context = self._sample_z(batch.next_observations)

        torch.compiler.cudagraph_mark_step_begin()
        fb_losses, fb_extras = self._update_forward_backward(batch)
        actor_losses, actor_extras = self._update_actor(batch)
        self._soft_update_targets()
        return {**fb_losses, **actor_losses}, {**fb_extras, **actor_extras}

    def _update_forward_backward(self, batch: ReplayBuffer.Batch) -> tuple[dict[str, torch.Tensor], dict]:
        with torch.autocast(device_type=self.device, dtype=self.dtype, enabled=self._autocast_enabled):
            with torch.no_grad():
                next_actions = self.actor(
                    batch.next_observations, batch.context, stochastic_output=True, std_clip=self.clip_actor_std
                )
                target_Fs = self.target_forward_map(batch.next_observations, batch.context, next_actions)
                target_B = self.target_backward_map(batch.next_observations)
                target_Ms = torch.matmul(target_Fs, target_B.T)
                target_M = compute_td_targets(target_Ms, self.forward_backward_pessimism)

            Fs = self.forward_map(batch.observations, batch.context, batch.actions)
            B = self.backward_map(batch.next_observations)
            Ms = torch.matmul(Fs, B.T)

            diff = Ms - batch.gammas * target_M
            fb_offdiag = 0.5 * (diff * self._off_diag).pow(2).sum() / self._off_diag_sum
            fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
            fb_loss = fb_offdiag + fb_diag

            Cov = torch.matmul(B, B.T)
            orth_offdiag = 0.5 * (Cov * self._off_diag).pow(2).sum() / self._off_diag_sum
            orth_diag = -Cov.diag().mean()
            orth_loss = self.ortho_loss_coef * (orth_offdiag + orth_diag)

            q_loss = torch.zeros(1, device=self.device, dtype=torch.float32)
            if self.value_loss_coef > 0.0:
                with torch.no_grad():
                    next_Qs = (target_Fs * batch.context).sum(dim=-1)
                    next_Q = compute_td_targets(next_Qs, self.forward_backward_pessimism)
                    # disable autocast so cov and B share a dtype
                    with torch.autocast(device_type=self.device, dtype=self.dtype, enabled=False):
                        cov = torch.matmul(B.T, B) / B.shape[0]
                    B_inv_cov = torch.linalg.solve(cov, B, left=False)
                    implicit_reward = (B_inv_cov * batch.context).sum(dim=-1)
                    target_Q = implicit_reward.detach() + batch.gammas.squeeze() * next_Q
                    target_Q = target_Q.expand(Fs.shape[0], -1)
                Qs = (Fs * batch.context).sum(dim=-1)
                q_loss = self.value_loss_coef * 0.5 * Fs.shape[0] * nn.functional.mse_loss(Qs, target_Q)

            loss = fb_loss + orth_loss + q_loss

        self.forward_optimizer.zero_grad()
        self.backward_optimizer.zero_grad()
        loss.backward()

        if self.is_multi_gpu:
            self.reduce_parameters(self.forward_map)
            self.reduce_parameters(self.backward_map)

        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.forward_map.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.backward_map.parameters(), self.max_grad_norm)

        self.forward_optimizer.step()
        self.backward_optimizer.step()

        with torch.no_grad():
            loss_dict = {
                "Forward_Backward_Loss/total_loss": loss.mean().detach(),
                "Forward_Backward_Loss/fb_loss": fb_loss.mean().detach(),
                "Forward_Backward_Loss/ortho_loss": orth_loss.mean().detach(),
                "Forward_Backward_Loss/value_reg_loss": q_loss.mean().detach(),
            }
            extras = {
                "fb_target_successor_measure": target_M.mean().detach(),
                "fb_successor_measure": Ms[0].mean().detach(),
                "fb_backward": B.mean().detach(),
                "fb_forward_norm": Fs[0].norm(dim=-1).mean().detach(),
            }
        return loss_dict, extras

    def _update_actor(self, batch: ReplayBuffer.Batch) -> tuple[dict[str, torch.Tensor], dict]:
        with torch.autocast(device_type=self.device, dtype=self.dtype, enabled=self._autocast_enabled):
            actions = self.actor(
                batch.observations, batch.context, stochastic_output=True, std_clip=self.clip_actor_std
            )
            Fs = self.forward_map(batch.observations, batch.context, actions)
            Qs_fb = (Fs * batch.context).sum(dim=-1)
            Q_fb = compute_td_targets(Qs_fb, self.actor_pessimism)

            behavior_loss = torch.zeros((), device=self.device, dtype=torch.float32)
            if self.behavior_reg_coef > 0.0:
                # scaled by |Q| so the pull toward dataset actions keeps its weight as values grow
                behavior_loss = (
                    self.behavior_reg_coef * Q_fb.abs().mean().detach() * nn.functional.mse_loss(actions, batch.actions)
                )
            actor_loss = -Q_fb.mean() + behavior_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        if self.is_multi_gpu:
            self.reduce_parameters(self.actor)

        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)

        self.actor_optimizer.step()

        with torch.no_grad():
            loss_dict = {
                "Actor_Loss/total_loss": actor_loss.mean().detach(),
                "Actor_Loss/fb_loss": (-Q_fb.mean()).detach(),
                "Actor_Loss/behavior_loss": behavior_loss.detach(),
            }
            extras = {"actor_F1": Fs[0].mean().detach()}
        return loss_dict, extras

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        for model in self.models:
            model.train()
        self.obs_normalizer.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        for model in self.models:
            model.eval()
        self.obs_normalizer.eval()

    def get_policy(self) -> MLPModel:
        """Return the model used for inference and export."""
        return self.actor

    def save(self) -> dict:
        """Return the checkpoint payload for this algorithm."""
        return {
            "actor_state_dict": self.actor.state_dict(),
            "forward_map_state_dict": self.forward_map.state_dict(),
            "backward_map_state_dict": self.backward_map.state_dict(),
            "obs_normalizer_state_dict": self.obs_normalizer.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "forward_optimizer_state_dict": self.forward_optimizer.state_dict(),
            "backward_optimizer_state_dict": self.backward_optimizer.state_dict(),
        }

    @staticmethod
    def policy_state_keys() -> tuple[str, ...]:
        """Checkpoint keys needed to run the policy, i.e. what a slimmed checkpoint keeps."""
        return (
            "actor_state_dict",
            "forward_map_state_dict",
            "backward_map_state_dict",
            "obs_normalizer_state_dict",
        )

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Restore models, and optimizers when the checkpoint still carries them.

        Args:
            loaded_dict: The checkpoint payload.
            load_cfg: Unused; accepted for a uniform load signature.
            strict: Whether module state dicts must match exactly.

        Returns:
            True when optimizer state was restored, i.e. training can resume exactly.
        """
        del load_cfg
        self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        self.forward_map.load_state_dict(loaded_dict["forward_map_state_dict"], strict=strict)
        self.backward_map.load_state_dict(loaded_dict["backward_map_state_dict"], strict=strict)
        self.obs_normalizer.load_state_dict(loaded_dict["obs_normalizer_state_dict"], strict=strict)
        self.target_forward_map = TargetNetwork(self.forward_map)
        self.target_backward_map = TargetNetwork(self.backward_map)
        if "actor_optimizer_state_dict" not in loaded_dict:
            return False
        self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
        self.forward_optimizer.load_state_dict(loaded_dict["forward_optimizer_state_dict"])
        self.backward_optimizer.load_state_dict(loaded_dict["backward_optimizer_state_dict"])
        return True

    def broadcast_parameters(self) -> None:
        """Broadcast rank-0 model parameters to every rank."""
        model_params = [m.state_dict() for m in self.models]
        torch.distributed.broadcast_object_list(model_params, src=0)
        for model, params in zip(self.models, model_params, strict=False):
            model.load_state_dict(params)

    def reduce_parameters(self, m: nn.Module) -> None:
        """Average one module's gradients across ranks, in place."""
        grads = [p.grad.view(-1) for p in m.parameters() if p.grad is not None]
        if not grads:
            return
        flat = torch.cat(grads)
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat /= self.gpu_world_size
        offset = 0
        for p in m.parameters():
            if p.grad is None:
                continue
            numel = p.numel()
            p.grad.data.copy_(flat[offset : offset + numel].view_as(p.grad.data))
            offset += numel

    @classmethod
    def expert_bundle_groups(cls, obs_groups: dict[str, list[str]]) -> list[str]:
        """Obs groups a stored dataset must carry: those the backward map and the actor read.

        Args:
            obs_groups: The resolved role -> group-name mapping.

        Returns:
            Sorted, deduplicated group names.
        """
        return sorted({g for role in ("backward", "actor") for g in obs_groups[role]})

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: URLVecEnv, cfg: dict, device: str, inference: bool = False) -> Fb:
        """Build the models, dataset, and algorithm from a resolved config.

        Args:
            obs: A sample observation, for model input shapes.
            env: Supplies the action count and observation groups; never stepped during training.
            cfg: The resolved runner config; model sub-configs are consumed from it.
            device: Device for models and updates.
            inference: When True, skip the dataset so a policy can be built without it.

        Returns:
            The constructed algorithm.
        """
        cfg["algorithm"].pop("class_name")
        actor_class: type[FuseModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        forward_map_cfg = cfg["algorithm"].pop("forward_map")
        backward_map_cfg = cfg["algorithm"].pop("backward_map")
        forward_map_class: type[FuseModel] = resolve_callable(forward_map_cfg.pop("class_name"))  # type: ignore
        backward_map_class: type[MLPModel] = resolve_callable(backward_map_cfg.pop("class_name"))  # type: ignore

        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic", "backward"])

        actor_dist_cfg = cfg["actor"].get("distribution_cfg")
        if actor_dist_cfg is not None and actor_dist_cfg.get("class_name") == "TruncatedGaussianDistribution":
            clip_actions = cfg["clip_actions"]
            actor_dist_cfg["low"] = -clip_actions
            actor_dist_cfg["high"] = clip_actions

        z_dim = cfg["algorithm"]["z_dim"]
        actor: FuseModel = actor_class(obs, cfg["obs_groups"], "actor", (z_dim, 0), env.num_actions, **cfg["actor"]).to(
            device
        )
        forward_map: FuseModel = forward_map_class(
            obs, cfg["obs_groups"], "critic", (z_dim, env.num_actions), z_dim, **forward_map_cfg
        ).to(device)
        backward_map: MLPModel = backward_map_class(obs, cfg["obs_groups"], "backward", z_dim, **backward_map_cfg).to(
            device
        )

        all_obs_keys: list[str] = []
        for obs_set in cfg["obs_groups"].values():
            for key in obs_set:
                if key not in all_obs_keys and key in obs:
                    all_obs_keys.append(key)
        obs_normalizer: DictModule[nn.BatchNorm1d] = DictModule({
            key: nn.BatchNorm1d(obs[key].shape[-1], momentum=0.01, affine=False) for key in all_obs_keys
        }).to(device)

        dataset = None
        if not inference:
            dataset = OfflineTransitionDataset(
                cfg["algorithm"].pop("dataset_path"),
                z_dim,
                cfg["algorithm"]["batch_size"],
                storage_device=cfg["storage_device"],
                gamma=cfg["algorithm"].get("gamma", 0.99),
            )
            required = Fb.expert_bundle_groups(cfg["obs_groups"])
            missing = [g for g in required if g not in dataset.obs_groups]
            if missing:
                raise ValueError(
                    f"Offline dataset is missing obs group(s) {missing}, which the backward map and actor"
                    f" read. It has {dataset.obs_groups}; rebuild it against this env."
                )
        else:
            cfg["algorithm"].pop("dataset_path", None)

        return Fb(
            actor=actor,
            forward_map=forward_map,
            backward_map=backward_map,
            obs_normalizer=obs_normalizer,
            dataset=dataset,
            device=device,
            clip_actions=cfg.get("clip_actions"),
            multi_gpu_cfg=cfg.get("multi_gpu_cfg"),
            **cfg["algorithm"],
        )
