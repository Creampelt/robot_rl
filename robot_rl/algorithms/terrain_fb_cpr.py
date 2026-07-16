"""Terrain-conditioned FB-CPR: encoder residual-dynamics loss + terrain-conditioned expert-z sampling ("coning").

Data contract: the env's ``terrain`` obs key carries ``[family, difficulty, ground_z, root_z,
root_lin_vel_h (3)]``; replay rows gain an in-place ``future_terrain`` key ``[dground(1), contacts(10), valid(1)]``.
"""

from __future__ import annotations

import torch
from collections import deque
from tensordict import TensorDict
from torch import nn
from typing import Any

from robot_rl.algorithms.fb_cpr import FbCpr
from robot_rl.storage import ReplayBuffer

# tile family -> clip families a coned draw may use. WIDER than the eval routing map on purpose
# (flat clips legally serve rough and slope tiles in TRAINING; eval keeps forbidding it).
TRAIN_TILE_CLIP_FAMILIES: dict[int, tuple[int, ...]] = {
    0: (0,),  # flat tile <- flat clips
    1: (0,),  # rough <- flat
    2: (0,),  # slope <- flat
    3: (3,),  # stairs <- stairs
    4: (4,),  # boxes <- boxes
    5: (5,),  # edge <- edge
}
NUM_FAMILIES = 6

_FUTURE_KEY = "future_terrain"
_FUTURE_DIM = 12  # dground(1) + contacts(10) + valid(1)


class _RunningStd(nn.Module):
    """Per-dim running std (momentum EMA) with a floor -- the residual RE-standardizer.

    The residual shrinks exactly as fast as the baseline improves; without re-standardizing, the
    encoder's gradient decays by the same factor, re-creating the dilution the residual exists to fix.
    The floor stops a genuinely-predictable dim from blowing the loss up.
    """

    def __init__(self, dim: int, momentum: float = 0.01, floor: float = 0.05) -> None:
        super().__init__()
        self.momentum = momentum
        self.floor = floor
        self.register_buffer("var", torch.ones(dim))

    def update(self, x: torch.Tensor) -> None:
        """Fold a batch of the TARGET into the running variance (never call on predictions)."""
        with torch.no_grad():
            self.var.lerp_(x.var(dim=0, unbiased=False), self.momentum)

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.var.sqrt().clamp(min=self.floor)


class _BilinearResidualHead(nn.Module):
    """``pred = W(s, a) @ c``: no additive (s, a)-only path to the output.

    With ``c = 0`` the prediction is EXACTLY the baseline, nothing else can absorb the residual, and
    ``dL/dc = W^T r`` is nonzero whenever the residual is -- the posterior-collapse dead basin is
    structurally removed while keeping the phase modulation we want (W depends on (s, a)).
    """

    def __init__(self, in_dim: int, c_dim: int, out_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.c_dim = c_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, out_dim * c_dim),
        )

    def forward(self, sa: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        w = self.trunk(sa).view(-1, self.out_dim, self.c_dim)
        return torch.bmm(w, c.unsqueeze(-1)).squeeze(-1)


class _NormedEncoder(nn.Module):
    """Encoder + output normalizer for INFERENCE: always running stats, never batch statistics."""

    def __init__(self, encoder: nn.Module, norm: nn.BatchNorm1d) -> None:
        super().__init__()
        self.encoder = encoder
        self.norm = norm

    def forward(self, obs: TensorDict) -> torch.Tensor:
        c = self.encoder(obs)
        return nn.functional.batch_norm(
            c, self.norm.running_mean, self.norm.running_var, training=False, eps=self.norm.eps
        )


class TerrainFbCpr(FbCpr):
    """FB-CPR + the terrain pieces: encoder dynamics objective, coning, family softening."""

    def __init__(
        self,
        *args: Any,
        dyn_horizon: int = 20,
        dyn_baseline_lr: float = 3e-4,
        dyn_residual_lr: float = 3e-4,
        dyn_weight_proprio: float = 0.3,
        dyn_weight_root: float = 1.0,
        dyn_weight_contact: float = 1.0,
        dyn_baseline_warmup: int = 1_000,
        cone_p_final: float = 0.6,
        cone_anneal_start: int = 10_000,
        cone_anneal_end: int = 30_000,
        family_alpha: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """See the module docstring; the extra kwargs cover the dynamics objective, coning, and family softening."""
        self.dyn_horizon = int(dyn_horizon)
        self._dyn_cfg = dict(
            baseline_lr=dyn_baseline_lr,
            residual_lr=dyn_residual_lr,
            w_proprio=dyn_weight_proprio,
            w_root=dyn_weight_root,
            w_contact=dyn_weight_contact,
            warmup=int(dyn_baseline_warmup),
        )
        self._cone_p_final = float(cone_p_final)
        self._cone_anneal = (int(cone_anneal_start), int(cone_anneal_end))
        self._family_alpha = float(family_alpha)

        encoder_cfg = kwargs.get("encoder_cfg")
        c_dim = int(encoder_cfg["output_dim"]) if encoder_cfg else 0

        super().__init__(*args, **kwargs)

        if self.encoder is None:
            raise ValueError("TerrainFbCpr requires an encoder (encoder_cfg).")

        # zero-init the consumers' c columns (identity-init gating; safe only with detached consumers).
        # Must run AFTER super().__init__ (init_weights re-randomizes) and re-sync the EMA targets.
        if c_dim:
            for model in (self.actor, self.forward_map, self.disc_critic, self.aux_critic):
                self._zero_c_pathway(model.embeddings[0], c_dim)
            for tgt in (self.target_forward_map, self.target_disc_critic, self.target_aux_critic):
                tgt.hard_sync()

        # encoder output normalizer: with VICReg gone nothing constrains ||c||; without this the
        # corruption knobs have drifting semantics and the rank probe is scale-sensitive
        self.encoder_norm = nn.BatchNorm1d(c_dim, momentum=0.01, affine=False).to(self.device)

        # --- the dynamics nets. f_base DELIBERATELY wider/deeper than the residual trunk: an under-capacity
        # baseline leaves (s,a)-structure in the residual and information_gain measures g, not c.
        obs_dim = int(self._norm_group_dim("obs"))
        act_dim = int(self.replay_buffer.actions.shape[-1])
        # heading-local root lin vel (terrain[:, 4:7]) joins the input AND the 1-step root block: the proprio
        # obs group has no root translation channel, so f(s, a) cannot otherwise know it moves INTO a step.
        in_dim = obs_dim + 3 + act_dim
        # targets: [ d(obs) 1-step (obs_dim) | d(root_height) 1-step (1) | d(root_vel_h) 1-step (3) |
        # dground n-step (1) ] as MSE blocks, + contacts@t+n (10) as BCE logits. MSE dim = obs_dim + 5.
        self._dyn_mse_dim = obs_dim + 5
        out_dim = self._dyn_mse_dim + 10
        self.dyn_baseline = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, out_dim),
        ).to(self.device)
        self.dyn_residual = _BilinearResidualHead(in_dim, c_dim, out_dim).to(self.device)
        self.dyn_target_std = _RunningStd(self._dyn_mse_dim).to(self.device)
        self.dyn_residual_std = _RunningStd(self._dyn_mse_dim).to(self.device)
        self.dyn_baseline_optimizer = torch.optim.Adam(self.dyn_baseline.parameters(), lr=dyn_baseline_lr)
        self.dyn_residual_optimizer = torch.optim.Adam(self.dyn_residual.parameters(), lr=dyn_residual_lr)
        # BCE pos_weight from running contact duty (clamped: a no-op for feet, keeps the rare
        # knee/hand/torso mount flags from being predicted always-0)
        self.register_buffer_compat("_contact_pos_rate", torch.full((10,), 0.3, device=self.device))

        # --- the staging ring: (rows, ground_z, episode_len_at_t) aged dyn_horizon steps
        self._ring: deque[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(maxlen=self.dyn_horizon)
        # future_terrain storage rides the replay buffer's obs TensorDict (every key is sampled)
        storage_dev = self.replay_buffer.observations.device
        self.replay_buffer.observations[_FUTURE_KEY] = torch.zeros(
            self.replay_buffer.capacity, _FUTURE_DIM, device=storage_dev
        )

        # --- coning state (skipped without an expert buffer: play/export construct with
        # inference=True and never draw expert-rollout z)
        self._tile_family: torch.Tensor | None = None
        self._compat: torch.Tensor | None = None
        self._cone_stats: dict[str, float] = {}
        if self.expert_buffer is not None:
            fam_key = self.expert_buffer.motions.get("family")
            if fam_key is None:
                raise ValueError("TerrainFbCpr needs a 'family' key in the motion bundle (coning + softening).")
            self._clip_family = fam_key[:, 0].long().to(self.device)
            compat = torch.zeros(NUM_FAMILIES, len(self._clip_family), dtype=torch.bool, device=self.device)
            for tile, clips in TRAIN_TILE_CLIP_FAMILIES.items():
                for cf in clips:
                    compat[tile] |= self._clip_family == cf
            self._compat = compat
            self.expert_buffer.set_family_softening(self._family_alpha)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _zero_c_pathway(branch: nn.Module, c_dim: int) -> None:
        """Silence the trailing ``c_dim`` input channels of a fused branch at init (c_init_scale = 0).

        Pre-norm LayerNorm still leaks c weakly through the input mean/std, but the gradient into the
        zeroed columns stays nonzero so the pathway can grow.
        """
        from robot_rl.modules.parallel import ParallelLinear

        for m in branch.modules():
            if isinstance(m, (nn.Linear, ParallelLinear)):
                with torch.no_grad():
                    if m.weight.dim() == 2:  # nn.Linear: (out, in)
                        m.weight[:, -c_dim:] = 0.0
                    else:  # ParallelLinear: (num_parallel, in, out)
                        m.weight[:, -c_dim:, :] = 0.0
                return
            w = getattr(m, "weight", None)
            if isinstance(w, nn.Parameter):  # a norm layer ahead of the projection: kill its c gains
                with torch.no_grad():
                    if w.dim() == 1:
                        w[-c_dim:] = 0.0
                    else:  # ParallelLayerNorm: (num_parallel, 1, in)
                        w[..., -c_dim:] = 0.0
        raise ValueError("embedding branch has no Linear/ParallelLinear layer")

    def register_buffer_compat(self, name: str, tensor: torch.Tensor) -> None:
        """nn.Module.register_buffer is unavailable (FbCpr is not a Module); keep a plain attribute."""
        setattr(self, name, tensor)

    def _norm_group_dim(self, group: str) -> int:
        """Flat dim of an obs group, from the replay buffer's storage."""
        return self.replay_buffer.observations[group].shape[-1]

    def _cone_p(self) -> float:
        start, end = self._cone_anneal
        if end <= start:
            return self._cone_p_final
        frac = (self._act_steps - start) / (end - start)
        return self._cone_p_final * min(max(frac, 0.0), 1.0)

    # ------------------------------------------------------------------ rollout side

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Stash each env's CURRENT tile family, then run the base rollout step."""
        # CURRENT tile family, read BEFORE super().act() refreshes the expert-rollout z (extras arrive in
        # process_env_step, too late); the SPAWN tile would mis-label envs that walked onto another tile.
        if "terrain" in obs:
            self._tile_family = obs["terrain"][:, 0].long().to(self.device)
        return super().act(obs)

    def _expert_rollout_rows(self, num_rows: int) -> torch.Tensor | None:
        """Cone the expert-rollout clip draw on the CURRENT tile family."""
        if self._tile_family is None or self._compat is None:
            return None
        assert self.expert_rollout_envs is not None
        fam = self._tile_family[self.expert_rollout_envs.to(self._tile_family.device)].to(self.device)
        weights = self.expert_buffer.sample_weights.to(self.device)
        uncond = torch.multinomial(weights.clamp(min=0), num_rows, replacement=True)

        p = self._cone_p()
        if p <= 0.0:
            self._cone_stats = {"cone_prob": 0.0, "match_frac": float("nan")}
            return uncond

        legal = self._compat[fam.clamp(min=0)]  # (E, M)
        coned_w = weights[None, :] * legal
        has_mass = coned_w.sum(-1) > 0
        use = (torch.rand(num_rows, device=self.device) < p) & (fam >= 0) & has_mass
        # rows for envs that cone; multinomial needs strictly positive row sums, so fill non-users
        safe_w = torch.where(use[:, None], coned_w, torch.ones_like(coned_w))
        coned = torch.multinomial(safe_w, 1).squeeze(-1)
        rows = torch.where(use, coned, uncond)

        match = self._compat[fam.clamp(min=0)].gather(1, rows[:, None]).squeeze(1) & (fam >= 0)
        self._cone_stats = {"cone_prob": p, "match_frac": float(match.float().mean())}
        return rows

    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        """Record the step, then age the staging ring and back-fill row t's future-terrain target."""
        super().process_env_step(obs, rewards, dones, extras)
        if "terrain" not in obs:
            return
        rows = self._last_added_rows
        ground_now = obs["terrain"][:, 2]
        contacts_now = obs["state"][:, -10:]
        ep_len = self._cur_episode_length.clone() if self._cur_episode_length is not None else None
        if ep_len is None or rows is None:
            return
        # a re-tenanted row must NOT inherit the previous occupant's payload: the buffer wraps ~5k
        # iters in, and stale [dground, contacts, valid=1] rows would silently corrupt L_dyn forever
        fresh = rows[rows >= 0]
        if fresh.numel():
            storage = self.replay_buffer.observations[_FUTURE_KEY]
            storage[fresh.to(storage.device)] = 0.0
        self._ring.append((rows, ground_now.clone(), ep_len))
        if len(self._ring) < self.dyn_horizon:
            return
        rows_t, ground_t, _ = self._ring[0]
        # valid = stored at t, AND no reset in (t, t+n]: the episode counter monotonically grew
        valid = (rows_t >= 0) & (ep_len.to(rows_t.device) >= self.dyn_horizon)
        if not bool(valid.any()):
            return
        idx = rows_t[valid]
        storage = self.replay_buffer.observations[_FUTURE_KEY]
        payload = torch.cat(
            [
                (ground_now - ground_t.to(ground_now.device))[valid.to(ground_now.device)].unsqueeze(-1),
                contacts_now[valid.to(contacts_now.device)],
                torch.ones(int(valid.sum()), 1, device=ground_now.device),
            ],
            dim=-1,
        )
        storage[idx.to(storage.device)] = payload.to(storage.device)

    def reset_rollout_state(self) -> None:
        """Clear the staging ring with the rest of the rollout bookkeeping (post-eval env reset)."""
        super().reset_rollout_state()
        self._ring.clear()

    # ------------------------------------------------------------------ encoder objective

    def _context_args(self, norm_obs: TensorDict, detach: bool = False, zero: bool = False) -> tuple:
        """Consumer-facing c: normalized, corrupted, and ALWAYS detached (full consumer detachment).

        L_dyn is the encoder's only teacher (consumer gradients cause rank collapse); the dynamics loss
        calls the encoder directly on clean obs, with gradients.
        """
        if self.encoder is None:
            return ()
        c = self.encoder_norm(self.encoder(norm_obs))
        c = torch.zeros_like(c) if (zero or self.zero_context) else self._corrupt_context(c)
        return (c.detach(),)

    def _auxiliary_losses(self, batch: ReplayBuffer.Batch) -> dict[str, torch.Tensor]:
        """Compute the residual dynamics loss, eager, riding the shared encoder step.

        Entirely eager and self-contained (a tensor crossing a CUDA-graph pool boundary
        is silently overwritten, which reads exactly like "c is ignored").
        """
        obs_n = batch.observations["obs"]
        next_obs_n = batch.next_observations["obs"]
        rh = batch.observations["state"][:, 0:1]
        next_rh = batch.next_observations["state"][:, 0:1]
        vel = batch.observations["terrain"][:, 4:7]  # heading-local root lin vel, never normalized
        next_vel = batch.next_observations["terrain"][:, 4:7]
        fut = batch.observations[_FUTURE_KEY]
        dground, contacts_fut, valid = fut[:, 0:1], fut[:, 1:11], fut[:, 11:12]

        sa = torch.cat([obs_n, vel, batch.actions], dim=-1)
        # MSE targets: 1-step deltas (normalized units; vel raw m/s -- the running std re-scales) + the
        # n-step terrain channel (meters)
        y_mse = torch.cat([next_obs_n - obs_n, next_rh - rh, next_vel - vel, dground], dim=-1)
        self.dyn_target_std.update(y_mse)
        y_mse = self.dyn_target_std.scale(y_mse)

        pred_base = self.dyn_baseline(sa)
        base_mse, base_logits = pred_base[:, : self._dyn_mse_dim], pred_base[:, self._dyn_mse_dim :]

        # per-block masks: the n-step dims only count where the future was back-filled
        d = self._dyn_mse_dim
        mse_mask = torch.ones_like(y_mse)
        mse_mask[:, d - 1 :] = valid  # dground column
        obs_dim = d - 5

        def blocks(err2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            proprio = err2[:, :obs_dim].mean()
            root_cols = err2[:, obs_dim:d] * mse_mask[:, obs_dim:d]
            root = root_cols.sum() / mse_mask[:, obs_dim:d].sum().clamp(min=1.0)
            return proprio, root

        w = self._dyn_cfg
        # ---- baseline loss (trains f_base only)
        e2 = (base_mse - y_mse.detach()) ** 2
        base_p, base_r = blocks(e2)
        pos_w = ((1 - self._contact_pos_rate) / self._contact_pos_rate.clamp(min=1e-3)).clamp(max=10.0)
        bce = nn.functional.binary_cross_entropy_with_logits
        base_c = bce(base_logits, contacts_fut, pos_weight=pos_w, reduction="none")
        base_c = (base_c * valid).sum() / (valid.sum() * 10).clamp(min=1.0)
        loss_base = w["w_proprio"] * base_p + w["w_root"] * base_r + w["w_contact"] * base_c

        self.dyn_baseline_optimizer.zero_grad()
        loss_base.backward()
        if self.is_multi_gpu:
            self.reduce_parameters(self.dyn_baseline)
        self.dyn_baseline_optimizer.step()

        # ---- residual loss (trains W-head + ENCODER; encoder step happens in base update())
        out = {
            "Encoder_Loss/baseline": loss_base.detach(),
            "Encoder_Loss/baseline_proprio": base_p.detach(),
            "Encoder_Loss/baseline_root": base_r.detach(),
            "Encoder_Loss/baseline_contact": base_c.detach(),
        }
        warmup_done = self._act_steps > self._dyn_cfg["warmup"]
        if warmup_done:
            c = self.encoder_norm(self.encoder(batch.observations))  # CLEAN c, with grads
            res_pred = self.dyn_residual(sa, c)
            res_mse, res_logits = res_pred[:, : self._dyn_mse_dim], res_pred[:, self._dyn_mse_dim :]

            r_target = (y_mse - base_mse).detach()  # the residual the baseline leaves behind
            # RE-standardize through the running std of the TARGET residual: it shrinks as f_base improves,
            # and without this the encoder's gradient decays by the same factor -- re-creating the dilution.
            self.dyn_residual_std.update(r_target)
            r_scaled_err = self.dyn_residual_std.scale(r_target) - self.dyn_residual_std.scale(res_mse)
            res_p, res_r = blocks(r_scaled_err**2)

            full_logits = base_logits.detach() + res_logits
            res_c = bce(full_logits, contacts_fut, pos_weight=pos_w, reduction="none")
            res_c = (res_c * valid).sum() / (valid.sum() * 10).clamp(min=1.0)
            loss_res = w["w_proprio"] * res_p + w["w_root"] * res_r + w["w_contact"] * res_c

            self.dyn_residual_optimizer.zero_grad()
            loss_res.backward()  # encoder grads accumulate; base update() steps the encoder
            if self.is_multi_gpu:
                self.reduce_parameters(self.dyn_residual)
            self.dyn_residual_optimizer.step()

            with torch.no_grad():
                # info gain (1 - ||r - g||^2/||r||^2) PER BLOCK under the valid mask: the flat mean is
                # ~32:1 proprio-diluted and hides a terrain-blind c. The TERRAIN gain is the gauge that matters.
                r_scaled = self.dyn_residual_std.scale(r_target)
                c_roll = torch.roll(c.detach(), 1, dims=0)  # null model: correspondence destroyed
                roll_pred = self.dyn_residual(sa, c_roll)[:, : self._dyn_mse_dim]
                roll_err2 = (r_scaled - self.dyn_residual_std.scale(roll_pred)) ** 2

                def gains(cols: slice) -> tuple[torch.Tensor, torch.Tensor]:
                    m = mse_mask[:, cols]
                    denom = (r_scaled[:, cols] ** 2 * m).sum().clamp(min=1e-8)
                    g = 1.0 - (r_scaled_err[:, cols] ** 2 * m).sum() / denom
                    g_roll = 1.0 - (roll_err2[:, cols] * m).sum() / denom
                    return g, g_roll

                gain, gain_roll = gains(slice(None))
                gain_p, _ = gains(slice(0, obs_dim))
                gain_r, _ = gains(slice(obs_dim, d - 1))
                gain_t, roll_t = gains(slice(d - 1, d))
                enc_grad = (
                    torch.stack([p.grad.norm() for p in self.encoder.parameters() if p.grad is not None]).norm()
                    if any(p.grad is not None for p in self.encoder.parameters())
                    else torch.zeros(())
                )
                out.update({
                    "Encoder_Loss/residual": loss_res.detach(),
                    "Metrics/encoder_information_gain": gain.detach(),
                    "Metrics/encoder_information_gain_proprio": gain_p.detach(),
                    "Metrics/encoder_information_gain_root": gain_r.detach(),
                    "Metrics/encoder_information_gain_terrain": gain_t.detach(),
                    "Metrics/encoder_shuffle_gain": gain_roll.detach(),
                    "Metrics/encoder_shuffle_gain_terrain": roll_t.detach(),
                    "Train/encoder_grad_norm": enc_grad.detach(),
                    "Train/c_std": c.detach().std(dim=0).mean(),
                })

        with torch.no_grad():
            if bool((valid > 0).any()):
                self._contact_pos_rate.lerp_(contacts_fut[valid.squeeze(-1) > 0].mean(dim=0), 0.01)
            out["Train/dyn_future_valid_frac"] = valid.mean().detach()
            out["Train/cone_prob"] = torch.tensor(self._cone_stats.get("cone_prob", 0.0))
            mf = self._cone_stats.get("match_frac", float("nan"))
            if mf == mf:
                out["Train/z_terrain_match_frac"] = torch.tensor(mf)
        return out

    def train_mode(self) -> None:
        """Set train mode for the base models plus the terrain modules."""
        super().train_mode()
        self.encoder_norm.train()
        self.dyn_baseline.train()
        self.dyn_residual.train()

    def eval_mode(self) -> None:
        """Set eval mode; encoder_norm MUST switch to running stats here.

        Base eval() calls this -- in train mode every eval batch would normalize with (meaningless,
        batch-1-degenerate) batch statistics AND pollute the running EMA.
        """
        super().eval_mode()
        self.encoder_norm.eval()
        self.dyn_baseline.eval()
        self.dyn_residual.eval()

    # ------------------------------------------------------------------ policy / persistence

    def get_policy(self) -> nn.Module:
        """Inference policy whose encoder INCLUDES the output normalizer, using RUNNING stats.

        Without this the deployed/played actor receives raw c although it only ever trained on
        BatchNorm-normalized c -- and at deployment batch sizes train-mode batch statistics are
        meaningless (batch-1 variance is zero).
        """
        from robot_rl.models.inference import EncoderInferencePolicy

        return EncoderInferencePolicy(
            _NormedEncoder(self.encoder, self.encoder_norm),
            self.actor,
            fuse_latent=True,
            zero_latent=self.zero_context,
        )

    @staticmethod
    def policy_state_keys() -> tuple[str, ...]:
        """Return the slim-checkpoint keys: base set + the encoder output normalizer's running stats.

        Checkpoint demotion (keep_full_checkpoints) strips everything else; without encoder_norm here
        a demoted checkpoint would deploy an actor whose c is scaled by garbage.
        """
        return (*FbCpr.policy_state_keys(), "encoder_norm_state_dict")

    def broadcast_parameters(self) -> None:
        """Broadcast the base models plus the terrain modules.

        Ranks otherwise start from different random inits for dyn_baseline/dyn_residual and
        silently diverge forever (their grads are all-reduced, but grads correct nothing that
        started different).
        """
        super().broadcast_parameters()
        extra = [self.dyn_baseline.state_dict(), self.dyn_residual.state_dict(), self.encoder_norm.state_dict()]
        torch.distributed.broadcast_object_list(extra, src=0)
        self.dyn_baseline.load_state_dict(extra[0])
        self.dyn_residual.load_state_dict(extra[1])
        self.encoder_norm.load_state_dict(extra[2])

    # ------------------------------------------------------------------ persistence

    def save(self) -> dict:
        """Extend the base checkpoint with the dynamics nets, standardizers and their optimizers."""
        d = super().save()
        d["encoder_norm_state_dict"] = self.encoder_norm.state_dict()
        d["dyn_baseline_state_dict"] = self.dyn_baseline.state_dict()
        d["dyn_residual_state_dict"] = self.dyn_residual.state_dict()
        d["dyn_target_std_state_dict"] = self.dyn_target_std.state_dict()
        d["dyn_residual_std_state_dict"] = self.dyn_residual_std.state_dict()
        d["dyn_baseline_optimizer_state_dict"] = self.dyn_baseline_optimizer.state_dict()
        d["dyn_residual_optimizer_state_dict"] = self.dyn_residual_optimizer.state_dict()
        return d

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Restore the base checkpoint plus the terrain-specific modules (absent keys are skipped)."""
        out = super().load(loaded_dict, load_cfg, strict)
        for name in (
            "encoder_norm",
            "dyn_baseline",
            "dyn_residual",
            "dyn_target_std",
            "dyn_residual_std",
            "dyn_baseline_optimizer",
            "dyn_residual_optimizer",
        ):
            key = f"{name}_state_dict"
            if key in loaded_dict:
                getattr(self, name).load_state_dict(loaded_dict[key])
        return out
