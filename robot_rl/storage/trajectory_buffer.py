import torch
from collections.abc import Iterator
from tensordict import TensorDict

from .expert_buffer import ExpertBuffer


def _window_idxs(
    ep_indices: torch.Tensor,
    valid_lengths: torch.Tensor,
    seq_length: int,
    bucket_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw a start frame per given row and expand to ``(episode, frame)`` window indices.

    Windows always START inside a row's valid (un-padded) frames: rows longer than the window yield
    fully-real windows; shorter rows start at 0 and run into the hold-final-frame padding (a
    track-then-settle target). Upper bound keeps ``+1`` headroom for the caller's next-frame lookup.
    """
    num_slices = ep_indices.shape[0]
    start_max = torch.clamp(valid_lengths[ep_indices] - seq_length, min=1, max=bucket_size - seq_length)
    starts = (torch.rand(num_slices, device=ep_indices.device) * start_max).long()
    offsets = torch.arange(seq_length, device=ep_indices.device)
    seq_indices = (starts.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1)
    ep_flat = ep_indices.unsqueeze(1).expand(num_slices, seq_length).reshape(-1)
    return ep_flat, seq_indices


def _get_idxs(
    weights: torch.Tensor,
    valid_lengths: torch.Tensor,
    num_slices: int,
    seq_length: int,
    bucket_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample rows from ``weights``, then a consecutive window inside each."""
    ep_indices = torch.multinomial(weights, num_slices, replacement=True)
    return _window_idxs(ep_indices, valid_lengths, seq_length, bucket_size)


def _get_idxs_for_eps(
    ep_indices: torch.Tensor,
    valid_lengths: torch.Tensor,
    seq_length: int,
    bucket_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Window indices for CALLER-CHOSEN rows (terrain-coned expert-rollout draws): no multinomial.

    A separate compiled entry point rather than an optional tensor arg on ``_get_idxs`` -- the latter
    would change that function's guards, and it runs on the hot rollout path.
    """
    return _window_idxs(ep_indices, valid_lengths, seq_length, bucket_size)


class TrajectoryBuffer(ExpertBuffer):
    """Subclass of :class:`ExpertBuffer` that stores expert observations in batches of fixed-length trajectories."""

    def __init__(
        self,
        motion_path: str,
        expert_obs_groups: list[str],
        device: str = "cpu",
    ) -> None:
        """Initialize the buffer storage."""
        self.device = device
        self.obs_groups = expert_obs_groups

        # motions: obs tensordict with batch shape (num_motions, bucket_size). map_location="cpu": GPU-converted
        # bundles carry cuda:0 tensors and torch.load would materialize the multi-GB file on GPU0.
        self.motions = torch.load(motion_path, weights_only=False, map_location="cpu").to(device)
        if len(self.motions.shape) != 2:
            raise ValueError(
                "Expected motions batch size to be 2-dimensional (num_motions, bucket_size), but instead got shape "
                f"{self.motions.shape}."
            )
        self.num_motions, self.bucket_size = self.motions.shape
        # per-row un-padded frame count ("length" key, written by play_dataset --convert); bundles
        # without it (e.g. LAFAN, fully-real buckets) fall back to bucket_size = legacy behavior
        if "length" in self.motions:
            self.valid_lengths = self.motions["length"][:, 0].long().clamp(1, self.bucket_size)
        else:
            self.valid_lengths = torch.full((self.num_motions,), self.bucket_size, dtype=torch.long, device=device)
        self.priorities = torch.ones((self.num_motions,), device=device)
        # Family softening and the held-out split live in their OWN tensors, never in `priorities`: eval()
        # overwrites priorities wholesale, erasing anything baked in; samplers compose via `sample_weights`.
        self.base_weights = torch.ones((self.num_motions,), device=device)
        self.train_mask = torch.ones((self.num_motions,), device=device)
        if "heldout" in self.motions:
            # pinned in the bundle at conversion, BY CLIP, so a clip's buckets never straddle the split
            self.train_mask = (self.motions["heldout"][:, 0].reshape(-1) == 0).float().to(device)
            n_held = int((self.train_mask == 0).sum())
            print(f"[INFO] Held-out clips: {n_held}/{self.num_motions} (excluded from training).")
        self._eval_order = torch.arange(0, self.num_motions, device=self.device)
        # optional: restrict eval (get_batch_motions) to these motion indices; None = all
        self.eval_motion_indices: torch.Tensor | None = None
        # motion rows of the most recent get_batch_motions mini-batch, env-ordered (env i replays row
        # [i]); envs read it in reset_to for eval-side routing (e.g. family-matched terrain tiles)
        self.current_eval_motion_indices: torch.Tensor | None = None
        self.current_eval_motion_lengths: torch.Tensor | None = None
        # (motion rows, frame cols) of the most recent sample_states batch, env-ordered; envs read it
        # in reset events for per-frame spawn fitting (e.g. the RSI foot-raycast z-fit)
        self.current_sample_indices: tuple[torch.Tensor, torch.Tensor] | None = None

        # mode="default" (no CUDA graphs): this samples under the inference_mode rollout, so a
        # reduce-overhead capture would poison the shared cudagraph pool that update() later reuses.
        self._get_idxs = torch.compile(_get_idxs, mode="default")
        self._get_idxs_for_eps = torch.compile(_get_idxs_for_eps, mode="default")

        print(f"[INFO] Successfully loaded {self.num_motions} motions with length {self.bucket_size}.")

    @property
    def sample_weights(self) -> torch.Tensor:
        """The distribution BOTH samplers draw from: EMD priority x family softening x held-out mask."""
        return self.priorities * self.base_weights * self.train_mask

    @property
    def heldout_indices(self) -> torch.Tensor:
        """Rows excluded from training -- the only instrument that can see clip overfitting."""
        return torch.nonzero(self.train_mask == 0, as_tuple=False).reshape(-1)

    def set_family_softening(self, alpha: float, weight_clip: float = 32.0) -> None:
        """Weight each clip by ``n_f^(alpha-1)``, ``n_f`` = the clip count of its family.

        ``alpha = 1`` is the natural corpus (box-dominated: 554 boxes vs 26 stairs). ``alpha = 0`` gives
        every family equal mass, hammering the rare clips ~21x -- and it COMPOUNDS with the EMD priority,
        which already favours exactly those hard stairs clips. Normalized to a median of 1 and clipped so
        the product cannot run away.

        Args:
            alpha: Softening exponent; 0.5 (sqrt class-balance) is the v3 starting point.
            weight_clip: Max weight relative to the median.
        """
        if "family" not in self.motions:
            raise KeyError("family softening needs a 'family' key in the motion bundle")
        fam = self.motions["family"][:, 0].reshape(-1).long()
        counts = torch.bincount(fam, minlength=int(fam.max().item()) + 1).float()
        weights = counts[fam].clamp(min=1.0).pow(alpha - 1.0)
        weights = weights / weights.median()
        self.base_weights = weights.clamp(max=weight_clip).to(self.device)

    def sample(
        self,
        batch_size: int,
        device: str | None = None,
        seq_length: int = 1,
        ep_indices: torch.Tensor | None = None,
    ) -> tuple[TensorDict, TensorDict]:
        """Sample current and next expert observations, weighted by :attr:`sample_weights`.

        When ``seq_length > 1``, samples are returned as ``batch_size // seq_length`` consecutive windows, each of
        length ``seq_length``, drawn from a single motion starting at a random frame. The flat output is ordered so
        that reshaping ``(batch_size, ...) -> (num_slices, seq_length, ...)`` recovers the windows row-wise. With
        ``seq_length = 1`` behavior is equivalent to iid transition sampling.

        Args:
            batch_size: The batch size to sample. Must be divisible by ``seq_length``.
            device: The device to move the output to. Defaults to None, which keeps the observations on the buffer's
                device.
            seq_length: Length of each consecutive window. Must be strictly less than ``bucket_size``.
            ep_indices: Caller-chosen motion rows (one per window), bypassing the weighted draw -- used by the
                terrain-coned expert-rollout z. Start frames are still valid-length aware.

        Returns:
            A tuple containing the expert obs and next obs as TensorDicts. Shape is (batch_size).
        """
        if batch_size % seq_length != 0:
            raise ValueError(f"batch_size ({batch_size}) must be divisible by seq_length ({seq_length}).")
        if seq_length >= self.bucket_size:
            raise ValueError(f"seq_length ({seq_length}) must be less than bucket_size ({self.bucket_size}).")
        num_slices = batch_size // seq_length
        if ep_indices is None:
            ep_flat, seq_indices = self._get_idxs(
                self.sample_weights, self.valid_lengths, num_slices, seq_length, self.bucket_size
            )
        else:
            if ep_indices.shape[0] != num_slices:
                raise ValueError(f"ep_indices has {ep_indices.shape[0]} rows, expected {num_slices}.")
            ep_flat, seq_indices = self._get_idxs_for_eps(
                ep_indices.to(self.device), self.valid_lengths, seq_length, self.bucket_size
            )
        return (
            self.motions[ep_flat, seq_indices].to(device),
            self.motions[ep_flat, seq_indices + 1].to(device),
        )

    def sample_states(self, num_envs: int, device: str | None = None) -> dict[str, torch.Tensor]:
        """Sample states for a vectorized environment. Returns a state dictionary.

        See :meth:`get_expert_state` for full state dictionary format.
        """
        ep_indices = torch.multinomial(self.sample_weights, num_envs, replacement=True)
        # spawn frames only from real (un-padded) motion
        motion_indices = (torch.rand(num_envs, device=self.device) * self.valid_lengths[ep_indices]).long()
        self.current_sample_indices = (ep_indices, motion_indices)
        motions = self.motions[ep_indices, motion_indices]
        return self.get_expert_state(motions, device=device)

    def get_batch_motions(
        self,
        mini_batch_size: int,
        device: str | None = None,
    ) -> Iterator[TensorDict]:
        """Sample entire motion trajectories in mini batches.

        Returns iterator containing batched observations as TensorDict with shape (mini_batch_size, bucket_size,
        *obs_size). Note that the final batch may be truncated.
        """
        if self.eval_motion_indices is not None:
            # pinned eval sets (e.g. video views) keep caller order: consecutive rows = clip segments in order
            self._eval_order = self.eval_motion_indices
        else:
            # randomize order since rigid body DR is fixed per-environment
            order = torch.arange(self.num_motions, device=self.device)
            self._eval_order = order[torch.randperm(len(order), device=self.device)]
        for idx in range(0, len(self._eval_order), mini_batch_size):
            eval_idxs = self._eval_order[idx : idx + mini_batch_size]
            self.current_eval_motion_indices = eval_idxs
            # per-row un-padded lengths of this mini-batch (eval-side EMD masking)
            self.current_eval_motion_lengths = self.valid_lengths[eval_idxs]
            yield self.motions[eval_idxs].to(device)

    def update_priorities(self, priorities: torch.Tensor, indices: torch.Tensor | slice) -> None:
        """Update a slice of priorities. Assumes :meth:`get_batch_motions` has already been called."""
        actual_indices = self._eval_order[indices]
        self.priorities[actual_indices] = priorities.to(self.device)

    def normalize_priorities(self) -> None:
        """Normalize all priorities by dividing by their sum, then re-zero the held-out rows.

        The re-zero is belt-and-braces -- ``sample_weights`` already masks them -- but ``eval()`` writes a
        LARGE priority to every row it scores, so a held-out row that ever leaks into a sampler would leak
        with a big weight. Keeping it zero here means the leak is bounded even if a future caller reads
        ``priorities`` directly.
        """
        self.priorities /= self.priorities.sum()
        self.priorities *= self.train_mask

    def state_dict(self) -> dict:
        """Return the per-motion sampling state for checkpointing."""
        return {
            "priorities": self.priorities,
            "base_weights": self.base_weights,
            "train_mask": self.train_mask,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore the per-motion sampling state from a checkpoint.

        ``train_mask`` MUST round-trip: a resume that redraws the split would train on rows it then
        evaluates as held out, and the overfitting gap -- the only instrument that can see clip
        memorization -- would silently read ~0.
        """
        priorities = state["priorities"].to(self.device)
        if priorities.shape != self.priorities.shape:
            raise ValueError(
                f"Loaded expert priorities shape {tuple(priorities.shape)} does not match the current buffer "
                f"{tuple(self.priorities.shape)}; the motion dataset likely differs from the checkpointed run."
            )
        self.priorities = priorities
        # older checkpoints predate these; fall back to the values derived from the bundle at construction
        for key in ("base_weights", "train_mask"):
            if key in state:
                setattr(self, key, state[key].to(self.device))

    def get_expert_state(self, obs: TensorDict, device: str | None = None) -> dict[str, torch.Tensor]:
        """Convert the observations TensorDict into a state dictionary of tensors.

        Dictionary structure and tensor shapes are:

        .. code:: python

            {
                "root_pose": (..., 7),
                "root_velocity": (..., 6),
                "joint_position": (..., num_joints),
                "joint_velocity": (..., num_joints),
            }
        """
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        state = torch.cat(obs_list, dim=-1).to(device)
        num_joints = (state.shape[-1] - 13) // 2
        return {
            "root_pose": state[..., :7],
            "root_velocity": state[..., 7:13],
            "joint_position": state[..., 13 : 13 + num_joints],
            "joint_velocity": state[..., 13 + num_joints :],
        }
