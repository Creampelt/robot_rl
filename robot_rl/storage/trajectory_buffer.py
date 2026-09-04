from __future__ import annotations

import torch
from collections.abc import Iterator
from typing import TYPE_CHECKING
from tensordict import TensorDict

from .expert_buffer import ExpertBuffer

if TYPE_CHECKING:
    from robot_rl.extensions.mirror import ObsMirror

# Rows drawn per window; only short rows consume more than one.
NUM_STITCH_SEGMENTS = 16


def _get_idxs(
    priorities: torch.Tensor,
    valid_lengths: torch.Tensor,
    num_slices: int,
    seq_length: int,
    bucket_size: int,
    num_segments: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate current/next ``(episode, frame)`` indices for consecutive windows, skipping padding.

    A window enters its first row at a random real frame and plays to that row's last real frame; if
    that does not fill the window it continues from the start of a further row, so hold-final-frame
    padding is never sampled. Rows long enough to cover the window on their own yield a single
    unbroken segment, which is the whole window for un-padded bundles.

    Returns:
        ``(ep, frame, next_ep, next_frame)``, each flattened to ``num_slices * seq_length``; the next
        indices are the following real frame, which a ``frame + 1`` lookup cannot express at a seam.
    """
    device = priorities.device
    window = seq_length + 1  # the caller also needs each frame's successor
    ep = torch.multinomial(priorities, num_slices * num_segments, replacement=True).view(num_slices, num_segments)
    lengths = valid_lengths[ep]

    # Only the entry point is random: later segments begin at their own clip's first frame.
    start_max = torch.clamp(lengths[:, :1] - seq_length, min=1, max=bucket_size - seq_length)
    first_start = (torch.rand(num_slices, 1, device=device) * start_max).long()
    starts = torch.cat([first_start, torch.zeros(num_slices, num_segments - 1, dtype=torch.long, device=device)], 1)
    seg_lengths = lengths - starts
    ends = torch.cumsum(seg_lengths, dim=1)

    pos = torch.arange(window, device=device).unsqueeze(0).expand(num_slices, window).contiguous()
    seg = torch.searchsorted(ends, pos, right=True)
    # Exhausting every segment is vanishingly rare; wrap inside the last one rather than pad.
    exhausted = seg >= num_segments
    seg = seg.clamp(max=num_segments - 1)
    consumed = torch.where(seg > 0, ends.gather(1, (seg - 1).clamp(min=0)), torch.zeros_like(seg))
    offset = pos - consumed
    offset = torch.where(exhausted, offset % seg_lengths.gather(1, seg), offset)
    frames = starts.gather(1, seg) + offset
    eps = ep.gather(1, seg)
    return (
        eps[:, :-1].reshape(-1),
        frames[:, :-1].reshape(-1),
        eps[:, 1:].reshape(-1),
        frames[:, 1:].reshape(-1),
    )


class TrajectoryBuffer(ExpertBuffer):
    """Subclass of :class:`ExpertBuffer` that stores expert observations in batches of fixed-length trajectories."""

    def __init__(
        self,
        motion_path: str,
        expert_obs_groups: list[str],
        device: str = "cpu",
        mirror: ObsMirror | None = None,
        mirror_prob: float = 0.0,
    ) -> None:
        """Initialize the buffer storage.

        Args:
            motion_path: Bundle of expert observation groups, batch shape (num_motions, bucket_size).
            expert_obs_groups: Groups concatenated into the expert state (root pose/velocity, joints).
            device: Storage device.
            mirror: Left-right mirror of the stored groups; each sampled window (and spawn state) is
                mirrored with probability ``mirror_prob``, whole, so a window's z is later derived from
                the mirrored states and never shared with the unmirrored window.
            mirror_prob: Probability of mirroring a sampled window or spawn state.
        """
        self.device = device
        self.obs_groups = expert_obs_groups
        self.mirror = mirror
        self._mirrors: dict[str, ObsMirror] = {}  # one copy per device the samples are consumed on
        self.mirror_prob = mirror_prob if mirror is not None else 0.0

        # motions file should be obs tensordict with batch shape (num_motions, bucket_size)
        self.motions = torch.load(motion_path, weights_only=False).to(device)
        if len(self.motions.shape) != 2:
            raise ValueError(
                "Expected motions batch size to be 2-dimensional (num_motions, bucket_size), but instead got shape "
                f"{self.motions.shape}."
            )
        self.num_motions, self.bucket_size = self.motions.shape
        # per-row un-padded frame count ("length" key from play_dataset --pad_to_segment); bundles
        # without it (e.g. LAFAN, fully-real buckets) fall back to bucket_size
        if "length" in self.motions:
            self.valid_lengths = self.motions["length"][:, 0].long().clamp(1, self.bucket_size)
        else:
            self.valid_lengths = torch.full((self.num_motions,), self.bucket_size, dtype=torch.long, device=device)
        self.priorities = torch.ones((self.num_motions,), device=device)
        self._eval_order = torch.arange(0, self.num_motions, device=self.device)
        # per-row valid lengths of the most recent get_batch_motions mini-batch
        self.current_eval_motion_lengths: torch.Tensor | None = None

        # mode="default" (no CUDA graphs): this samples under the inference_mode rollout, so a
        # reduce-overhead capture would poison the shared cudagraph pool that update() later reuses.
        self._get_idxs = torch.compile(_get_idxs, mode="default")

        print(f"[INFO] Successfully loaded {self.num_motions} motions with length {self.bucket_size}.")

    def sample(
        self,
        batch_size: int,
        device: str | None = None,
        seq_length: int = 1,
    ) -> tuple[TensorDict, TensorDict]:
        """Sample current and next expert observations from multinomial distribution weighted by priorities.

        When ``seq_length > 1``, samples are returned as ``batch_size // seq_length`` consecutive windows, each of
        length ``seq_length``, drawn from a single motion starting at a random frame. The flat output is ordered so
        that reshaping ``(batch_size, ...) -> (num_slices, seq_length, ...)`` recovers the windows row-wise. With
        ``seq_length = 1`` behavior is equivalent to iid transition sampling.

        Args:
            batch_size: The batch size to sample. Must be divisible by ``seq_length``.
            device: The device to move the output to. Defaults to None, which keeps the observations on the buffer's
                device.
            seq_length: Length of each consecutive window. Must be strictly less than ``bucket_size``.

        Returns:
            A tuple containing the expert obs and next obs as TensorDicts. Shape is (batch_size).
        """
        if batch_size % seq_length != 0:
            raise ValueError(f"batch_size ({batch_size}) must be divisible by seq_length ({seq_length}).")
        if seq_length >= self.bucket_size:
            raise ValueError(f"seq_length ({seq_length}) must be less than bucket_size ({self.bucket_size}).")
        num_slices = batch_size // seq_length
        ep_flat, seq_indices, next_ep_flat, next_seq_indices = self._get_idxs(
            self.priorities, self.valid_lengths, num_slices, seq_length, self.bucket_size, NUM_STITCH_SEGMENTS
        )
        # mirror AFTER the device move: a host-resident buffer would otherwise pay the flip on CPU tensors
        obs = self.motions[ep_flat, seq_indices].to(device)
        next_obs = self.motions[next_ep_flat, next_seq_indices].to(device)
        if self.mirror_prob > 0:
            mirror = self._mirror_on(obs.device)
            flip = (torch.rand(num_slices, device=obs.device) < self.mirror_prob).repeat_interleave(seq_length)
            obs, next_obs = mirror.apply(obs, flip), mirror.apply(next_obs, flip)
        return obs, next_obs

    def sample_states(self, num_envs: int, device: str | None = None) -> dict[str, torch.Tensor]:
        """Sample states for a vectorized environment. Returns a state dictionary.

        See :meth:`get_expert_state` for full state dictionary format.
        """
        ep_indices = torch.multinomial(self.priorities, num_envs, replacement=True)
        # spawn only from real frames: hold-final-frame padding is a static pose, not a start state
        motion_indices = (torch.rand(num_envs, device=self.device) * self.valid_lengths[ep_indices]).long()
        motions = self.motions[ep_indices, motion_indices].to(device)
        if self.mirror_prob > 0:
            flip = torch.rand(num_envs, device=motions.device) < self.mirror_prob
            motions = self._mirror_on(motions.device).apply(motions, flip)
        return self.get_expert_state(motions, device=device)

    def _mirror_on(self, device: torch.device | str) -> ObsMirror:
        key = str(device)
        if key not in self._mirrors:
            from robot_rl.extensions.mirror import ObsMirror

            m = self.mirror
            copy = ObsMirror(dict(m.group_perm), dict(m.group_sign), m.action_perm, m.action_sign)
            self._mirrors[key] = copy.to(device)
        return self._mirrors[key]

    def get_batch_motions(
        self,
        mini_batch_size: int,
        device: str | None = None,
    ) -> Iterator[TensorDict]:
        """Sample entire motion trajectories in mini batches.

        Returns iterator containing batched observations as TensorDict with shape (mini_batch_size, bucket_size,
        *obs_size). Note that the final batch may be truncated.
        """
        # Randomize order since rigid body DR is fixed per-environment
        self._eval_order = torch.randperm(self.num_motions, device=self.device)
        for idx in range(0, self.num_motions, mini_batch_size):
            eval_idxs = self._eval_order[idx : idx + mini_batch_size]
            # rows are yielded whole, so the caller needs the valid lengths to ignore padded frames
            self.current_eval_motion_lengths = self.valid_lengths[eval_idxs]
            yield self.motions[eval_idxs].to(device)

    def update_priorities(self, priorities: torch.Tensor, indices: torch.Tensor | slice) -> None:
        """Update a slice of priorities. Assumes :meth:`get_batch_motions` has already been called."""
        actual_indices = self._eval_order[indices]
        self.priorities[actual_indices] = priorities.to(self.device)

    def normalize_priorities(self) -> None:
        """Normalize all priorities by dividing by their sum."""
        self.priorities /= self.priorities.sum()

    def state_dict(self) -> dict:
        """Return the per-motion sampling priorities for checkpointing."""
        return {"priorities": self.priorities}

    def load_state_dict(self, state: dict) -> None:
        """Restore the per-motion sampling priorities from a checkpoint."""
        priorities = state["priorities"].to(self.device)
        if priorities.shape != self.priorities.shape:
            raise ValueError(
                f"Loaded expert priorities shape {tuple(priorities.shape)} does not match the current buffer "
                f"{tuple(self.priorities.shape)}; the motion dataset likely differs from the checkpointed run."
            )
        self.priorities = priorities

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
