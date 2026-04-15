import torch
from collections.abc import Iterator
from tensordict import TensorDict

from .expert_buffer import ExpertBuffer


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

        # motions file should be obs tensordict with batch shape (num_motions, bucket_size)
        self.motions = torch.load(motion_path, weights_only=False).to(device)
        if len(self.motions.shape) != 2:
            raise ValueError(
                "Expected motions batch size to be 2-dimensional (num_motions, bucket_size), but instead got shape "
                f"{self.motions.shape}."
            )
        self.num_motions, self.bucket_size = self.motions.shape
        self.priorities = torch.ones((self.num_motions,), device=device)
        self._eval_order = torch.arange(0, self.num_motions, device=self.device)

        print(f"[INFO] Successfully loaded {self.num_motions} motions with length {self.bucket_size}.")

    def sample(self, batch_size: int, device: str | None = None) -> tuple[TensorDict, TensorDict]:
        """Sample current and next expert observations from multinomial distribution weighted by priorities.

        Args:
            batch_size: The batch size to sample.
            device: The device to move the output to. Defaults to None, which keeps the observations on the buffer's
                device.

        Returns:
            A tuple containing the expert obs and next obs as TensorDicts. Shape is (batch_size).
        """
        # sample episodes according to priorities
        ep_indices = torch.multinomial(self.priorities, batch_size, replacement=True)
        # uniformly sample from sequence (exclude last so there will always be a next obs)
        seq_indices = torch.randint(0, self.bucket_size - 1, (batch_size,), device=self.device)
        return (
            self.motions[ep_indices, seq_indices].to(device),
            self.motions[ep_indices, seq_indices + 1].to(device),
        )

    def sample_states(self, num_envs: int, device: str | None = None) -> dict[str, torch.Tensor]:
        """Sample states for a vectorized environment. Returns a state dictionary.

        See :meth:`get_expert_state` for full state dictionary format.
        """
        ep_indices = torch.multinomial(self.priorities, num_envs, replacement=True)
        motion_indices = torch.randint(0, self.bucket_size, (num_envs,), device=self.device)
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
        # Randomize order since rigid body DR is fixed per-environment
        self._eval_order = torch.randperm(self.num_motions, device=self.device)
        for idx in range(0, self.num_motions, mini_batch_size):
            eval_idxs = self._eval_order[idx : idx + mini_batch_size]
            yield self.motions[eval_idxs].to(device)

    def update_priorities(self, priorities: torch.Tensor, indices: torch.Tensor | slice) -> None:
        """Update a slice of priorities. Assumes :meth:`get_batch_motions` has already been called."""
        actual_indices = self._eval_order[indices]
        self.priorities[actual_indices] = priorities.to(self.device)

    def normalize_priorities(self) -> None:
        """Normalize all priorities by dividing by their sum."""
        self.priorities /= self.priorities.sum()

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
