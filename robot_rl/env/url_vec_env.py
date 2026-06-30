from __future__ import annotations

import torch
from abc import abstractmethod
from collections.abc import Sequence
from tensordict import TensorDict

from robot_rl.storage import ExpertBuffer

from .vec_env import VecEnv


class URLVecEnv(VecEnv):
    """Abstract class for a vectorized environment for unsupervised RL.

    Extends the standard VecEnv class with extra utility methods.
    """

    """
    Operations.
    """

    @abstractmethod
    def reset(self) -> tuple[TensorDict, dict]:
        """Reset all environments.

        Returns:
            observations (TensorDict): Observations from the environment.
            extras (dict): Extra information from the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_to(
        self,
        state: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids: Sequence[int] | None = None,
        seed: int | None = None,
        is_relative: bool = False,
    ) -> tuple[TensorDict, dict]:
        """Reset specified environments to provided states.

        State is a dictionary with the following format:

        .. code-block:: python

            {
                "articulation": {
                    "entity_1_name": {
                        "root_pose": torch.Tensor,
                        "root_velocity": torch.Tensor,
                        "joint_position": torch.Tensor,
                        "joint_velocity": torch.Tensor,
                    },
                    "entity_2_name": {
                        "root_pose": torch.Tensor,
                        "root_velocity": torch.Tensor,
                        "joint_position": torch.Tensor,
                        "joint_velocity": torch.Tensor,
                    },
                },
                "deformable_object": {
                    "entity_3_name": {
                        "nodal_position": torch.Tensor,
                        "nodal_velocity": torch.Tensor,
                    }
                },
                "rigid_object": {
                    "entity_4_name": {
                        "root_pose": torch.Tensor,
                        "root_velocity": torch.Tensor,
                    }
                },
            }

        Args:
            state (dict): The state to reset the specified environments to. The first dimension of each tensor should be
                len(env_ids) or num_envs, if env_ids is None.
            env_ids (Sequence or None): The environment ids to reset. Defaults to None, in which case all environments
                are reset.
            seed (int or None): The seed to use for randomization. Defaults to None, in which case the seed is not set.
            is_relative (bool): if set to True, the state is considered relative to the environment origins. Defaults to
                False.

        Returns:
            observations (TensorDict): Observations from the environment.
            extras (dict): Extra information from the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def set_expert_buffer(self, buffer: ExpertBuffer) -> None:
        """Add an expert buffer to the environment, which can be used to sample states on reset.

        Args:
            buffer (ExpertBuffer): The buffer to use.
        """
        raise NotImplementedError

    @abstractmethod
    def train_mode(self) -> None:
        """Set environment to training mode, which can affect env events."""
        raise NotImplementedError

    @abstractmethod
    def eval_mode(self) -> None:
        """Set environment to eval mode, which can affect env events."""
        raise NotImplementedError
