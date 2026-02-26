import copy
from collections.abc import Sequence

import torch
import torch.nn as nn
from tensordict import TensorDict

from robot_rl.networks import (
    MLP,
    EmbeddedNet,
    EmbeddedResNet,
    ScaledNormalization,
    TruncatedNormal,
)
from robot_rl.utils import eval_mode, get_obs, get_obs_dimensions


class ForwardBackward(nn.Module):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        z_dim: int = 100,
        fb_tau: float = 0.01,
        critic_tau: float = 0.005,
        init_noise_std: float = 0.2,
        actor_obs_normalization: bool = True,
        critic_obs_normalization: bool = True,
        z_normalization: bool = True,
        backward_out_normalization: bool = True,
        actor_num_parallel: int = 1,
        actor_embedding_dims: Sequence[int] = [2048, 2048, 2048, 2048],
        actor_hidden_dims: Sequence[int] = [2048, 2048, 2048, 2048, 2048, 2048],
        critic_num_parallel: int = 2,
        critic_embedding_dims: Sequence[int] = [2048, 2048, 2048, 2048],
        critic_hidden_dims: Sequence[int] = [2048, 2048, 2048, 2048, 2048, 2048],
        backward_hidden_dims: Sequence[int] = [256],
        discriminator_hidden_dims: Sequence[int] = [1024, 1024],
        **kwargs,
    ):
        if kwargs:
            print(
                f"{self.__class__.__name__}.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.z_dim = z_dim
        self.fb_tau = fb_tau
        self.critic_tau = critic_tau
        self.init_noise_std = init_noise_std

        # get the observation dimensions
        self.obs_groups = obs_groups
        self.num_actor_obs = get_obs_dimensions(obs, obs_groups["policy"])
        self.num_critic_obs = get_obs_dimensions(obs, obs_groups["critic"])
        self.num_backward_obs = get_obs_dimensions(obs, obs_groups["backward"])
        self.num_discriminator_obs = get_obs_dimensions(obs, obs_groups["discriminator"])

        # actor
        self.actor = EmbeddedResNet(
            (self.num_actor_obs + z_dim, self.num_actor_obs),
            num_actions,
            actor_embedding_dims,
            actor_hidden_dims,
            actor_num_parallel,
            last_activation="tanh",
        )
        self.actor_obs_normalizer = (
            nn.BatchNorm1d(self.num_actor_obs, affine=False, momentum=0.01)
            if actor_obs_normalization
            else nn.Identity()
        )

        # backward mapping
        self.backward_map = MLP(
            self.num_backward_obs,
            z_dim,
            backward_hidden_dims,
            activation="relu",
            last_activation=ScaledNormalization() if backward_out_normalization else "identity",
        )
        self.backward_obs_normalizer = (
            nn.BatchNorm1d(self.num_backward_obs, affine=False, momentum=0.01)
            if critic_obs_normalization
            else nn.Identity()
        )

        # discriminator
        self.discriminator = MLP(
            self.num_discriminator_obs + self.z_dim,
            1,
            discriminator_hidden_dims,
            activation="relu",
            last_activation="sigmoid",
        )
        self.discriminator_obs_normalizer = (
            nn.BatchNorm1d(self.num_discriminator_obs, affine=False, momentum=0.01)
            if critic_obs_normalization
            else nn.Identity()
        )

        # critics (forward, discriminator, auxiliary)
        self.forward_map = EmbeddedResNet(
            (self.num_critic_obs + self.z_dim, self.num_critic_obs + num_actions),
            self.z_dim,
            critic_embedding_dims,
            critic_hidden_dims,
            critic_num_parallel,
        )
        self.disc_critic = EmbeddedResNet(
            (self.num_critic_obs + self.z_dim, self.num_critic_obs + num_actions),
            1,
            critic_embedding_dims,
            critic_hidden_dims,
            critic_num_parallel,
        )
        self.aux_critic = EmbeddedResNet(
            (self.num_critic_obs + self.z_dim, self.num_critic_obs + num_actions),
            1,
            critic_embedding_dims,
            critic_hidden_dims,
            critic_num_parallel,
        )

        self.critic_obs_normalizer = (
            nn.BatchNorm1d(self.num_critic_obs, affine=False, momentum=0.01)
            if critic_obs_normalization
            else nn.Identity()
        )

        # create z normalizer module
        self.z_normalizer = ScaledNormalization() if z_normalization else nn.Identity()
        self.distribution: TruncatedNormal | None = None

        # placeholders for target networks and paramlists
        self.target_forward_map: EmbeddedNet | None = None
        self.target_backward_map: MLP | None = None
        self.target_disc_critic: EmbeddedNet | None = None
        self.target_aux_critic: EmbeddedNet | None = None

        self._forward_paramlist: tuple[torch.Tensor, ...] | None = None
        self._backward_paramlist: tuple[torch.Tensor, ...] | None = None
        self._disc_critic_paramlist: tuple[torch.Tensor, ...] | None = None
        self._aux_critic_paramlist: tuple[torch.Tensor, ...] | None = None
        self._target_forward_paramlist: tuple[torch.Tensor, ...] | None = None
        self._target_backward_paramlist: tuple[torch.Tensor, ...] | None = None
        self._target_disc_critic_paramlist: tuple[torch.Tensor, ...] | None = None
        self._target_aux_critic_paramlist: tuple[torch.Tensor, ...] | None = None

    def init_targets(self, device: str | None = None):
        self.target_forward_map = copy.deepcopy(self.forward_map).to(device)
        self.target_backward_map = copy.deepcopy(self.backward_map).to(device)
        self.target_disc_critic = copy.deepcopy(self.disc_critic).to(device)
        self.target_aux_critic = copy.deepcopy(self.aux_critic).to(device)
        # create paramlists
        self._forward_paramlist = tuple(x.data for x in self.forward_map.parameters())
        self._backward_paramlist = tuple(x.data for x in self.backward_map.parameters())
        self._disc_critic_paramlist = tuple(x.data for x in self.disc_critic.parameters())
        self._aux_critic_paramlist = tuple(x.data for x in self.aux_critic.parameters())
        self._target_forward_paramlist = tuple(x.data for x in self.target_forward_map.parameters())
        self._target_backward_paramlist = tuple(x.data for x in self.target_backward_map.parameters())
        self._target_disc_critic_paramlist = tuple(x.data for x in self.target_disc_critic.parameters())
        self._target_aux_critic_paramlist = tuple(x.data for x in self.target_aux_critic.parameters())
        return self

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    def update_distribution(self, obs_z: torch.Tensor, obs: torch.Tensor) -> None:
        mean = self.actor(obs_z, obs)
        self.distribution = TruncatedNormal(mean, self.init_noise_std)

    def update_normalization(self, obs: TensorDict) -> None:
        # extract obs tensors from tensordicts
        actor_obs = get_obs(obs, self.obs_groups["policy"])
        critic_obs = get_obs(obs, self.obs_groups["critic"])
        backward_obs = get_obs(obs, self.obs_groups["backward"])
        discriminator_obs = get_obs(obs, self.obs_groups["discriminator"])

        # update normalization states
        self.actor_obs_normalizer(actor_obs)
        self.critic_obs_normalizer(critic_obs)
        self.backward_obs_normalizer(backward_obs)
        self.discriminator_obs_normalizer(discriminator_obs)

    def act(self, obs: TensorDict, z: torch.Tensor, clip: float | None = None, **kwargs) -> torch.Tensor:
        normalized_obs = self.get_actor_obs(obs)
        obs_z = torch.cat([normalized_obs, z], dim=-1)
        self.update_distribution(obs_z, normalized_obs)
        assert self.distribution is not None
        return self.distribution.sample(clip=clip)

    def F(
        self,
        obs: TensorDict,
        z: torch.Tensor,
        action: torch.Tensor,
        use_target: bool = False,
    ) -> torch.Tensor:
        normalized_obs = self.get_critic_obs(obs)
        obs_z = torch.cat([normalized_obs, z], dim=-1)
        obs_action = torch.cat([normalized_obs, action], dim=-1)

        forward_map = self.target_forward_map if use_target else self.forward_map
        assert forward_map is not None
        return forward_map(obs_z, obs_action)

    def B(self, goal_obs: TensorDict, use_target: bool = False) -> torch.Tensor:
        normalized_obs = self.get_backward_obs(goal_obs)
        backward_map = self.target_backward_map if use_target else self.backward_map
        assert backward_map is not None, "backward_map is None. Did you call `init_targets` before training?"
        return backward_map(normalized_obs)

    def D(self, obs: TensorDict, z: torch.Tensor) -> torch.Tensor:
        normalized_obs = self.get_discriminator_obs(obs)
        return self.discriminator(torch.cat([normalized_obs, z], dim=-1))

    def evaluate_discriminator(
        self,
        obs: TensorDict,
        z: torch.Tensor,
        action: torch.Tensor,
        use_target: bool = False,
    ) -> torch.Tensor:
        normalized_obs = self.get_critic_obs(obs)
        obs_z = torch.cat([normalized_obs, z], dim=-1)
        obs_action = torch.cat([normalized_obs, action], dim=-1)

        disc_critic = self.target_disc_critic if use_target else self.disc_critic
        assert disc_critic is not None, "disc_critic is None. Did you call `init_targets` before training?"
        return disc_critic(obs_z, obs_action)

    def evaluate_aux(
        self,
        obs: TensorDict,
        z: torch.Tensor,
        action: torch.Tensor,
        use_target: bool = False,
    ) -> torch.Tensor:
        normalized_obs = self.get_critic_obs(obs)
        obs_z = torch.cat([normalized_obs, z], dim=-1)
        obs_action = torch.cat([normalized_obs, action], dim=-1)

        aux_critic = self.target_aux_critic if use_target else self.aux_critic
        assert aux_critic is not None, "aux_critic is None. Did you call `init_targets` before training?"
        return aux_critic(obs_z, obs_action)

    """
    Inference
    """

    @torch.no_grad()
    def goal_inference(self, goal_obs: TensorDict) -> torch.Tensor:
        z = self.B(goal_obs)
        return self.z_normalizer(z)

    @torch.no_grad()
    def reward_inference(
        self,
        next_obs: TensorDict,
        reward: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        wr = reward if weight is None else reward * weight
        B = self.B(next_obs)
        z = torch.matmul(wr.T, B)
        return self.z_normalizer(z)

    @torch.no_grad()
    def weighted_reward_inference(self, next_obs: TensorDict, reward: torch.Tensor) -> torch.Tensor:
        return self.reward_inference(next_obs, reward, nn.functional.softmax(10 * reward, dim=0))

    """
    Public utils
    """

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True  # training resumes

    def soft_update_targets(self) -> None:
        self._soft_update_params(self._forward_paramlist, self._target_forward_paramlist, self.fb_tau)
        self._soft_update_params(self._backward_paramlist, self._target_backward_paramlist, self.fb_tau)
        self._soft_update_params(self._disc_critic_paramlist, self._target_disc_critic_paramlist, self.critic_tau)
        self._soft_update_params(self._aux_critic_paramlist, self._target_aux_critic_paramlist, self.critic_tau)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        with torch.no_grad(), eval_mode(self.actor_obs_normalizer):
            obs_tens = get_obs(obs, self.obs_groups["policy"])
            return self.actor_obs_normalizer(obs_tens)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        with torch.no_grad(), eval_mode(self.critic_obs_normalizer):
            obs_tens = get_obs(obs, self.obs_groups["critic"])
            return self.critic_obs_normalizer(obs_tens)

    def get_backward_obs(self, obs: TensorDict) -> torch.Tensor:
        with torch.no_grad(), eval_mode(self.backward_obs_normalizer):
            obs_tens = get_obs(obs, self.obs_groups["backward"])
            return self.backward_obs_normalizer(obs_tens)

    def get_discriminator_obs(self, obs: TensorDict) -> torch.Tensor:
        with torch.no_grad(), eval_mode(self.discriminator_obs_normalizer):
            obs_tens = get_obs(obs, self.obs_groups["discriminator"])
            return self.discriminator_obs_normalizer(obs_tens)

    """
    Helpers
    """

    def _soft_update_params(
        self,
        params: tuple[torch.Tensor, ...] | None,
        target_params: tuple[torch.Tensor, ...] | None,
        tau: float,
    ) -> None:
        assert params is not None and target_params is not None, (
            "Params or target params is None. Ensure that you call `init_targets` before training."
        )
        torch._foreach_mul_(target_params, tau)  # type: ignore
        torch._foreach_add_(target_params, params, alpha=(1 - tau))  # type: ignore

    def _get_obs(self, obs: TensorDict, group: str) -> torch.Tensor:
        obs_list = []
        for obs_group in self.obs_groups[group]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)
