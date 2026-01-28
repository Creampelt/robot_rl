from typing import Sequence, cast
import copy
import math

import torch
import torch.nn as nn
import torch.distributions as pyd
from tensordict import TensorDict

from robot_rl.networks import (
    ParallelLinear,
    ParallelLayerNorm,
    ScaledNormalization,
    TruncatedNormal,
)
from robot_rl.utils import get_obs_dimensions, resolve_nn_activation, reset_parameters
import matplotlib.pyplot as plt


class _SimpleEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = [],
        num_parallel: int = 1,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        layers = [
            ParallelLinear(input_dim, hidden_dims[0], num_parallel),
            ParallelLayerNorm((hidden_dims[0],), num_parallel),
            nn.Tanh(),
        ]
        for i, dim in enumerate(hidden_dims[:-1]):
            layers.append(
                ParallelLinear(
                    dim,
                    hidden_dims[i + 1],
                    num_parallel,
                )
            )
            layers.append(resolve_nn_activation(activation))
        layers.append(
            ParallelLinear(
                hidden_dims[-1],
                output_dim,
                num_parallel,
            )
        )
        layers.append(resolve_nn_activation(activation))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _SimpleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int | Sequence[int],
        output_dim: int,
        hidden_dims: Sequence[int],
        num_embedding_layers: int = 0,
        num_parallel: int = 1,
        last_activation: str | nn.Module = "identity",
    ) -> None:
        super().__init__()
        assert hidden_dims, "hidden_dims must have at least one layer."
        self.num_parallel = num_parallel
        self.embeddings: nn.ModuleList | None = None

        layers = []
        if num_embedding_layers > 0:
            assert isinstance(input_dim, Sequence)
            embedding_dims = hidden_dims[: num_embedding_layers - 1]
            embed_out_dim = hidden_dims[num_embedding_layers - 1] // num_embedding_layers

            self.embeddings = nn.ModuleList(
                [
                    _SimpleEmbedding(
                        d,
                        embed_out_dim,
                        embedding_dims,
                        num_parallel,
                    )
                    for d in input_dim
                ]
            )
        else:
            assert isinstance(input_dim, int)
            layers += [
                ParallelLinear(input_dim, hidden_dims[0], num_parallel),
                ParallelLayerNorm((hidden_dims[0],), num_parallel),
                nn.Tanh(),
            ]

        for i, dim in enumerate(hidden_dims[num_embedding_layers - 1 : -1]):
            layers.append(
                ParallelLinear(
                    dim,
                    hidden_dims[i + 1],
                    num_parallel,
                )
            )
            layers.append(nn.ReLU())
        layers.append(ParallelLinear(hidden_dims[-1], output_dim, num_parallel))
        if isinstance(last_activation, str):
            layers.append(resolve_nn_activation(last_activation))
        else:
            layers.append(last_activation)

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if self.num_parallel > 1:
            if isinstance(x, tuple):
                x = tuple([x_i.expand(self.num_parallel, -1, -1) for x_i in x])
            else:
                x = x.expand(self.num_parallel, -1, -1)

        if self.embeddings is not None:
            assert isinstance(x, tuple) and len(x) == len(self.embeddings)
            embed_out = [e(x_i) for e, x_i in zip(self.embeddings, x)]
            x = torch.concat(embed_out, dim=-1)
        assert isinstance(x, torch.Tensor)
        return self.model(x)


class ForwardBackward(nn.Module):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        z_dim: int = 100,
        tau: float = 0.01,
        init_noise_std: float = 0.2,
        actor_num_parallel: int = 1,
        forward_num_parallel: int = 2,
        backward_num_parallel: int = 1,
        actor_obs_normalization: bool = True,
        critic_obs_normalization: bool = True,
        z_normalization: bool = True,
        backward_out_normalization: bool = True,
        actor_hidden_dims: Sequence[int] = [1024, 1024, 1024],
        actor_num_embedding_layers: int = 2,
        forward_hidden_dims: Sequence[int] = [1024, 1024, 1024],
        forward_num_embedding_layers: int = 2,
        backward_hidden_dims: Sequence[int] = [256, 256],
        **kwargs,
    ):
        if kwargs:
            print(
                f"{self.__class__.__name__}.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.z_dim = z_dim
        self.tau = tau
        self.init_noise_std = init_noise_std

        # get the observation dimensions
        self.obs_groups = obs_groups
        self.num_actor_obs = get_obs_dimensions(obs, obs_groups["policy"])
        self.num_critic_obs = get_obs_dimensions(obs, obs_groups["critic"])

        # actor
        self.actor = _SimpleMLP(
            (self.num_actor_obs + z_dim, self.num_actor_obs),
            num_actions,
            actor_hidden_dims,
            actor_num_embedding_layers,
            actor_num_parallel,
            last_activation="tanh",
        )
        self.actor_obs_normalizer = (
            nn.BatchNorm1d(self.num_actor_obs, affine=False, momentum=0.01)
            if actor_obs_normalization
            else nn.Identity()
        )

        # forward and backward mappings
        self.forward_map = _SimpleMLP(
            (self.num_critic_obs + self.z_dim, self.num_critic_obs + num_actions),
            self.z_dim,
            forward_hidden_dims,
            forward_num_embedding_layers,
            forward_num_parallel,
        )

        self.backward_map = _SimpleMLP(
            self.num_critic_obs,
            z_dim,
            backward_hidden_dims,
            num_parallel=backward_num_parallel,
            last_activation=ScaledNormalization(),
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
        self.target_forward_map: _SimpleMLP | None = None
        self.target_backward_map: _SimpleMLP | None = None

        self._forward_paramlist: tuple[torch.Tensor, ...] | None = None
        self._backward_paramlist: tuple[torch.Tensor, ...] | None = None
        self._target_forward_paramlist: tuple[torch.Tensor, ...] | None = None
        self._target_backward_paramlist: tuple[torch.Tensor, ...] | None = None

    def train(self, mode: bool = True, device: str | None = None):
        self.target_forward_map = copy.deepcopy(self.forward_map).to(device)
        self.target_backward_map = copy.deepcopy(self.backward_map).to(device)
        # create paramlists
        self._forward_paramlist = tuple(x.data for x in self.forward_map.parameters())
        self._backward_paramlist = tuple(x.data for x in self.backward_map.parameters())
        self._target_forward_paramlist = tuple(x.data for x in self.target_forward_map.parameters())
        self._target_backward_paramlist = tuple(x.data for x in self.target_backward_map.parameters())
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

    def update_distribution(self, x: tuple[torch.Tensor, torch.Tensor]) -> None:
        mean = self.actor(x)
        self.distribution = TruncatedNormal(mean, self.init_noise_std)

    def act(self, obs: TensorDict, z: torch.Tensor, clip: float | None = None, **kwargs) -> torch.Tensor:
        normalized_obs = self.get_actor_obs(obs)
        normalized_obs = self.actor_obs_normalizer(normalized_obs)
        obs_z = torch.cat([normalized_obs, z], dim=-1)
        self.update_distribution((obs_z, normalized_obs))
        return self.distribution.sample(clip=clip)

    def F(self, obs: TensorDict, z: torch.Tensor, action: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        # Note: does not affect torch gradient tracking. Only computes stopgrad from TD targets
        normalized_obs = self.get_critic_obs(obs)
        normalized_obs = self.critic_obs_normalizer(normalized_obs)
        obs_z = torch.cat([normalized_obs, z], dim=-1)
        obs_action = torch.cat([normalized_obs, action], dim=-1)

        forward_map = self.target_forward_map if use_target else self.forward_map
        return forward_map((obs_z, obs_action))

    def B(self, goal_obs: TensorDict, use_target: bool = False) -> torch.Tensor:
        normalized_obs = self.get_critic_obs(goal_obs)
        normalized_obs = self.critic_obs_normalizer(normalized_obs)
        backward_map = self.target_backward_map if use_target else self.backward_map
        return backward_map(normalized_obs)

    def evaluate(
        self,
        obs: TensorDict,
        z: torch.Tensor,
        action: torch.Tensor,
        use_target: bool = False,
    ) -> torch.Tensor:
        Fs = self.F(obs, z, action, use_target=use_target)
        return torch.sum(Fs * z, dim=-1)  # num_parallel x batch_size

    def sample_z(
        self,
        size: int,
        goal_obs: TensorDict | None = None,
        goal_ratio: float = 0.5,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Sample random z from normal distribution, normalize if enabled."""
        z = torch.randn((size, self.z_dim), dtype=torch.float32, device=device)
        if goal_obs is not None:
            perm = torch.randperm(size, device=device)
            goal_z = self.goal_inference(goal_obs[perm])
            mask = torch.rand((size, 1), device=device) < goal_ratio
            z = torch.where(mask, goal_z, z)
        return self.z_normalizer(z)

    """
    Inference
    """

    @torch.no_grad()
    def goal_inference(self, goal_obs: TensorDict) -> torch.Tensor:
        return self.B(goal_obs)

    @torch.no_grad()
    def reward_inference(
        self,
        next_obs: TensorDict,
        reward: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        wr = reward if weight is None else reward * weight
        B = self.goal_inference(next_obs)
        z = torch.matmul(wr.T, B)
        return self.z_normalizer(z)

    @torch.no_grad()
    def weighted_reward_inference(self, next_obs: TensorDict, reward: torch.Tensor) -> torch.Tensor:
        return self.reward_inference(next_obs, reward, nn.functional.softmax(10 * reward, dim=0))

    """
    Public utils
    """

    def soft_update_targets(self) -> None:
        self._soft_update_params(self._forward_paramlist, self._target_forward_paramlist)
        self._soft_update_params(self._backward_paramlist, self._target_backward_paramlist)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._get_obs(obs, "policy")

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._get_obs(obs, "critic")

    """
    Helpers
    """

    def _soft_update_params(
        self,
        params: tuple[torch.Tensor, ...],
        target_params: tuple[torch.Tensor, ...],
    ) -> None:
        torch._foreach_mul_(target_params, self.tau)  # type: ignore
        torch._foreach_add_(target_params, params, alpha=(1 - self.tau))  # type: ignore

    def _get_obs(self, obs: TensorDict, group: str) -> torch.Tensor:
        obs_list = []
        for obs_group in self.obs_groups[group]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)
