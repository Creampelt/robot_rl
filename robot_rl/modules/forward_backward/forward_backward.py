from typing import Sequence, cast
import copy
import math

import torch
import torch.nn as nn
import torch.distributions as pyd
from tensordict import TensorDict

from robot_rl.networks import ParallelMLP, ParallelEmbedding, MLP, ScaledNormalization, TruncatedNormal
from robot_rl.utils import get_obs_dimensions


def simple_embedding(input_dim: int, output_dim: int, hidden_dims: Sequence[int], num_parallel: int = 1) -> ParallelMLP:
    activation = ["tanh"] + ["relu" for _ in hidden_dims] + ["relu"]
    return ParallelMLP(
        input_dim,
        output_dim,
        hidden_dims,
        num_parallel,
        activation=activation,
        last_activation=True,
        reset_params=True,
    )


class _Forward(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        z_dim: int,
        action_dim: int,
        num_parallel: int,
        hidden_dims: Sequence[int],
        num_embedding_layers: int,
    ) -> None:
        assert len(hidden_dims) > num_embedding_layers, (
            f"Expected at least {num_embedding_layers + 1} hidden layers, got {len(hidden_dims)}."
        )
        super().__init__()
        self.num_parallel = num_parallel

        f_input_dim = hidden_dims[num_embedding_layers]
        embedding_layers = hidden_dims[:num_embedding_layers]

        self.embed_z = ParallelEmbedding(obs_dim + z_dim, f_input_dim // 2, embedding_layers, num_parallel)
        self.embed_sa = ParallelEmbedding(obs_dim + action_dim, f_input_dim // 2, embedding_layers, num_parallel)
        self.F = ParallelMLP(
            f_input_dim,
            z_dim,
            hidden_dims[num_embedding_layers:],
            self.num_parallel,
            activation="relu",
        )

    def forward(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
            action = action.expand(self.num_parallel, -1, -1)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))
        sa_embedding = self.embed_sa(torch.cat([obs, action], dim=-1))

        return self.F(torch.cat([z_embedding, sa_embedding], dim=-1))


class _Backward(nn.Module):
    def __init__(
        self,
        goal_dim: int,
        z_dim: int,
        hidden_dims: Sequence[int],
        normalize_output: bool = True,
        reset_params: bool = True,
    ) -> None:
        assert hidden_dims, "Must have at least one hidden layer."
        super().__init__()

        layers = [nn.Linear(goal_dim, hidden_dims[0]), nn.LayerNorm(hidden_dims[0]), nn.Tanh()]
        for i, dim in enumerate(hidden_dims[:-1]):
            layers.append(nn.Linear(dim, hidden_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], z_dim))

        if normalize_output:
            layers.append(ScaledNormalization())

        self.model = nn.Sequential(*layers)

        if reset_params:
            self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def reset_parameters(self) -> None:
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight.data)
                if hasattr(layer.bias, "data"):
                    layer.bias.data.fill_(0.0)


class _Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        z_dim: int,
        action_dim: int,
        num_parallel: int,
        hidden_dims: Sequence[int],
        init_noise_std: float = 0.2,
        num_embedding_layers: int = 2,
    ) -> None:
        assert len(hidden_dims) > num_embedding_layers, (
            f"Expected at least {num_embedding_layers + 1} hidden layers, got {len(hidden_dims)}."
        )
        super().__init__()

        self.num_parallel = num_parallel

        embed_out_dim = hidden_dims[num_embedding_layers]
        embedding_dims = hidden_dims[:num_embedding_layers]

        self.embed_z = ParallelEmbedding(obs_dim + z_dim, embed_out_dim // 2, embedding_dims, num_parallel)
        self.embed_s = ParallelEmbedding(obs_dim, embed_out_dim // 2, embedding_dims, num_parallel)
        self.policy = ParallelMLP(
            embed_out_dim,
            action_dim,
            hidden_dims[num_embedding_layers:],
            num_parallel,
            activation="relu",
        )
        self.std = init_noise_std
        # TODO: non-fixed std?
        # self.std = nn.Parameter(init_noise_std * torch.ones(embed_out_dim))
        self.distribution: TruncatedNormal | None = None

    @property
    def action_mean(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.stddev

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> TruncatedNormal:
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))  # num_parallel x embed_out_dim // 2
        s_embedding = self.embed_s(obs)  # num_parallel x embed_out_dim // 2
        raw_out = self.policy(torch.cat([s_embedding, z_embedding], dim=-1))

        mu = torch.tanh(raw_out)
        self.distribution = TruncatedNormal(mu, self.std)
        return self.distribution


class ForwardBackward(nn.Module):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        z_dim: int = 100,
        init_noise_std: float = 0.2,
        actor_num_parallel: int = 1,
        critic_num_parallel: int = 2,
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
        self.critic_num_parallel = critic_num_parallel

        # get the observation dimensions
        self.obs_groups = obs_groups
        self.num_actor_obs = get_obs_dimensions(obs, obs_groups["policy"])
        self.num_critic_obs = get_obs_dimensions(obs, obs_groups["critic"])

        # actor
        self.actor = _Actor(
            self.num_actor_obs,
            z_dim,
            num_actions,
            actor_num_parallel,
            actor_hidden_dims,
            init_noise_std,
            actor_num_embedding_layers,
        )
        self.actor_obs_normalizer = (
            nn.BatchNorm1d(self.num_actor_obs, affine=False, momentum=0.01)
            if actor_obs_normalization
            else nn.Identity()
        )

        # forward and backward mappings
        self.forward_map = _Forward(
            self.num_critic_obs,
            z_dim,
            num_actions,
            critic_num_parallel,
            forward_hidden_dims,
            forward_num_embedding_layers,
        )
        self.backward_map = _Backward(self.num_critic_obs, z_dim, backward_hidden_dims, backward_out_normalization)
        self.critic_obs_normalizer = (
            nn.BatchNorm1d(self.num_critic_obs, affine=False, momentum=0.01)
            if critic_obs_normalization
            else nn.Identity()
        )

        # create z normalizer module
        self.z_normalizer = ScaledNormalization() if z_normalization else nn.Identity()

        # create target networks and paramlists
        self.target_forward_map = cast(_Forward, copy.deepcopy(self.forward_map))
        self.target_backward_map = cast(_Backward, copy.deepcopy(self.backward_map))

        self._forward_paramlist = tuple(x.data for x in self.forward_map.parameters())
        self._target_forward_paramlist = tuple(x.data for x in self.target_forward_map.parameters())
        self._backward_paramlist = tuple(x.data for x in self.backward_map.parameters())
        self._target_backward_paramlist = tuple(x.data for x in self.target_backward_map.parameters())

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def act(self, obs: TensorDict, z: torch.Tensor, clip: float | None = None, **kwargs) -> torch.Tensor:
        normalized_obs = self.get_actor_obs(obs)
        normalized_obs = self.actor_obs_normalizer(normalized_obs)
        dist = self.actor(normalized_obs, z)
        return dist.sample(clip=clip)

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

    def F(self, obs: TensorDict, z: torch.Tensor, action: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        normalized_obs = self.get_critic_obs(obs)
        normalized_obs = self.critic_obs_normalizer(normalized_obs)
        if use_target:
            return self.target_forward_map(normalized_obs, z, action)
        else:
            return self.forward_map(normalized_obs, z, action)

    def B(self, goal_obs: TensorDict, use_target: bool = False) -> torch.Tensor:
        normalized_obs = self.get_critic_obs(goal_obs)
        normalized_obs = self.critic_obs_normalizer(normalized_obs)
        if use_target:
            return self.target_backward_map(normalized_obs)
        else:
            return self.backward_map(normalized_obs)

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

    @property
    def action_mean(self) -> torch.Tensor:
        return self.actor.action_mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.actor.action_std

    def soft_update_targets(self, tau: float) -> None:
        self._soft_update_params(self._forward_paramlist, self._target_forward_paramlist, tau)
        self._soft_update_params(self._backward_paramlist, self._target_backward_paramlist, tau)

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
        tau: float,
    ) -> None:
        torch._foreach_mul_(target_params, 1 - tau)  # type: ignore
        torch._foreach_add_(target_params, params, alpha=tau)  # type: ignore

    def _get_obs(self, obs: TensorDict, group: str) -> torch.Tensor:
        obs_list = []
        for obs_group in self.obs_groups[group]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)
