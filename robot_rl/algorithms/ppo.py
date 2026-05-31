# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict
from typing import Any

from robot_rl.env import VecEnv
from robot_rl.extensions import RandomNetworkDistillation, resolve_rnd_config, resolve_symmetry_config
from robot_rl.models import MLPModel
from robot_rl.storage import RolloutStorage
from robot_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer


class _SharedMemoryInferencePolicy(nn.Module):
    """Adapter that chains a shared memory module into actor inference."""

    is_recurrent: bool = True

    def __init__(self, memory: nn.Module, actor: MLPModel) -> None:
        super().__init__()
        self.memory = memory
        self.actor = actor

    @property
    def output_mean(self) -> torch.Tensor:
        """Return the mean of the current output distribution."""
        return self.actor.output_mean

    @property
    def output_std(self) -> torch.Tensor:
        """Return the standard deviation of the current output distribution."""
        return self.actor.output_std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Return the entropy of the current output distribution."""
        return self.actor.output_entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Return raw parameters of the current output distribution."""
        return self.actor.output_distribution_params

    def forward(self, obs: TensorDict, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        latent = self.memory(obs)
        return self.actor.forward_from_latent(latent, *args, **kwargs)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.memory.reset(dones)
        self.actor.reset(dones)


class PPO:
    """Proximal Policy Optimization algorithm.

    Reference:
        - Schulman et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
    """

    actor: MLPModel
    """The actor model."""

    critic: MLPModel
    """The critic model."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        adaptive_lr_once_per_iteration: bool = False,
        device: str = "cpu",
        # Optional shared memory module (consumed by both actor and critic as heads)
        memory: nn.Module | None = None,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # Meta-RL parameters
        meta_rl_cfg: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the algorithm with models, storage, and optimization settings."""
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg:
            # Extract parameters used in ppo
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # Resolve the data augmentation function (supports string names or direct callables)
            symmetry_cfg["data_augmentation_func"] = resolve_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    f"Symmetry configuration exists but the function is not callable: "
                    f"{symmetry_cfg['data_augmentation_func']}"
                )
            # Check if the policy is compatible with symmetry
            if actor.is_recurrent or critic.is_recurrent or memory is not None:
                raise ValueError(
                    "Symmetry augmentation is not supported for recurrent policies (including shared memory)."
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # Meta RL components
        self.meta_rl = meta_rl_cfg is not None
        self.detach_critic_memory = False
        if meta_rl_cfg is not None:
            self.num_episodes_per_trial: int = meta_rl_cfg["num_episodes_per_trial"]
            self.detach_critic_memory = meta_rl_cfg.get("detach_critic_memory", False)
            # Wait to initialize episode counter since we use data shape to get num_envs
            self.ep_counter: torch.Tensor | None = None

        # PPO components
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        # Shared memory (optional). When set, actor/critic must be MLP heads on top of the memory's latent.
        if memory is not None and (actor.is_recurrent or critic.is_recurrent):
            raise ValueError(
                "Shared memory is not supported with recurrent actor/critic models. "
                "When `meta_rl_cfg.memory` is set, actor and critic must be plain MLP heads."
            )
        self.memory: nn.Module | None = memory.to(self.device) if memory is not None else None

        # Create the optimizer
        params: Any = chain(self.actor.parameters(), self.critic.parameters())
        if self.memory is not None:
            params = chain(params, self.memory.parameters())
        self.optimizer = resolve_optimizer(optimizer)(params, lr=learning_rate)  # type: ignore

        # Add storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.adaptive_lr_once_per_iteration = adaptive_lr_once_per_iteration
        self.learning_rate = learning_rate
        # Bounds for the once-per-iteration adaptive LR (per-minibatch uses 1e-2/1e-5).
        self.max_learning_rate = 1e-3
        self.min_learning_rate = 1e-5
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        if self.memory is not None:
            self.transition.memory_hidden_state = self.memory.get_hidden_state()
            self.transition.hidden_states = (None, None)
            latent = self.memory(obs).detach()
            self.transition.actions = self.actor.forward_from_latent(latent, stochastic_output=True).detach()
            # Include additional critic obs for asymmetric actor-critic
            self.transition.values = self.critic.forward_from_latent(latent, obs=obs).detach()
        else:
            self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
            self.transition.actions = self.actor(obs, stochastic_output=True).detach()
            self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()  # type: ignore
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        """Record one environment step and update the normalizers."""
        # Update the normalizers
        if self.memory is not None:
            self.memory.update_normalization(obs)
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Record trial done status for meta-RL
        if self.meta_rl:
            # Initialize episode counter if necessary
            if self.ep_counter is None:
                self.ep_counter = torch.zeros(dones.shape[0], dtype=torch.long, device=self.device)
            # Increment episode counter
            new_ids = (dones > 0).nonzero(as_tuple=False)
            self.ep_counter[new_ids] += 1
            # Compute whether trial is finished
            trial_dones = (dones.bool() & (self.ep_counter % self.num_episodes_per_trial == 0)).byte()
            self.transition.meta_dones = trial_dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
                1,
            )

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()

        # For meta-RL environments, we only want to reset hidden states at end of trial
        # Otherwise we reset at end of episode
        do_reset = trial_dones if self.meta_rl else dones
        if self.memory is not None:
            self.memory.reset(do_reset)
        self.actor.reset(do_reset)
        self.critic.reset(do_reset)
        return trial_dones.nonzero(as_tuple=False).squeeze(1) if self.meta_rl else None

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute return and advantage targets from stored transitions."""
        st = self.storage
        # Compute value for the last step
        if self.memory is not None:
            latent = self.memory(obs).detach()
            last_values = self.critic.forward_from_latent(latent, obs=obs).detach()
        else:
            last_values = self.critic(obs).detach()
        # GAE runs over storage tensors; bring the bootstrap value to the storage device.
        last_values = last_values.to(st.device)
        # Compute returns and advantages
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - st.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            st.returns[step] = advantage + st.values[step]
        # Compute the advantages
        st.advantages = st.returns - st.values
        # Normalize the advantages if per minibatch normalization is not used
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_log_prob = 0
        sum_kl = 0.0
        max_kl = 0.0
        kl_first_minibatch: float | None = None
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # Get mini batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent or self.memory is not None:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs, device=self.device
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
                device=self.device,
            )

        # Iterate over batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                # Compute number of augmentations per sample
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            if self.memory is not None:
                # Run shared memory once per mini-batch; both heads consume the same unpadded latent.
                latent = self.memory(
                    batch.observations,
                    masks=batch.masks,
                    hidden_state=batch.memory_hidden_state,
                )
                self.actor.forward_from_latent(latent, stochastic_output=True)
                actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
                # Optionally stop the value loss from backpropagating into the shared memory
                critic_latent = latent.detach() if self.detach_critic_memory else latent
                values = self.critic.forward_from_latent(critic_latent, obs=batch.observations, masks=batch.masks)
            else:
                self.actor(
                    batch.observations,
                    masks=batch.masks,
                    hidden_state=batch.hidden_states[0],
                    stochastic_output=True,
                )
                actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
                values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            # Note: We only keep the distribution parameters and entropy of the first augmentation (the original one)
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Compute KL divergence (always, for logging) and adapt the learning rate if scheduled
            with torch.inference_mode():
                kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                kl_mean = torch.mean(kl)

                # Reduce the KL divergence across all GPUs
                if self.is_multi_gpu:
                    torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                    kl_mean /= self.gpu_world_size

                kl_value = kl_mean.item()
                if kl_first_minibatch is None:
                    kl_first_minibatch = kl_value
                sum_kl += kl_value
                if kl_value > max_kl:
                    max_kl = kl_value

                # Per-minibatch adaptive LR
                if (
                    self.desired_kl is not None
                    and self.schedule == "adaptive"
                    and not self.adaptive_lr_once_per_iteration
                ):
                    if kl_value > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif self.desired_kl / 2.0 > kl_value > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions = self.actor(batch.observations.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the batch.actions from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions[original_batch_size:], actions_mean_symm.detach()[original_batch_size:]
                )
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            if self.rnd:
                # Extract the rnd_state
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])  # type: ignore
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            if self.memory is not None:
                nn.utils.clip_grad_norm_(self.memory.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_log_prob += actions_log_prob.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches

        # Adapt the LR once per iteration
        if self.desired_kl is not None and self.schedule == "adaptive" and self.adaptive_lr_once_per_iteration:
            mean_kl_iter = sum_kl / num_updates
            if self.gpu_global_rank == 0:
                if mean_kl_iter > self.desired_kl * 2.0:
                    self.learning_rate = max(self.min_learning_rate, self.learning_rate / 1.5)
                elif 0.0 < mean_kl_iter < self.desired_kl / 2.0:
                    self.learning_rate = min(self.max_learning_rate, self.learning_rate * 1.5)
            if self.is_multi_gpu:
                lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                torch.distributed.broadcast(lr_tensor, src=0)
                self.learning_rate = lr_tensor.item()
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_log_prob /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "log_prob": mean_log_prob,
        }
        loss_dict["kl_mean"] = sum_kl / num_updates
        loss_dict["kl_max"] = max_kl
        if kl_first_minibatch is not None:
            loss_dict["kl_first_minibatch"] = kl_first_minibatch
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.actor.train()
        self.critic.train()
        if self.memory is not None:
            self.memory.train()
        if self.rnd:
            self.rnd.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.actor.eval()
        self.critic.eval()
        if self.memory is not None:
            self.memory.eval()
        if self.rnd:
            self.rnd.eval()

    def eval(self, env: VecEnv, max_steps: int = 200) -> list[dict[str, torch.Tensor]]:
        """Run a deterministic evaluation rollout for ``max_steps`` environment steps.

        For vanilla PPO an "eval" rollout is just the inference loop -- no learning, no
        stochasticity in the action (use the actor's deterministic mean), no transition
        storage. Subclasses (FB-CPR etc.) override this to run a task-specific eval (e.g.
        motion replay + EMD). Returns an empty list of per-batch info dicts for API parity
        with :meth:`fb_cpr.FbCpr.eval`.

        The caller is responsible for any env-side video wrapping: each ``env.step()`` here
        will produce a rendered frame for any active ``gym.wrappers.RecordVideo`` wrapping
        the env.
        """
        was_training = self.actor.training
        self.eval_mode()
        if hasattr(env, "eval_mode"):
            env.eval_mode()

        obs = env.get_observations() if hasattr(env, "get_observations") else env.reset()[0]
        # Reset any recurrent / TXL state on actor + memory so video starts from a clean context.
        if hasattr(self.actor, "reset"):
            self.actor.reset()
        if self.memory is not None and hasattr(self.memory, "reset"):
            self.memory.reset()

        with torch.inference_mode():
            for _ in range(max_steps):
                if self.memory is not None:
                    latent = self.memory(obs)
                    actions = self.actor.forward_from_latent(latent, stochastic_output=False)
                else:
                    actions = self.actor(obs, stochastic_output=False)
                obs, _, _, _ = env.step(actions)

        if was_training:
            self.train_mode()
            if hasattr(env, "train_mode"):
                env.train_mode()
        return []

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.memory is not None:
            saved_dict["memory_state_dict"] = self.memory.state_dict()
        if self.rnd:
            saved_dict["rnd_state_dict"] = self.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.rnd_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        # If no load_cfg is provided, load all models and states
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "memory": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
            }

        # Load the specified models
        if load_cfg.get("actor"):
            self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("memory") and self.memory is not None and "memory_state_dict" in loaded_dict:
            self.memory.load_state_dict(loaded_dict["memory_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> nn.Module:
        """Get the policy model.

        Wraps the actor with the shared memory module when configured, since the actor is then a head over a
        precomputed latent and cannot consume raw obs.
        """
        if self.memory is not None:
            return _SharedMemoryInferencePolicy(self.memory, self.actor)
        return self.actor

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
        """Construct the PPO algorithm."""
        # Resolve class callables
        alg_class: type[PPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        # Optional shared memory config
        meta_rl_cfg = cfg["algorithm"].get("meta_rl_cfg")
        shared_memory_cfg: dict | None = meta_rl_cfg.get("memory") if isinstance(meta_rl_cfg, dict) else None

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve RND config if used
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

        # Resolve symmetry config if used
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # Build the optional shared memory module first so we can size actor/critic heads from its latent_dim
        memory: nn.Module | None = None
        head_kwargs: dict = {}
        critic_head_kwargs: dict = {}
        if shared_memory_cfg is not None:
            mem_class: type[MLPModel] = resolve_callable(shared_memory_cfg.pop("class_name"))  # type: ignore
            memory = mem_class(obs, cfg["obs_groups"], "actor", 1, memory_only=True, **shared_memory_cfg).to(device)
            print(f"Shared Memory Model: {memory}")
            head_kwargs["input_dim_override"] = memory.latent_dim  # type: ignore[attr-defined]
            # Critic consumes privileged obs + memory latent
            critic_head_kwargs["input_dim_override"] = memory.latent_dim  # type: ignore[attr-defined]
            critic_head_kwargs["append_obs_groups"] = True

        # Initialize the policy
        actor: MLPModel = actor_class(
            obs, cfg["obs_groups"], "actor", env.num_actions, **head_kwargs, **cfg["actor"]
        ).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):  # Share CNN encoders between actor and critic
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_head_kwargs, **cfg["critic"]).to(
            device
        )
        print(f"Critic Model: {critic}")

        # Initialize the storage. Use "meta_rl" when meta-RL is configured so the trajectory generator splits
        # at trial boundaries (where memory was reset) rather than episode boundaries.
        training_type = "meta_rl" if cfg["algorithm"].get("meta_rl_cfg") is not None else "rl"
        storage_device = cfg.get("storage_device")
        if storage_device is None:
            storage_device = device
        storage = RolloutStorage(
            training_type, env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], storage_device
        )

        # Initialize the algorithm
        alg: PPO = alg_class(
            actor,
            critic,
            storage,
            device=device,
            memory=memory,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self.actor.state_dict(), self.critic.state_dict()]
        if self.memory is not None:
            model_params.append(self.memory.state_dict())
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self.actor.load_state_dict(model_params[0])
        self.critic.load_state_dict(model_params[1])
        idx = 2
        if self.memory is not None:
            self.memory.load_state_dict(model_params[idx])
            idx += 1
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[idx])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        all_params = chain(self.actor.parameters(), self.critic.parameters())
        if self.memory is not None:
            all_params = chain(all_params, self.memory.parameters())
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
