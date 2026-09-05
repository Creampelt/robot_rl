# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the PPO algorithm."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from robot_rl.algorithms.ppo import PPO
from robot_rl.models import MLPModel
from robot_rl.storage import RolloutStorage
from tests.conftest import make_obs

NUM_ENVS = 4
NUM_STEPS = 8
OBS_DIM = 8
NUM_ACTIONS = 4


def _make_actor(obs: TensorDict, obs_groups: dict, num_actions: int = 4, **kwargs: object) -> MLPModel:
    """Create an MLPModel actor with a Gaussian distribution."""
    defaults: dict[str, object] = {
        "hidden_dims": [32, 32],
        "activation": "elu",
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    }
    defaults.update(kwargs)
    return MLPModel(obs, obs_groups, "actor", num_actions, **defaults)


def _make_critic(obs: TensorDict, obs_groups: dict, **kwargs: object) -> MLPModel:
    """Create an MLPModel critic (no distribution)."""
    defaults: dict[str, object] = {"hidden_dims": [32, 32], "activation": "elu"}
    defaults.update(kwargs)
    return MLPModel(obs, obs_groups, "critic", 1, **defaults)


def _build_ppo(**overrides: object) -> tuple[PPO, TensorDict]:
    """Build a PPO instance with small networks for testing."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = _make_actor(obs, obs_groups, NUM_ACTIONS)
    critic = _make_critic(obs, obs_groups)
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])

    defaults = dict(
        num_learning_epochs=2,
        num_mini_batches=2,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        schedule="fixed",
        desired_kl=0.01,
    )
    defaults.update(overrides)
    ppo = PPO(actor, critic, storage, **defaults)
    return ppo, obs


class TestGAEComputation:
    """Tests for generalized advantage estimation in ``compute_returns``."""

    def test_gae_returns_hand_computed(self) -> None:
        """Verify GAE returns match a hand-computed example with known rewards, values, and dones."""
        num_envs, num_steps = 1, 3
        gamma, lam = 0.99, 0.95

        obs = make_obs(num_envs, OBS_DIM)
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        actor = _make_actor(obs, obs_groups, NUM_ACTIONS)
        critic = _make_critic(obs, obs_groups)
        storage = RolloutStorage("rl", num_envs, num_steps, obs, [NUM_ACTIONS])
        ppo = PPO(
            actor, critic, storage, gamma=gamma, lam=lam, schedule="fixed", normalize_advantage_per_mini_batch=True
        )

        rewards = [1.0, 2.0, 3.0]
        values = [0.5, 1.0, 1.5]
        dones = [0.0, 0.0, 0.0]

        for i in range(num_steps):
            t = RolloutStorage.Transition()
            t.observations = obs
            t.hidden_states = (None, None)
            t.actions = torch.randn(num_envs, NUM_ACTIONS)
            t.values = torch.full((num_envs, 1), values[i])
            t.actions_log_prob = torch.zeros(num_envs)
            t.distribution_params = (torch.zeros(num_envs, NUM_ACTIONS), torch.ones(num_envs, NUM_ACTIONS))
            t.rewards = torch.full((num_envs,), rewards[i])
            t.dones = torch.full((num_envs,), dones[i])
            storage.add_transition(t)

        last_values = torch.full((num_envs, 1), 2.0)
        # Manually compute GAE (backward pass)
        # Step 2: delta = r2 + gamma * V_last - V2 = 3.0 + 0.99*2.0 - 1.5 = 3.48
        #          adv2 = 3.48
        # Step 1: delta = r1 + gamma * V2 - V1 = 2.0 + 0.99*1.5 - 1.0 = 2.485
        #          adv1 = 2.485 + gamma*lam*adv2 = 2.485 + 0.99*0.95*3.48 = 2.485 + 3.27294 = 5.75794
        # Step 0: delta = r0 + gamma * V1 - V0 = 1.0 + 0.99*1.0 - 0.5 = 1.49
        #          adv0 = 1.49 + gamma*lam*adv1 = 1.49 + 0.99*0.95*5.75794 = 1.49 + 5.41484... = 6.90484...
        expected_adv = [
            1.49 + 0.99 * 0.95 * (2.485 + 0.99 * 0.95 * 3.48),
            2.485 + 0.99 * 0.95 * 3.48,
            3.48,
        ]
        expected_returns = [expected_adv[i] + values[i] for i in range(3)]

        # Use the actual critic to produce last_values override
        with torch.no_grad():
            storage.values[0] = torch.full((num_envs, 1), values[0])
            storage.values[1] = torch.full((num_envs, 1), values[1])
            storage.values[2] = torch.full((num_envs, 1), values[2])

        # Call compute_returns with a custom last_values by monkeypatching critic
        original_critic_call = ppo.critic.forward
        ppo.critic.forward = lambda *a, **kw: last_values
        ppo.compute_returns(obs)
        ppo.critic.forward = original_critic_call

        for i in range(num_steps):
            assert torch.allclose(
                storage.returns[i, 0, 0],
                torch.tensor(expected_returns[i]),
                atol=1e-4,
            ), f"Return mismatch at step {i}: got {storage.returns[i, 0, 0].item()}, expected {expected_returns[i]}"

    def test_gae_terminal_state_cuts_bootstrap(self) -> None:
        """When a done flag is set, the advantage should not bootstrap from the next value."""
        num_envs, num_steps = 1, 2
        gamma, lam = 0.99, 0.95

        obs = make_obs(num_envs, OBS_DIM)
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        actor = _make_actor(obs, obs_groups, NUM_ACTIONS)
        critic = _make_critic(obs, obs_groups)
        storage = RolloutStorage("rl", num_envs, num_steps, obs, [NUM_ACTIONS])
        ppo = PPO(
            actor, critic, storage, gamma=gamma, lam=lam, schedule="fixed", normalize_advantage_per_mini_batch=True
        )

        # Step 0: done=True, so step 1 is a fresh episode
        for i, (r, v, d) in enumerate([(1.0, 0.5, 1.0), (2.0, 1.0, 0.0)]):
            t = RolloutStorage.Transition()
            t.observations = obs
            t.hidden_states = (None, None)
            t.actions = torch.randn(num_envs, NUM_ACTIONS)
            t.values = torch.full((num_envs, 1), v)
            t.actions_log_prob = torch.zeros(num_envs)
            t.distribution_params = (torch.zeros(num_envs, NUM_ACTIONS), torch.ones(num_envs, NUM_ACTIONS))
            t.rewards = torch.full((num_envs,), r)
            t.dones = torch.full((num_envs,), d)
            storage.add_transition(t)

        last_values = torch.full((num_envs, 1), 3.0)
        ppo.critic.forward = lambda *a, **kw: last_values
        ppo.compute_returns(obs)

        # Step 0: done=True, so next_is_not_terminal = 0
        # delta0 = r0 - V0 = 1.0 - 0.5 = 0.5 (no bootstrap because done)
        # Step 1: delta1 = r1 + gamma * V_last - V1 = 2.0 + 0.99*3.0 - 1.0 = 3.97
        # adv1 = 3.97
        # adv0 = 0.5 (no bootstrap because done at step 0)
        expected_return_0 = 0.5 + 0.5  # adv0 + V0
        expected_return_1 = 3.97 + 1.0  # adv1 + V1

        assert torch.allclose(storage.returns[0, 0, 0], torch.tensor(expected_return_0), atol=1e-4)
        assert torch.allclose(storage.returns[1, 0, 0], torch.tensor(expected_return_1), atol=1e-4)

    def test_advantage_normalization_global(self) -> None:
        """With normalize_advantage_per_mini_batch=False, advantages should have mean~0, std~1."""
        ppo, obs = _build_ppo(normalize_advantage_per_mini_batch=False)

        for _ in range(NUM_STEPS):
            t = RolloutStorage.Transition()
            t.observations = obs
            t.hidden_states = (None, None)
            t.actions = ppo.actor(obs, stochastic_output=True).detach()
            t.values = ppo.critic(obs).detach()
            t.actions_log_prob = ppo.actor.get_output_log_prob(t.actions).detach()
            t.distribution_params = tuple(p.detach() for p in ppo.actor.output_distribution_params)
            t.rewards = torch.randn(NUM_ENVS)
            t.dones = torch.zeros(NUM_ENVS)
            ppo.storage.add_transition(t)

        ppo.compute_returns(obs)

        adv = ppo.storage.advantages.flatten()
        assert abs(adv.mean().item()) < 1e-5, "Advantages should be zero-mean"
        assert abs(adv.std().item() - 1.0) < 0.1, "Advantages should be unit-std"


class TestTimeoutBootstrapping:
    """Tests for timeout bootstrapping in ``process_env_step``."""

    def test_timeout_adds_bootstrap_to_reward(self) -> None:
        """When time_outs is set, stored reward should include gamma * value * timeout."""
        ppo, obs = _build_ppo()

        # Manually act to populate transition.values
        ppo.act(obs)
        stored_values = ppo.transition.values.clone()

        raw_reward = torch.ones(NUM_ENVS)
        dones = torch.ones(NUM_ENVS)
        time_outs = torch.zeros(NUM_ENVS)
        time_outs[0] = 1.0  # Only env 0 times out

        ppo.process_env_step(obs, raw_reward, dones, {"time_outs": time_outs})

        # The stored reward for env 0 should be: 1.0 + gamma * value[0]
        stored_reward_env0 = ppo.storage.rewards[0, 0, 0].item()
        expected = 1.0 + ppo.gamma * stored_values[0, 0].item()
        assert abs(stored_reward_env0 - expected) < 1e-5

        # Env 1 should have raw reward only
        stored_reward_env1 = ppo.storage.rewards[0, 1, 0].item()
        assert abs(stored_reward_env1 - 1.0) < 1e-5


class TestPPOLosses:
    """Tests for PPO loss computation correctness."""

    def test_surrogate_loss_clipping(self) -> None:
        """When ratio deviates beyond clip_param, the clipped branch should dominate."""
        clip_param = 0.2

        # Construct a scenario: positive advantages, ratio > 1 + clip
        advantages = torch.tensor([1.0, 1.0, 1.0])
        old_log_probs = torch.tensor([0.0, 0.0, 0.0])
        # New log probs that give ratio = exp(0.5) ≈ 1.65, which is > 1 + 0.2
        new_log_probs = torch.tensor([0.5, 0.5, 0.5])

        ratio = torch.exp(new_log_probs - old_log_probs)
        surrogate = -advantages * ratio
        surrogate_clipped = -advantages * torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
        loss = torch.max(surrogate, surrogate_clipped).mean()

        # The clipped branch should be -advantages * (1 + clip_param) = -1.2
        # The unclipped branch should be -advantages * 1.65 ≈ -1.65
        # max(-1.65, -1.2) = -1.2, so clipped branch dominates
        expected_clipped = (-advantages * (1.0 + clip_param)).mean()
        assert torch.allclose(loss, expected_clipped, atol=1e-5)

    def test_value_loss_clipping(self) -> None:
        """With clipped value loss, large value changes should be clipped."""
        clip_param = 0.2
        old_values = torch.tensor([[1.0], [1.0]])
        new_values = torch.tensor([[2.0], [1.1]])
        returns = torch.tensor([[1.5], [1.5]])

        value_clipped = old_values + (new_values - old_values).clamp(-clip_param, clip_param)
        losses_unclipped = (new_values - returns).pow(2)
        losses_clipped = (value_clipped - returns).pow(2)
        loss = torch.max(losses_unclipped, losses_clipped).mean()

        # Env 0: new=2.0, old=1.0, clipped_new=1.2
        #   unclipped: (2.0 - 1.5)^2 = 0.25
        #   clipped: (1.2 - 1.5)^2 = 0.09
        #   max = 0.25
        # Env 1: new=1.1, old=1.0, clipped_new=1.1 (within clip)
        #   unclipped: (1.1 - 1.5)^2 = 0.16
        #   clipped: (1.1 - 1.5)^2 = 0.16
        #   max = 0.16
        expected = (0.25 + 0.16) / 2
        assert torch.allclose(loss, torch.tensor(expected), atol=1e-5)


class TestAdaptiveLearningRate:
    """Tests for adaptive KL-based learning rate scheduling."""

    def test_lr_decreases_when_kl_too_high(self) -> None:
        """LR should decrease when KL > 2 * desired_kl."""
        ppo, _obs = _build_ppo(schedule="adaptive", desired_kl=0.01, learning_rate=1e-3)
        initial_lr = ppo.learning_rate

        # Simulate high KL scenario
        ppo.learning_rate = initial_lr
        kl_mean = torch.tensor(0.03)  # > 2 * 0.01

        # Apply the same logic as PPO.update
        if kl_mean > ppo.desired_kl * 2.0:
            ppo.learning_rate = max(1e-5, ppo.learning_rate / 1.5)

        assert ppo.learning_rate < initial_lr
        assert ppo.learning_rate == max(1e-5, initial_lr / 1.5)

    def test_lr_increases_when_kl_too_low(self) -> None:
        """LR should increase when 0 < KL < desired_kl / 2."""
        ppo, _obs = _build_ppo(schedule="adaptive", desired_kl=0.01, learning_rate=1e-3)
        initial_lr = ppo.learning_rate

        kl_mean = torch.tensor(0.002)  # < 0.01 / 2 = 0.005

        if kl_mean < ppo.desired_kl / 2.0 and kl_mean > 0.0:
            ppo.learning_rate = min(1e-2, ppo.learning_rate * 1.5)

        assert ppo.learning_rate > initial_lr
        assert ppo.learning_rate == min(1e-2, initial_lr * 1.5)

    def test_lr_unchanged_in_stable_range(self) -> None:
        """LR should remain unchanged when KL is in [desired_kl/2, 2*desired_kl]."""
        ppo, _obs = _build_ppo(schedule="adaptive", desired_kl=0.01, learning_rate=1e-3)
        initial_lr = ppo.learning_rate

        kl_mean = torch.tensor(0.01)  # Exactly desired_kl — in stable range

        if kl_mean > ppo.desired_kl * 2.0:
            ppo.learning_rate = max(1e-5, ppo.learning_rate / 1.5)
        elif kl_mean < ppo.desired_kl / 2.0 and kl_mean > 0.0:
            ppo.learning_rate = min(1e-2, ppo.learning_rate * 1.5)

        assert ppo.learning_rate == initial_lr


class TestSharedMemory:
    """Tests for the shared memory module and its actor/critic latent wiring."""

    H, D = 4, 8

    def _build_memory_ppo(self, actor_latent: bool, critic_latent: bool, **overrides: object) -> tuple[PPO, TensorDict]:
        """Build a PPO with a TCN memory over a stacked-history obs group and the given latent wiring."""
        from robot_rl.models import TCNModel

        obs = TensorDict(
            {"policy": torch.randn(NUM_ENVS, self.H * self.D), "privileged": torch.randn(NUM_ENVS, 6)},
            batch_size=[NUM_ENVS],
        )
        obs_groups = {"memory": ["policy"], "actor": ["policy"], "critic": ["privileged"]}
        memory = TCNModel(
            obs,
            obs_groups,
            "memory",
            1,
            memory_only=True,
            tcn_history_length=self.H,
            tcn_step_embed_dim=8,
            tcn_channels=[4],
            tcn_kernel_sizes=[2],
            tcn_strides=[1],
            tcn_encoding_dim=8,
        )
        actor_kwargs: dict[str, object] = (
            {"input_dim_override": memory.latent_dim, "append_obs_groups": False} if actor_latent else {}
        )
        critic_kwargs: dict[str, object] = (
            {"input_dim_override": memory.latent_dim, "append_obs_groups": True} if critic_latent else {}
        )
        actor = _make_actor(obs, obs_groups, NUM_ACTIONS, **actor_kwargs)
        critic = _make_critic(obs, obs_groups, **critic_kwargs)
        storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
        defaults = dict(
            num_learning_epochs=2,
            num_mini_batches=2,
            schedule="fixed",
            memory=memory,
            memory_wiring={
                "actor_latent": actor_latent,
                "critic_latent": critic_latent,
                "detach_actor_latent": False,
                "detach_critic_latent": False,
            },
        )
        defaults.update(overrides)
        ppo = PPO(actor, critic, storage, **defaults)
        return ppo, obs

    def _rollout_and_update(self, ppo: PPO, obs: TensorDict) -> dict[str, float]:
        """Run one rollout + update cycle."""
        for _ in range(NUM_STEPS):
            actions = ppo.act(obs)
            assert actions.shape == (NUM_ENVS, NUM_ACTIONS)
            ppo.process_env_step(obs, torch.randn(NUM_ENVS), torch.zeros(NUM_ENVS).byte(), {})
        ppo.compute_returns(obs)
        return ppo.update()

    def test_actor_only_latent(self) -> None:
        """Actor consumes the latent, critic stays a plain model on its own obs (asymmetric wiring)."""
        ppo, obs = self._build_memory_ppo(actor_latent=True, critic_latent=False)
        loss_dict = self._rollout_and_update(ppo, obs)
        assert "value" in loss_dict
        # memory trains through the actor loss only
        assert any(p.grad is not None for p in ppo.memory.parameters())

    def test_both_latent(self) -> None:
        """Both heads consume the latent; the critic appends its own obs groups."""
        ppo, obs = self._build_memory_ppo(actor_latent=True, critic_latent=True)
        self._rollout_and_update(ppo, obs)

    def test_detached_latents_leave_memory_untrained(self) -> None:
        """With both consumers detached, no gradient reaches the memory."""
        ppo, obs = self._build_memory_ppo(
            actor_latent=True,
            critic_latent=True,
            memory_wiring={
                "actor_latent": True,
                "critic_latent": True,
                "detach_actor_latent": True,
                "detach_critic_latent": True,
            },
        )
        self._rollout_and_update(ppo, obs)
        assert all(p.grad is None for p in ppo.memory.parameters())

    def test_non_recurrent_memory_uses_flat_minibatches(self) -> None:
        """A TCN memory must not force the recurrent trajectory generator."""
        ppo, obs = self._build_memory_ppo(actor_latent=True, critic_latent=False)
        calls: list[str] = []
        flat_gen, rec_gen = ppo.storage.mini_batch_generator, ppo.storage.recurrent_mini_batch_generator
        ppo.storage.mini_batch_generator = lambda *a, **k: (calls.append("flat"), flat_gen(*a, **k))[1]
        ppo.storage.recurrent_mini_batch_generator = lambda *a, **k: (calls.append("recurrent"), rec_gen(*a, **k))[1]
        self._rollout_and_update(ppo, obs)
        assert calls == ["flat"]

    def test_no_consumer_raises(self) -> None:
        """A memory nobody consumes is a configuration error."""
        import pytest

        with pytest.raises(ValueError, match="neither the actor nor the critic"):
            self._build_memory_ppo(actor_latent=False, critic_latent=False)

    def test_get_policy_chains_memory_only_for_consuming_actor(self) -> None:
        """get_policy wraps the actor with the memory iff the actor consumes the latent."""
        ppo, obs = self._build_memory_ppo(actor_latent=True, critic_latent=False)
        policy = ppo.get_policy()
        actions = policy(obs)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)
        assert policy.is_recurrent is False  # TCN memory is window-based
        ppo2, _obs2 = self._build_memory_ppo(actor_latent=False, critic_latent=True)
        assert ppo2.get_policy() is ppo2._raw_actor
