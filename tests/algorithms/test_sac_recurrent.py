# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for SAC with a recurrent CNN actor (dummy VecEnv with an image group, no Isaac Sim)."""

import torch
from tensordict import TensorDict

from robot_rl.algorithms import SAC

OBS_DIM, ACT_DIM, NUM_ENVS = 6, 3, 8
IMG = (1, 8, 12)


class _DummyImageVecEnv:
    """A minimal VecEnv with a 1D actor group, a 2D image group, and a privileged critic group."""

    def __init__(self, num_envs: int = NUM_ENVS, num_actions: int = ACT_DIM, done_prob: float = 0.1) -> None:
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.done_prob = done_prob
        self.cfg = None

    def _obs(self) -> TensorDict:
        return TensorDict(
            {
                "policy": torch.randn(self.num_envs, OBS_DIM),
                "image": torch.rand(self.num_envs, *IMG),
                "critic": torch.randn(self.num_envs, OBS_DIM),
            },
            batch_size=self.num_envs,
        )

    def get_observations(self) -> TensorDict:
        return self._obs()

    def step(self, actions: torch.Tensor) -> tuple:
        next_obs = self._obs()
        rewards = torch.randn(self.num_envs)
        dones = torch.rand(self.num_envs) < self.done_prob
        extras = {"time_outs": torch.zeros(self.num_envs, dtype=torch.bool), "time_outs_obs": None}
        return next_obs, rewards, dones, extras


def _make_cfg(obs_normalization: bool = True) -> dict:
    return {
        "num_steps_per_env": 4,
        "obs_groups": {"actor": ["policy", "image"], "critic": ["critic"]},
        "actor": {
            "class_name": "CNNRNNModel",
            "hidden_dims": [32],
            "obs_normalization": obs_normalization,
            "rnn_type": "gru",
            "rnn_hidden_dim": 16,
            "rnn_num_layers": 1,
            "cnn_cfg": {
                "output_channels": [4, 4],
                "kernel_size": 3,
                "stride": 2,
                "activation": "elu",
                "global_pool": "avg",
            },
            "distribution_cfg": {"class_name": "SquashedTanhGaussianDistribution", "init_noise_std": 1.0},
        },
        "critic": {"class_name": "FuseModel", "embedding_dims": [16], "hidden_dims": [32]},
        "algorithm": {
            "class_name": "SAC",
            "replay_buffer_size": 512,
            "mini_batch_size": 8,
            "num_learning_epochs": 1,
            "num_mini_batches": 2,
            "actor_learning_rate": 1e-3,
            "critic_learning_rate": 1e-3,
            "alpha_learning_rate": 1e-3,
            "auto_alpha": True,
            "alpha": 0.1,
            "tau": 0.1,
            "gamma": 0.99,
            "target_entropy_scale": 1.0,
            "max_grad_norm": 1.0,
            "policy_frequency": 1,
            "n_steps": 1,
            "seq_len": 4,
            "burn_in": 2,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }


def _build(done_prob: float = 0.1, obs_normalization: bool = True) -> tuple[SAC, _DummyImageVecEnv]:
    env = _DummyImageVecEnv(done_prob=done_prob)
    alg = SAC.construct_algorithm(env.get_observations(), env, _make_cfg(obs_normalization), device="cpu")
    return alg, env


def _collect(alg: SAC, env: _DummyImageVecEnv, steps: int) -> list[torch.Tensor]:
    """Roll out and return the dones of every step."""
    obs = env.get_observations()
    all_dones = []
    for _ in range(steps):
        actions = alg.act(obs)
        next_obs, rewards, dones, extras = env.step(actions)
        alg.process_env_step(next_obs, rewards, dones, extras)
        all_dones.append(dones)
        obs = next_obs
    return all_dones


class TestRecurrentSAC:
    """Recurrent actor end to end: stored state, reset on done, windowed update."""

    def test_construct_stores_hidden(self) -> None:
        """A recurrent actor makes the buffer allocate hidden-state storage sized from the RNN."""
        alg, _ = _build()
        assert alg.recurrent
        assert alg.replay_buffer.hidden is not None
        assert alg.replay_buffer.hidden.shape[-1] == 16

    def test_stored_state_is_self_consistent(self) -> None:
        """Stored states chain: the state after obs_t equals the state stored before obs_t+1, and a done zeros it.

        Normalization is off so the recomputation sees the same features the rollout did; with it on, the
        running statistics move between the two and the comparison measures that drift instead.
        """
        torch.manual_seed(0)
        alg, env = _build(done_prob=0.5, obs_normalization=False)
        dones = _collect(alg, env, 6)
        buf = alg.replay_buffer
        n = env.num_envs
        for t in range(5):
            row_t, row_next = t * n, (t + 1) * n
            h_t = buf.hidden[row_t : row_t + n, 0].permute(1, 0, 2)  # (layers, N, H)
            obs_t = buf.observations[row_t : row_t + n]
            with torch.no_grad():
                _, h_after = alg.actor.encode_sequence(obs_t.unsqueeze(0), h_t)
            h_stored_next = buf.hidden[row_next : row_next + n, 0].permute(1, 0, 2)
            done = dones[t]
            assert torch.allclose(h_after[:, ~done], h_stored_next[:, ~done], atol=1e-6)
            if done.any():
                assert torch.all(h_stored_next[:, done] == 0)

    def test_update_runs_and_is_finite(self) -> None:
        """After enough contiguous data, the windowed update runs every loss and returns finite values."""
        torch.manual_seed(0)
        alg, env = _build()
        _collect(alg, env, 12)
        losses = alg.update()
        for key in ("critic_1", "critic_2", "actor", "alpha"):
            assert torch.isfinite(torch.tensor(losses[key])), key
        assert losses["critic_1"] > 0

    def test_update_before_window_exists_is_a_noop(self) -> None:
        """With fewer rows than burn_in + seq_len, no update is attempted and nothing crashes."""
        alg, env = _build()
        _collect(alg, env, 2)
        step_before = alg.update_step
        losses = alg.update()
        assert alg.update_step == step_before
        assert losses["critic_1"] == 0.0

    def test_sequence_act_matches_stepwise_rollout(self) -> None:
        """Encoding a window in one call equals stepping the rollout RNN through it one step at a time."""
        torch.manual_seed(0)
        alg, env = _build()
        obs_seq = [env.get_observations() for _ in range(5)]
        alg.actor.reset()
        with torch.no_grad():
            stepwise = [alg.actor.get_latent(obs) for obs in obs_seq]
            window = TensorDict.stack(obs_seq, dim=0)
            latent, _ = alg.actor.encode_sequence(window, None)
        assert torch.allclose(latent, torch.stack(stepwise), atol=1e-6)
