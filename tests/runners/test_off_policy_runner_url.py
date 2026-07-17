# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locks the OffPolicyRunner's URL-path loop cadence (seed phase, update gate, eval interval).

Uses a stub URL algorithm (has ``expert_buffer``) so the FbCpr control flow -- ``num_steps_per_env``
env steps per iteration, ``num_agent_updates`` updates every iteration after the seed phase, eval every
``eval_interval`` iterations -- is pinned without needing Isaac Sim or motion data.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import Any

from robot_rl.runners import OffPolicyRunner

NUM_ENVS, OBS_DIM, NUM_ACTIONS = 4, 6, 3


class DummyUrlEnv:
    """Minimal URL VecEnv: adds reset/set_expert_buffer/train_mode over the plain dummy env."""

    def __init__(self, device: str = "cpu") -> None:  # noqa: D107
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.device = device
        self.cfg = {}
        self.expert_buffer = None

    def get_observations(self) -> TensorDict:  # noqa: D102
        return TensorDict({"policy": torch.randn(self.num_envs, OBS_DIM)}, batch_size=[self.num_envs])

    def reset(self) -> tuple[TensorDict, dict]:  # noqa: D102
        return self.get_observations(), {}

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:  # noqa: D102
        obs = self.get_observations()
        rewards = torch.randn(self.num_envs)
        dones = (torch.rand(self.num_envs) < 0.05).float()
        return obs, rewards, dones, {"time_outs": torch.zeros(self.num_envs)}

    def set_expert_buffer(self, buffer: Any) -> None:  # noqa: D102
        self.expert_buffer = buffer

    def train_mode(self) -> None:  # noqa: D102
        pass

    @property
    def unwrapped(self) -> DummyUrlEnv:  # noqa: D102
        return self


class FakeUrlAlg:
    """Stub URL algorithm: records the runner's call sequence; has an ``expert_buffer`` (URL dispatch)."""

    def __init__(self, num_actions: int, device: str) -> None:  # noqa: D107
        self.expert_buffer = object()
        self.device = device
        self.num_actions = num_actions
        self.calls: list[str] = []

    @staticmethod
    def construct_algorithm(  # noqa: D102
        obs: TensorDict, env: DummyUrlEnv, cfg: dict, device: str, inference: bool = False
    ) -> FakeUrlAlg:
        return FakeUrlAlg(env.num_actions, device)

    def act(self, obs: TensorDict) -> torch.Tensor:  # noqa: D102
        self.calls.append("act")
        return torch.zeros(obs.batch_size[0], self.num_actions)

    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:  # noqa: D102
        self.calls.append("step")

    def compute_gammas(self) -> None:  # noqa: D102
        self.calls.append("gammas")

    def update(self) -> tuple[dict, dict]:  # noqa: D102
        self.calls.append("update")
        return {"loss": 0.0}, {}

    def eval(self, env: DummyUrlEnv) -> list[dict]:  # noqa: D102
        self.calls.append("eval")
        return [{"emd": torch.tensor(0.0)}]

    def reset_rollout_state(self) -> None:  # noqa: D102
        self.calls.append("reset_rollout")

    def train_mode(self) -> None:  # noqa: D102
        pass


def _make_cfg() -> dict:
    return {
        "num_steps_per_env": 2,
        "num_agent_updates": 3,
        "num_seed_steps_per_env": 2,
        "eval_interval": 4,
        "log_interval": 1,
        "save_interval": 100,
        "clip_actions": 1.0,
        "check_for_nan": True,
        "obs_groups": {"actor": ["policy"]},
        "algorithm": {"class_name": FakeUrlAlg, "rnd_cfg": None},
    }


class TestOffPolicyRunnerUrlCadence:
    """The unified loop must reproduce the FbCpr cadence for URL algorithms."""

    def test_loop_cadence(self) -> None:
        """8 iterations: num_steps_per_env acts/iter; updates every iteration past the seed phase."""
        env = DummyUrlEnv()
        runner = OffPolicyRunner(env, _make_cfg(), log_dir=None, device="cpu")
        runner.learn(num_learning_iterations=8)
        calls = runner.alg.calls

        # num_steps_per_env (2) env steps per iteration
        assert calls.count("act") == 16
        assert calls.count("step") == 16
        # Update gate: it > start_it + num_seed_steps_per_env (2) -> iterations 3..7 -> 5 x num_agent_updates (3)
        assert calls.count("gammas") == 5
        assert calls.count("update") == 15
        # Eval every eval_interval (4) iterations -> it 0 and 4; env reset follows each eval
        assert calls.count("eval") == 2
        assert calls.count("reset_rollout") == 2
        # Expert buffer was attached to the env before training
        assert env.expert_buffer is runner.alg.expert_buffer

    def test_updates_follow_collection_within_iteration(self) -> None:
        """Within an update iteration the order is act/step -> gammas -> updates (external loop shape)."""
        runner = OffPolicyRunner(DummyUrlEnv(), _make_cfg(), log_dir=None, device="cpu")
        runner.learn(num_learning_iterations=5)
        calls = runner.alg.calls
        first_update = calls.index("gammas")
        assert calls[first_update - 1] == "step"  # collection precedes the update phase
        assert calls[first_update + 1 : first_update + 4] == ["update", "update", "update"]
