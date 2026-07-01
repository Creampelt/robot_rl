# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Characterization tests for the off-policy ReplayBuffer (locks FbCpr-facing behavior on CPU)."""

import torch
from tensordict import TensorDict

from robot_rl.storage.replay_buffer import ReplayBuffer

OBS_DIM, ACT_DIM, Z_DIM = 5, 3, 4


def _make_buffer(num_envs: int = 6, capacity_per_env: int = 4, batch_size: int = 8) -> ReplayBuffer:
    obs = TensorDict({"policy": torch.zeros(num_envs, OBS_DIM)}, batch_size=num_envs)
    return ReplayBuffer(
        num_envs=num_envs,
        capacity_per_env=capacity_per_env,
        obs=obs,
        actions_shape=(ACT_DIM,),
        z_dim=Z_DIM,
        batch_size=batch_size,
        device="cpu",
    )


def _make_transition(num_envs: int, dones: torch.Tensor) -> ReplayBuffer.Transition:
    tr = ReplayBuffer.Transition()
    tr.observations = TensorDict({"policy": torch.randn(num_envs, OBS_DIM)}, batch_size=num_envs)
    tr.next_observations = TensorDict({"policy": torch.randn(num_envs, OBS_DIM)}, batch_size=num_envs)
    tr.actions = torch.randn(num_envs, ACT_DIM)
    tr.rewards = torch.randn(num_envs)
    tr.context = torch.randn(num_envs, Z_DIM)
    tr.dones = dones
    tr.next_terminated = dones.byte()
    return tr


class TestReplayBuffer:
    """Locks the current add/sample/filter/wrap semantics."""

    def test_add_filters_done_transitions(self) -> None:
        """add_transitions stores only non-done transitions (post-reset next_obs would be invalid)."""
        buf = _make_buffer(num_envs=6)
        dones = torch.tensor([0, 1, 0, 1, 0, 0]).bool()  # 4 valid
        buf.add_transitions(_make_transition(6, dones))
        assert len(buf) == 4

    def test_add_none_dones_stores_all(self) -> None:
        """With dones=None, all transitions are stored."""
        buf = _make_buffer(num_envs=6)
        tr = _make_transition(6, torch.zeros(6).bool())
        tr.dones = None
        buf.add_transitions(tr)
        assert len(buf) == 6

    def test_len_and_circular_wrap(self) -> None:
        """The buffer fills to capacity then wraps, capping len() at capacity."""
        buf = _make_buffer(num_envs=6, capacity_per_env=2)  # capacity = 12
        for _ in range(5):  # 5 * 6 = 30 valid adds into capacity 12
            buf.add_transitions(_make_transition(6, torch.zeros(6).bool()))
        assert len(buf) == 12  # capped at capacity
        assert buf._is_full is True

    def test_sample_shapes(self) -> None:
        """sample_mini_batch returns a Batch with the configured batch size and correct field shapes."""
        buf = _make_buffer(num_envs=6, batch_size=8)
        buf.add_transitions(_make_transition(6, torch.zeros(6).bool()))
        batch = buf.sample_mini_batch(device="cpu")
        assert batch.observations["policy"].shape == (8, OBS_DIM)
        assert batch.next_observations["policy"].shape == (8, OBS_DIM)
        assert batch.actions.shape == (8, ACT_DIM)
        assert batch.rewards.shape == (8,)
        assert batch.context.shape == (8, Z_DIM)
        assert batch.gammas.shape == (8, 1)

    def test_stored_values_roundtrip(self) -> None:
        """A single all-valid add should be recoverable (values land in the buffer)."""
        buf = _make_buffer(num_envs=3, batch_size=64)
        tr = _make_transition(3, torch.zeros(3).bool())
        buf.add_transitions(tr)
        # All stored actions should be present among the first len() rows.
        assert torch.allclose(buf.actions[:3], tr.actions)
        assert torch.allclose(buf.context[:3], tr.context)
