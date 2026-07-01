# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Characterization tests for the off-policy ReplayBuffer (locks FbCpr-facing behavior on CPU)."""

import torch
from tensordict import TensorDict

from robot_rl.storage.replay_buffer import ReplayBuffer

OBS_DIM, ACT_DIM, Z_DIM = 5, 3, 4


def _make_buffer(
    num_envs: int = 6, capacity_per_env: int = 4, batch_size: int = 8, keep_terminal: bool = False, z_dim: int = Z_DIM
) -> ReplayBuffer:
    obs = TensorDict({"policy": torch.zeros(num_envs, OBS_DIM)}, batch_size=num_envs)
    return ReplayBuffer(
        num_envs=num_envs,
        capacity_per_env=capacity_per_env,
        obs=obs,
        actions_shape=(ACT_DIM,),
        z_dim=z_dim,
        batch_size=batch_size,
        device="cpu",
        keep_terminal=keep_terminal,
    )


def _make_transition(num_envs: int, dones: torch.Tensor, z_dim: int = Z_DIM) -> ReplayBuffer.Transition:
    tr = ReplayBuffer.Transition()
    tr.observations = TensorDict({"policy": torch.randn(num_envs, OBS_DIM)}, batch_size=num_envs)
    tr.next_observations = TensorDict({"policy": torch.randn(num_envs, OBS_DIM)}, batch_size=num_envs)
    tr.actions = torch.randn(num_envs, ACT_DIM)
    tr.rewards = torch.randn(num_envs)
    tr.context = torch.randn(num_envs, z_dim)
    tr.dones = dones
    tr.next_terminated = dones.byte()
    return tr


def _nstep_buffer(
    num_envs: int = 4, capacity_per_env: int = 16, batch_size: int = 64, n_steps: int = 3, gamma: float = 0.5
) -> ReplayBuffer:
    obs = TensorDict({"policy": torch.zeros(num_envs, OBS_DIM)}, batch_size=num_envs)
    return ReplayBuffer(
        num_envs=num_envs,
        capacity_per_env=capacity_per_env,
        obs=obs,
        actions_shape=(ACT_DIM,),
        z_dim=0,
        batch_size=batch_size,
        device="cpu",
        keep_terminal=True,
        n_steps=n_steps,
        gamma=gamma,
    )


def _add_step(buf: ReplayBuffer, num_envs: int, reward: float, done: bool) -> None:
    tr = _make_transition(num_envs, torch.full((num_envs,), done).bool(), z_dim=0)
    tr.rewards = torch.full((num_envs,), reward)
    buf.add_transitions(tr)


class TestReplayBufferNStep:
    """n-step return aggregation (keep_terminal, row-aligned layout)."""

    def test_requires_keep_terminal(self) -> None:
        """n_steps > 1 without keep_terminal is rejected (n-step needs the per-env time sequence)."""
        obs = TensorDict({"policy": torch.zeros(2, OBS_DIM)}, batch_size=2)
        import pytest

        with pytest.raises(ValueError, match="keep_terminal"):
            ReplayBuffer(2, 8, obs, (ACT_DIM,), z_dim=0, batch_size=8, device="cpu", n_steps=3)

    def test_nstep_return_no_dones(self) -> None:
        """With reward=1 everywhere and no episode ends, every 3-step return is 1 + g + g^2 and horizon = n."""
        gamma, n = 0.5, 3
        buf = _nstep_buffer(n_steps=n, gamma=gamma)
        for _ in range(8):  # 8 rows -> valid windows exist (rows 0..5 with max_offset 2)
            _add_step(buf, 4, reward=1.0, done=False)
        batch = buf.sample_mini_batch(device="cpu")
        expected = 1.0 + gamma + gamma**2
        assert batch.effective_n_steps is not None
        assert torch.allclose(batch.rewards, torch.full_like(batch.rewards, expected), atol=1e-5)
        assert (batch.effective_n_steps.view(-1) == n).all()

    def test_nstep_stops_at_episode_end(self) -> None:
        """If every step is an episode end, the return collapses to the single-step reward (horizon = 1)."""
        buf = _nstep_buffer(n_steps=3, gamma=0.5)
        for _ in range(8):
            _add_step(buf, 4, reward=1.0, done=True)
        batch = buf.sample_mini_batch(device="cpu")
        assert torch.allclose(batch.rewards, torch.ones_like(batch.rewards), atol=1e-5)
        assert (batch.effective_n_steps.view(-1) == 1).all()

    def test_single_step_batch_has_no_effective_n(self) -> None:
        """The 1-step path leaves effective_n_steps unset (only n-step populates it)."""
        obs = TensorDict({"policy": torch.zeros(4, OBS_DIM)}, batch_size=4)
        buf = ReplayBuffer(4, 16, obs, (ACT_DIM,), z_dim=0, batch_size=32, device="cpu", keep_terminal=True)
        _add_step(buf, 4, reward=1.0, done=False)
        assert buf.sample_mini_batch(device="cpu").effective_n_steps is None


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
        assert batch.next_terminated.shape == (8, 1)

    def test_keep_terminal_stores_all(self) -> None:
        """With keep_terminal=True (SAC), done transitions are retained instead of dropped."""
        buf = _make_buffer(num_envs=6, keep_terminal=True, z_dim=0)
        dones = torch.tensor([0, 1, 0, 1, 1, 0]).bool()  # 3 done, but all kept
        buf.add_transitions(_make_transition(6, dones, z_dim=0))
        assert len(buf) == 6

    def test_keep_terminal_records_next_terminated(self) -> None:
        """The next_terminated flags are stored and recoverable when keeping terminals."""
        buf = _make_buffer(num_envs=4, keep_terminal=True, z_dim=0, batch_size=64)
        dones = torch.tensor([0, 1, 0, 1]).bool()
        buf.add_transitions(_make_transition(4, dones, z_dim=0))
        assert torch.equal(buf.next_terminated[:4].view(-1).bool(), dones)

    def test_stored_values_roundtrip(self) -> None:
        """A single all-valid add should be recoverable (values land in the buffer)."""
        buf = _make_buffer(num_envs=3, batch_size=64)
        tr = _make_transition(3, torch.zeros(3).bool())
        buf.add_transitions(tr)
        # All stored actions should be present among the first len() rows.
        assert torch.allclose(buf.actions[:3], tr.actions)
        assert torch.allclose(buf.context[:3], tr.context)
