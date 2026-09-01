# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for recurrent sequence sampling from the replay buffer."""

import torch
from tensordict import TensorDict

from robot_rl.storage.replay_buffer import ReplayBuffer

NUM_ENVS, CAPACITY_PER_ENV, HIDDEN, LAYERS = 4, 20, 8, 1


def _make_buffer(batch_size: int = 64, capacity_per_env: int = CAPACITY_PER_ENV) -> ReplayBuffer:
    obs = TensorDict({"policy": torch.zeros(NUM_ENVS, 3)}, batch_size=NUM_ENVS)
    return ReplayBuffer(
        NUM_ENVS,
        capacity_per_env,
        obs,
        [2],
        z_dim=0,
        batch_size=batch_size,
        device="cpu",
        keep_terminal=True,
        hidden_dim=HIDDEN,
        hidden_layers=LAYERS,
    )


def _fill(buffer: ReplayBuffer, num_steps: int, done_at: set[int] = frozenset()) -> None:
    """Store steps whose obs encodes (env id, step) and whose hidden encodes the step it precedes."""
    for step in range(num_steps):
        transition = ReplayBuffer.Transition()
        obs = torch.stack(
            [torch.arange(NUM_ENVS).float(), torch.full((NUM_ENVS,), float(step)), torch.zeros(NUM_ENVS)], dim=1
        )
        transition.observations = TensorDict({"policy": obs}, batch_size=NUM_ENVS)
        transition.next_observations = TensorDict({"policy": obs + 0.5}, batch_size=NUM_ENVS)
        transition.actions = torch.full((NUM_ENVS, 2), float(step))
        transition.rewards = torch.full((NUM_ENVS,), float(step))
        transition.context = torch.zeros(NUM_ENVS, 0)
        transition.next_terminated = torch.zeros(NUM_ENVS).byte()
        transition.dones = torch.ones(NUM_ENVS) if step in done_at else torch.zeros(NUM_ENVS)
        transition.hidden_state = torch.full((LAYERS, NUM_ENVS, HIDDEN), float(step))
        buffer.add_transitions(transition)


def test_sequences_are_single_env_and_time_ordered() -> None:
    """A sampled window comes from one env and is consecutive in time."""
    buffer = _make_buffer()
    _fill(buffer, 18)
    batch = buffer.sample_sequences(seq_len=4, burn_in=2)
    assert batch is not None
    assert batch.actions.shape == (6, 64, 2)
    assert batch.masks.shape == (6, 64)
    assert batch.burn_in == 2

    env_ids = batch.observations["policy"][..., 0]
    steps = batch.observations["policy"][..., 1]
    assert torch.all(env_ids == env_ids[0:1]), "a window must come from one environment"
    assert torch.all(steps[1:] - steps[:-1] == 1), "a window must be consecutive in time"


def test_init_hidden_matches_window_start() -> None:
    """The returned recurrent state is the one stored at the window's first step."""
    buffer = _make_buffer()
    _fill(buffer, 18)
    batch = buffer.sample_sequences(seq_len=4, burn_in=2)
    assert batch is not None and batch.init_hidden is not None
    steps = batch.observations["policy"][..., 1]
    assert torch.allclose(batch.init_hidden[0, :, 0], steps[0])


def test_mask_zeroes_after_episode_end() -> None:
    """Steps after the window's first episode end are masked out."""
    done_at = {5, 12}
    buffer = _make_buffer()
    _fill(buffer, 18, done_at=done_at)
    batch = buffer.sample_sequences(seq_len=4, burn_in=2)
    assert batch is not None
    steps = batch.observations["policy"][..., 1]
    for col in range(steps.shape[1]):
        ended = False
        for row, step in enumerate(steps[:, col].tolist()):
            assert batch.masks[row, col] == (0.0 if ended else 1.0)
            ended = ended or step in done_at


def test_wrapped_buffer_never_spans_the_write_head() -> None:
    """After wrapping, windows must stay in still-live data and remain consecutive across the wrap."""
    capacity_per_env, total = 10, 37
    buffer = _make_buffer(batch_size=256, capacity_per_env=capacity_per_env)
    _fill(buffer, total)
    assert buffer._is_full

    batch = buffer.sample_sequences(seq_len=4, burn_in=2)
    assert batch is not None
    steps = batch.observations["policy"][..., 1]
    assert torch.all(steps[1:] - steps[:-1] == 1)
    assert steps.min() >= total - capacity_per_env, "read a step that had been overwritten"


def test_returns_none_when_no_window_fits() -> None:
    """Sampling yields nothing until a full window exists."""
    buffer = _make_buffer()
    _fill(buffer, 2)
    assert buffer.sample_sequences(seq_len=4, burn_in=2) is None
