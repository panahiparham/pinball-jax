"""Tests ensuring the Pinball environment adheres to the GymEnv protocol."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from pinball_jax.gym_env import GymEnv
from pinball_jax.pinball import (
    NUM_ACTIONS,
    OBSERVATION_SHAPE,
    Pinball,
    PinballParams,
    PinballState,
)


@pytest.fixture
def env() -> Pinball:
    return Pinball()


@pytest.fixture
def key() -> jax.Array:
    return jax.random.PRNGKey(0)


def test_pinball_conforms_to_gym_env_protocol(env: Pinball) -> None:
    assert isinstance(env, GymEnv)


def test_observation_space(env: Pinball) -> None:
    obs_space = env.observation_space()
    assert obs_space.shape == OBSERVATION_SHAPE
    assert obs_space.dtype == jnp.float32


def test_action_space(env: Pinball) -> None:
    action_space = env.action_space()
    assert action_space.n == NUM_ACTIONS


def test_reset_returns_zero_observation_and_initial_state(env: Pinball, key: jax.Array) -> None:
    obs, state = env.reset(key)

    assert obs.shape == OBSERVATION_SHAPE
    assert obs.dtype == jnp.float32
    assert jnp.array_equal(obs, jnp.zeros(OBSERVATION_SHAPE, dtype=jnp.float32))
    assert isinstance(state, PinballState)
    assert state.timestep == 0


@pytest.mark.parametrize("action", range(NUM_ACTIONS))
def test_step_returns_expected_tuple(env: Pinball, key: jax.Array, action: int) -> None:
    _, state = env.reset(key)
    obs, next_state, reward, terminated, truncated, info = env.step(key, state, action)

    assert obs.shape == OBSERVATION_SHAPE
    assert obs.dtype == jnp.float32
    assert jnp.array_equal(obs, jnp.zeros(OBSERVATION_SHAPE, dtype=jnp.float32))
    assert isinstance(next_state, PinballState)
    assert next_state.timestep == state.timestep + 1
    assert reward == -1.0
    assert terminated == jnp.asarray(False)
    assert truncated == jnp.asarray(False)
    assert info == {}


def test_terminated_is_always_false(env: Pinball, key: jax.Array) -> None:
    _, state = env.reset(key)
    params = PinballParams(max_steps_in_episode=5)

    for _ in range(10):
        _, state, _, terminated, _, _ = env.step(key, state, 0, params)
        assert terminated == jnp.asarray(False)


def test_truncates_at_max_steps_in_episode(env: Pinball, key: jax.Array) -> None:
    max_steps = 3
    _, state = env.reset(key)
    params = PinballParams(max_steps_in_episode=max_steps)

    for expected_timestep in range(1, max_steps):
        _, state, _, _, truncated, _ = env.step(key, state, 0, params)
        assert state.timestep == expected_timestep
        assert truncated == jnp.asarray(False)

    _, state, _, _, truncated, _ = env.step(key, state, 0, params)
    assert state.timestep == max_steps
    assert truncated == jnp.asarray(True)
