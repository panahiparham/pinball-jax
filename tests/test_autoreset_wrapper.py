"""Tests for pinball_jax.wrappers.AutoresetWrapper."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from pinball_jax.pinball import NUM_ACTIONS, Pinball, PinballParams
from pinball_jax.wrappers import AutoresetMode, AutoresetState, AutoresetWrapper


@pytest.fixture
def key() -> jax.Array:
    return jax.random.PRNGKey(0)


def test_disabled_mode_reset_matches_wrapped_env(key: jax.Array) -> None:
    """DISABLED-mode reset returns the same observation as the wrapped env."""
    env = Pinball("box")
    wrapped = AutoresetWrapper(env, mode=AutoresetMode.DISABLED)

    obs, state = wrapped.reset(key)
    inner_obs, _ = env.reset(key)

    assert jnp.array_equal(obs, inner_obs)
    assert isinstance(state, AutoresetState)
    assert not bool(state.pending)


def test_disabled_mode_trajectory_matches_wrapped_env(key: jax.Array) -> None:
    """DISABLED-mode step-by-step transitions match the wrapped env exactly.

    This is the existing manual-reset behaviour: wrapping in DISABLED mode
    must not change any observation, reward, or done flag.
    """
    env = Pinball("box")
    wrapped = AutoresetWrapper(env, mode=AutoresetMode.DISABLED)
    params = PinballParams(max_steps_in_episode=5)

    _, state = wrapped.reset(key)
    _, inner_state = env.reset(key)

    for action in range(NUM_ACTIONS):
        obs, state, reward, terminated, truncated, info = wrapped.step(
            key, state, action, params
        )
        inner_obs, inner_state, inner_reward, inner_term, inner_trunc, inner_info = (
            env.step(key, inner_state, action, params)
        )

        assert jnp.array_equal(obs, inner_obs)
        assert reward == inner_reward
        assert terminated == inner_term
        assert truncated == inner_trunc
        assert info == inner_info
        # DISABLED mode never reports a pending reset; the caller must reset.
        assert not bool(state.pending)


def test_disabled_mode_truncates_like_wrapped_env(key: jax.Array) -> None:
    """DISABLED mode surfaces the wrapped env's own truncation, not an autoreset."""
    env = Pinball("box")
    wrapped = AutoresetWrapper(env, mode=AutoresetMode.DISABLED)
    params = PinballParams(max_steps_in_episode=2)

    _, state = wrapped.reset(key)
    for _ in range(2):
        _, state, _, _, truncated, _ = wrapped.step(key, state, NUM_ACTIONS - 1, params)

    assert bool(truncated)
    assert not bool(state.pending)
