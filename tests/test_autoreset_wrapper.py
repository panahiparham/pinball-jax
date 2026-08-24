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


def test_next_step_termination_then_dead_step_ignores_action(key: jax.Array) -> None:
    """Terminal step reports the true final obs; the next ignores its action."""
    env = Pinball("empty")
    params = PinballParams(max_steps_in_episode=10**9)

    for dead_action in range(2):  # prove the dead step's action is irrelevant
        wrapped = AutoresetWrapper(env, mode=AutoresetMode.NEXT_STEP)
        jstep = jax.jit(lambda s, a: wrapped.step(key, s, a, params))
        obs, state = wrapped.reset(key)
        target = jnp.asarray(env.target)

        terminated = False
        for _ in range(2000):
            action = 0 if float(state.inner.x) < target[0] else 3
            obs, state, reward, terminated, truncated, _ = jstep(state, action)
            if bool(terminated):
                break

        assert bool(terminated)
        # The terminal step reports the true boundary observation.
        dist = jnp.linalg.norm(obs[:2] - target)
        assert dist < env.target_rad
        assert bool(state.pending)

        obs2, state2, reward2, term2, trunc2, _ = wrapped.step(
            key, state, dead_action, params
        )
        assert reward2 == 0.0
        assert not bool(term2)
        assert not bool(trunc2)
        assert not bool(state2.pending)


def test_next_step_truncation_then_dead_step_ignores_action(key: jax.Array) -> None:
    """Truncating step keeps the true final obs; the next ignores its action."""
    env = Pinball("box")
    max_steps = 3
    params = PinballParams(max_steps_in_episode=max_steps)

    for dead_action in range(2):  # prove the dead step's action is irrelevant
        wrapped = AutoresetWrapper(env, mode=AutoresetMode.NEXT_STEP)
        _, state = wrapped.reset(key)

        terminated = truncated = False
        for _ in range(max_steps):
            _, state, _, terminated, truncated, _ = wrapped.step(
                key, state, NUM_ACTIONS - 1, params
            )

        assert bool(truncated)
        assert not bool(terminated)
        assert bool(state.pending)

        _, state2, reward2, term2, trunc2, _ = wrapped.step(
            key, state, dead_action, params
        )
        assert reward2 == 0.0
        assert not bool(term2)
        assert not bool(trunc2)
        assert not bool(state2.pending)


def test_next_step_dead_step_obs_is_a_real_start_point(key: jax.Array) -> None:
    """The dead step's observation matches one of the config's start points."""
    env = Pinball("box")
    wrapped = AutoresetWrapper(env, mode=AutoresetMode.NEXT_STEP)
    params = PinballParams(max_steps_in_episode=2)

    _, state = wrapped.reset(key)
    for _ in range(2):
        _, state, _, _, truncated, _ = wrapped.step(key, state, NUM_ACTIONS - 1, params)
    assert bool(truncated)

    obs, state, reward, terminated, truncated, _ = wrapped.step(key, state, 0, params)

    assert jnp.all(obs[2:] == 0.0)  # at rest, like any fresh reset
    matches_a_start = jnp.any(jnp.all(jnp.isclose(env.start_pts, obs[:2]), axis=-1))
    assert bool(matches_a_start)


def test_next_step_pending_clears_after_dead_step(key: jax.Array) -> None:
    """The step after the dead step behaves like a normal new-episode step."""
    env = Pinball("box")
    wrapped = AutoresetWrapper(env, mode=AutoresetMode.NEXT_STEP)
    params = PinballParams(max_steps_in_episode=2)

    _, state = wrapped.reset(key)
    for _ in range(2):
        _, state, _, _, truncated, _ = wrapped.step(key, state, NUM_ACTIONS - 1, params)
    assert bool(truncated)

    _, state, _, _, _, _ = wrapped.step(key, state, 0, params)
    assert not bool(state.pending)

    obs, state, reward, terminated, truncated, _ = wrapped.step(key, state, 0, params)

    assert state.inner.timestep == 1
    assert reward == -1.0
    assert not bool(terminated)
    assert not bool(truncated)
    assert not bool(state.pending)
