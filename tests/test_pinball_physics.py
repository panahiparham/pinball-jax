"""Physics sanity checks, numpy parity oracle, and jit/vmap smoke tests."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pinball_jax
from pinball_jax.pinball import Pinball, PinballParams

from _reference_pinball import PinballModel

CONFIGS = ["empty", "box", "easy", "medium", "hard"]
CONFIG_DIR = Path(pinball_jax.__file__).parent / "configs"
NO_LIMIT = PinballParams(max_steps_in_episode=10**9)

# Actions (see pinball.ACTION_EFFECTS): 0=ACC_X, 1=ACC_Y, 2=DEC_X, 3=DEC_Y, 4=NONE.


@pytest.fixture
def key() -> jax.Array:
    return jax.random.PRNGKey(0)


# --------------------------------------------------------------------------- #
# Sanity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "action,axis,sign",
    [(0, 2, 1.0), (2, 2, -1.0), (1, 3, 1.0), (3, 3, -1.0)],
)
def test_impulse_direction(key: jax.Array, action: int, axis: int, sign: float) -> None:
    """Each cardinal action pushes the ball's velocity the right way."""
    env = Pinball("empty")
    _, state = env.reset(key)
    obs, *_ = env.step(key, state, action)
    assert np.sign(float(obs[axis])) == sign
    # The orthogonal velocity component stays zero.
    other = 5 - axis  # 2<->3
    assert float(obs[other]) == 0.0


def test_no_force_action_keeps_ball_at_rest(key: jax.Array) -> None:
    env = Pinball("empty")
    _, state = env.reset(key)
    obs, _, reward, terminated, _, _ = env.step(key, state, 4)
    assert float(obs[2]) == 0.0 and float(obs[3]) == 0.0
    assert reward == -1.0
    assert not bool(terminated)


def test_ball_stays_in_bounds_when_pushed_into_wall(key: jax.Array) -> None:
    """Driving the ball hard against a wall never leaves the unit square."""
    env = Pinball("empty")
    _, state = env.reset(key)
    jstep = jax.jit(lambda s, a: env.step(key, s, a, NO_LIMIT))
    bounced = False
    prev_xdot = 0.0
    for _ in range(80):
        obs, state, *_ = jstep(state, 0)  # keep pushing +x into the right wall
        x, xdot = float(obs[0]), float(obs[2])
        assert 0.0 <= x <= 1.0
        if xdot < 0.0 < prev_xdot:
            bounced = True
        prev_xdot = xdot
    assert bounced, "ball never bounced back off the wall"


def test_reaching_goal_terminates(key: jax.Array) -> None:
    """Greedily steering toward the target ends the episode with terminated=True."""
    env = Pinball("empty")
    obs, state = env.reset(key)
    jstep = jax.jit(lambda s, a: env.step(key, s, a, NO_LIMIT))
    target = np.asarray(env.target)
    terminated = False
    for _ in range(2000):
        action = 0 if float(state.x) < target[0] else 3  # push toward x, then down in y
        obs, state, _, term, _, _ = jstep(state, action)
        if bool(term):
            terminated = True
            break
    assert terminated
    # Final position is within the target radius.
    dist = np.linalg.norm(np.asarray(obs[:2]) - target)
    assert dist < float(env.target_rad)


# --------------------------------------------------------------------------- #
# Parity oracle: JAX env must match the vendored numpy reference.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_numpy_reference(key: jax.Array, config: str, seed: int) -> None:
    ref = PinballModel(str(CONFIG_DIR / f"{config}.cfg"), np.random.default_rng(0))
    ref.reset_ball_to_start_state()

    env = Pinball(config)
    _, state = env.reset(key)
    jstep = jax.jit(lambda s, a: env.step(key, s, a, NO_LIMIT))

    # Starts must agree (single-start configs are deterministic).
    np.testing.assert_allclose(env.reset(key)[0][:2], ref.get_state()[:2], atol=1e-12)

    actions = np.random.default_rng(seed).integers(0, 5, size=300)
    for action in actions:
        action = int(action)
        ref.take_action(action)
        ref_state = np.asarray(ref.get_state())

        _, state, _, terminated, _, _ = jstep(state, action)
        jax_state = np.asarray([state.x, state.y, state.xdot, state.ydot])

        np.testing.assert_allclose(jax_state, ref_state, atol=1e-9)
        assert bool(terminated) == bool(ref.episode_ended())
        if bool(terminated):
            break


# --------------------------------------------------------------------------- #
# jit / vmap smoke
# --------------------------------------------------------------------------- #


def test_jit_step_runs(key: jax.Array) -> None:
    env = Pinball("box")
    _, state = env.reset(key)
    jstep = jax.jit(lambda s, a: env.step(key, s, a, NO_LIMIT))
    obs, next_state, reward, terminated, truncated, _ = jstep(state, jnp.int32(0))
    assert obs.shape == (4,)
    assert next_state.timestep == 1


def test_vmap_over_batch(key: jax.Array) -> None:
    env = Pinball("box")
    keys = jax.random.split(key, 8)
    obs, states = jax.vmap(env.reset)(keys)
    assert obs.shape == (8, 4)

    actions = jnp.arange(8) % 5
    step = jax.vmap(lambda s, a: env.step(key, s, a, NO_LIMIT))
    obs2, states2, reward, terminated, truncated, _ = step(states, actions)
    assert obs2.shape == (8, 4)
    assert reward.shape == (8,)
    assert terminated.shape == (8,)
    assert bool(jnp.all(states2.timestep == 1))
