"""Stubbed Pinball environment.

Implements the ``GymEnv`` protocol (see ``gym_env.py``) with placeholder
dynamics: the observation is always the zero vector, every action is a
no-op, and every step incurs a reward of -1. Episodes never terminate on
their own but are truncated once ``max_steps_in_episode`` timesteps have
elapsed.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from pinball_jax.gym_env import DiscreteActionSpace, ObservationSpace

OBSERVATION_SHAPE = (4,)
NUM_ACTIONS = 5
DEFAULT_MAX_STEPS_IN_EPISODE = 1000


class PinballParams(NamedTuple):
    max_steps_in_episode: int = DEFAULT_MAX_STEPS_IN_EPISODE


class PinballState(NamedTuple):
    timestep: jax.Array


class _PinballObservationSpace:
    @property
    def shape(self) -> tuple[int, ...]:
        return OBSERVATION_SHAPE

    @property
    def dtype(self) -> jnp.dtype:
        return jnp.float32


class _PinballActionSpace:
    @property
    def n(self) -> int:
        return NUM_ACTIONS


class Pinball:
    """Placeholder Pinball environment. Dynamics are not yet implemented."""

    def observation_space(self, params: PinballParams | None = None) -> ObservationSpace:
        del params
        return _PinballObservationSpace()

    def action_space(self, params: PinballParams | None = None) -> DiscreteActionSpace:
        del params
        return _PinballActionSpace()

    def reset(
        self,
        key: jax.Array,
        params: PinballParams | None = None,
    ) -> tuple[jax.Array, PinballState]:
        del key, params
        obs = jnp.zeros(OBSERVATION_SHAPE, dtype=jnp.float32)
        state = PinballState(timestep=jnp.asarray(0, dtype=jnp.int32))
        return obs, state

    def step(
        self,
        key: jax.Array,
        state: PinballState,
        action: jax.Array,
        params: PinballParams | None = None,
    ) -> tuple[jax.Array, PinballState, jax.Array, jax.Array, dict[str, jax.Array]]:
        del key, action
        params = params if params is not None else PinballParams()

        timestep = state.timestep + 1
        next_state = PinballState(timestep=timestep)

        obs = jnp.zeros(OBSERVATION_SHAPE, dtype=jnp.float32)
        reward = jnp.asarray(-1.0, dtype=jnp.float32)
        terminated = jnp.asarray(False)
        truncated = timestep >= params.max_steps_in_episode
        # The protocol carries a single combined "done" signal; terminated
        # and truncated are surfaced separately via info for callers that
        # need to distinguish the two.
        done = jnp.logical_or(terminated, truncated)
        info: dict[str, jax.Array] = {"terminated": terminated, "truncated": truncated}

        return obs, next_state, reward, done, info
