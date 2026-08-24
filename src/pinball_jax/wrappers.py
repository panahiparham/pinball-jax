"""Autoreset wrappers for GymEnv-conforming environments.

Gymnax-style environments (see `pinball_jax.gym_env`) report the true
boundary observation on the step that ends an episode and leave the caller
to call `reset` itself. `AutoresetWrapper` adds an opt-in scheme where the
environment resets itself instead, following one of two conventions:

* `AutoresetMode.DISABLED`: passthrough, matching the wrapped env unchanged.
* `AutoresetMode.NEXT_STEP`: the step after termination/truncation ignores
  its action and returns the next episode's initial observation, with
  `reward=0` and both flags `False`.
"""

from __future__ import annotations

import enum
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from pinball_jax.gym_env import GymEnv, ObservationSpace


class AutoresetMode(enum.Enum):
    """Autoreset scheme applied by `AutoresetWrapper.step`."""

    DISABLED = "disabled"
    NEXT_STEP = "next_step"


class AutoresetState(NamedTuple):
    """Wrapped environment state plus a pending-reset flag.

    `pending` is `True` exactly when the previous step ended an episode
    (terminated or truncated) and `inner` has not yet been reset for the
    next episode.
    """

    inner: Any
    pending: jax.Array


class AutoresetWrapper[ActionSpaceT]:
    """Adds autoreset semantics to a `GymEnv`-conforming environment.

    Args:
        env: The environment to wrap. Its own `step`/`reset` are left
            unmodified; this wrapper only manages the pending-reset flag.
        mode: `AutoresetMode.DISABLED` (default) passes `env` through
            unchanged. `AutoresetMode.NEXT_STEP` adds a "dead" step per
            episode: the step after termination/truncation ignores its
            action and returns the next episode's initial observation.
    """

    def __init__(
        self,
        env: GymEnv[ActionSpaceT],
        mode: AutoresetMode = AutoresetMode.DISABLED,
    ) -> None:
        self.env = env
        self.mode = mode

    def observation_space(self, params: object | None = None) -> ObservationSpace:
        """Returns the space describing valid observations."""
        return self.env.observation_space(params)

    def action_space(self, params: object | None = None) -> ActionSpaceT:
        """Returns the space describing valid actions."""
        return self.env.action_space(params)

    def reset(
        self, key: jax.Array, params: object | None = None
    ) -> tuple[jax.Array, AutoresetState]:
        """Returns the initial `(observation, state)` for a new episode."""
        obs, inner = self.env.reset(key, params)
        return obs, AutoresetState(inner=inner, pending=jnp.asarray(False))

    def step(
        self,
        key: jax.Array,
        state: AutoresetState,
        action: jax.Array,
        params: object | None = None,
    ) -> tuple[
        jax.Array, AutoresetState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]
    ]:
        """Returns `(observation, state, reward, terminated, truncated, info)`.

        In `AutoresetMode.NEXT_STEP`, a step taken when `state.pending` is
        `True` ignores `action`, resets the wrapped environment, and returns
        its initial observation with `reward=0` and both flags `False`.
        """
        if self.mode is AutoresetMode.DISABLED:
            obs, inner, reward, terminated, truncated, info = self.env.step(
                key, state.inner, action, params
            )
            next_state = AutoresetState(inner=inner, pending=jnp.asarray(False))
            return obs, next_state, reward, terminated, truncated, info

        raise NotImplementedError(
            f"AutoresetMode.{self.mode.name} is not yet implemented"
        )
