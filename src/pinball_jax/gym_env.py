"""Gymnax-style environment Protocol for RL agent typing.

Agents built on the Gymnax compatibility layer accept the tuple-returning
interface described here. New-style environments implementing
``EnvProtocol`` should be wrapped with ``make_gymnax_compat_env`` before
being passed to any ``make_train`` function.

Adapted from:
https://github.com/andnp/jax-research-template/blob/main/libs/rl-components/src/rl_components/gym_env.py

``step`` differs from the source: it returns separate ``terminated`` and
``truncated`` signals instead of a single combined ``done`` flag.
``GymEnv`` is also marked ``@runtime_checkable`` so conformance can be
verified with ``isinstance`` in tests.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp


class ObservationSpace(Protocol):
    """Shape and dtype of an environment's observation array."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Observation array shape."""
        ...

    @property
    def dtype(self) -> jnp.dtype:
        """Observation array dtype."""
        ...


class DiscreteActionSpace(Protocol):
    """A discrete action space of ``n`` actions, indexed ``0`` to ``n - 1``."""

    @property
    def n(self) -> int:
        """Number of discrete actions."""
        ...


class ContinuousActionSpace(Protocol):
    """A continuous action space of the given shape."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Action array shape."""
        ...


@runtime_checkable
class GymEnv[ActionSpaceT](Protocol):
    """Tuple-returning JAX environment interface (Gymnax-style).

    Implementations are pure functions of an explicit PRNG key and state,
    making ``reset``/``step`` JIT- and vmap-able.
    """

    def observation_space(self, params: object | None = None) -> ObservationSpace:
        """Returns the space describing valid observations."""
        ...

    def action_space(self, params: object | None = None) -> ActionSpaceT:
        """Returns the space describing valid actions."""
        ...

    def reset(
        self, key: jax.Array, params: object | None = None
    ) -> tuple[jax.Array, object]:
        """Returns the initial ``(observation, state)`` for a new episode."""
        ...

    def step(
        self,
        key: jax.Array,
        state: Any,
        action: jax.Array,
        params: object | None = None,
    ) -> tuple[
        jax.Array, object, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]
    ]:
        """Returns ``(observation, state, reward, terminated, truncated, info)``."""
        ...
