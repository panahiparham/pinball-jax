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
