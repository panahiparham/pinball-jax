"""Shared test configuration.

Enable JAX float64 for the whole test session. The environment casts
observations to float32 (matching production), but running the internal physics
in float64 lets the parity tests compare against the float64 numpy reference at
machine precision instead of a loose float32 tolerance.
"""

import jax

jax.config.update("jax_enable_x64", True)

CONFIGS = ["empty", "box", "easy", "medium", "hard"]
