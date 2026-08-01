# pinball-jax

A JAX implementation of the Pinball reinforcement learning environment: a ball
driven by cardinal-direction impulses and drag, bouncing off polygon obstacles,
with the episode terminating when it reaches a circular target. `reset` and
`step` are pure functions and are fully JIT- and vmap-able.

## Installation

Add it to your project with [uv](https://docs.astral.sh/uv/):

```sh
uv add git+https://github.com/panahiparham/pinball-jax
```

## Usage

```python
import jax
from pinball_jax import Pinball, PinballParams

env = Pinball("box")  # bundled config name, or a path to a .cfg file
params = PinballParams(max_steps_in_episode=100)

key = jax.random.PRNGKey(0)
obs, state = env.reset(key)

# obs = [x, y, xdot, ydot]
# actions: 0 = +x, 1 = +y, 2 = -x, 3 = -y, 4 = no force
obs, state, reward, terminated, truncated, info = env.step(key, state, 0, params)
```

Five domains are bundled and selectable by name: `empty`, `box`, `easy`,
`medium`, `hard`. See [`example.py`](example.py) for a jitted `lax.scan` rollout.

## Status

Early development.
