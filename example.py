"""Minimal Pinball usage example: a jitted ``lax.scan`` rollout.

Run with::

    uv run --python 3.12 python example.py
"""

import jax
import jax.numpy as jnp

from pinball_jax import Pinball, PinballParams

NUM_STEPS = 20


def main() -> None:
    env = Pinball("box")  # bundled config name, or a path to a .cfg file
    params = PinballParams(max_steps_in_episode=100)

    key = jax.random.PRNGKey(0)
    reset_key, action_key, rollout_key = jax.random.split(key, 3)

    obs, state = env.reset(reset_key)
    actions = jax.random.randint(action_key, (NUM_STEPS,), 0, 5)

    @jax.jit
    def rollout(key, state, actions):
        def step(carry, action):
            key, state = carry
            key, subkey = jax.random.split(key)
            obs, state, reward, terminated, truncated, _ = env.step(subkey, state, action, params)
            return (key, state), (obs, reward, terminated, truncated)

        (_, final_state), traj = jax.lax.scan(step, (key, state), actions)
        return final_state, traj

    final_state, (obs_seq, rewards, terminated, truncated) = rollout(rollout_key, state, actions)

    print(f"start obs: {obs}")
    for t in range(NUM_STEPS):
        o = obs_seq[t]
        print(
            f"step {t:2d}  action={int(actions[t])}  "
            f"obs=[{o[0]:+.3f}, {o[1]:+.3f}, {o[2]:+.3f}, {o[3]:+.3f}]  "
            f"reward={float(rewards[t]):+.0f}  "
            f"term={bool(terminated[t])}  trunc={bool(truncated[t])}"
        )
    print(f"return over {NUM_STEPS} steps: {float(jnp.sum(rewards)):+.0f}")


if __name__ == "__main__":
    main()
