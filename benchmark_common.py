"""Shared benchmark code for DQN vs. random agents on Pinball."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

# --- DQN hyperparameters (from coresets pinball_1000/large.json) ------------

LR = 0.002
BUFFER_SIZE = 10_000
BATCH_SIZE = 32
LEARNING_STARTS = 1_000
TARGET_REFRESH = 100          # hard target-network copy every N steps
GAMMA = 0.99
EPSILON = 0.1                 # constant epsilon-greedy
HIDDEN_SIZE = 32

optimizer = optax.adam(LR)


# --- Q-network (a plain MLP as a list of (W, b) params) ---------------------

def init_mlp(key, sizes):
    """Initialize a plain MLP: list of (W, b) tuples with He initialization."""
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        w = jax.random.normal(k, (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in)
        params.append((w, jnp.zeros(fan_out)))
    return params


def mlp(params, x):
    """Q-values for a single obs ``(OBS_DIM,)`` or a batch ``(B, OBS_DIM)``."""
    for w, b in params[:-1]:
        x = jax.nn.relu(x @ w + b)
    w, b = params[-1]
    return x @ w + b


# --- uniform replay buffer (fixed-size, in-JAX) -----------------------------

class Buffer(NamedTuple):
    """Fixed-size ring buffer of transitions, stored as stacked JAX arrays."""

    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    terminated: jax.Array
    pos: jax.Array      # number of transitions ever added
    size: jax.Array     # number currently stored (<= BUFFER_SIZE)


def buffer_init(obs_dim):
    """Returns an empty ``Buffer`` sized for observations of ``obs_dim``."""
    z = jnp.zeros((BUFFER_SIZE, obs_dim), dtype=jnp.float32)
    return Buffer(
        obs=z,
        action=jnp.zeros(BUFFER_SIZE, jnp.int32),
        reward=jnp.zeros(BUFFER_SIZE),
        next_obs=z,
        terminated=jnp.zeros(BUFFER_SIZE, bool),
        pos=jnp.int32(0),
        size=jnp.int32(0),
    )


def buffer_add(b, obs, action, reward, next_obs, terminated):
    """Returns ``b`` with one transition written at its ring position."""
    i = b.pos % BUFFER_SIZE
    return Buffer(
        obs=b.obs.at[i].set(obs),
        action=b.action.at[i].set(action),
        reward=b.reward.at[i].set(reward),
        next_obs=b.next_obs.at[i].set(next_obs),
        terminated=b.terminated.at[i].set(terminated),
        pos=b.pos + 1,
        size=jnp.minimum(b.size + 1, BUFFER_SIZE),
    )


def buffer_sample(b, key):
    """Returns a uniformly sampled ``(obs, action, reward, next_obs, terminated)``."""
    idx = jax.random.randint(key, (BATCH_SIZE,), 0, b.size)
    return b.obs[idx], b.action[idx], b.reward[idx], b.next_obs[idx], b.terminated[idx]


# --- one agent-environment interaction, per seed ----------------------------
# Each returns per-timestep metrics {reward, terminated, truncated, obs}
# (obs is the true post-step observation, i.e. before any auto-reset);
# vmapping over the rng key runs many of them at once.

def random_train(rng, env, env_params, action_dim, total_timesteps):
    """Run a uniform-random agent, auto-resetting on episode end."""
    rng, k = jax.random.split(rng)
    obs, state = env.reset(k, env_params)

    def step(carry, _):
        state, obs, rng = carry
        rng, k_a, k_step, k_reset = jax.random.split(rng, 4)
        action = jax.random.randint(k_a, (), 0, action_dim, dtype=jnp.int32)
        next_obs, next_state, reward, term, trunc, _ = env.step(
            k_step, state, action, env_params
        )
        done = term | trunc
        r_obs, r_state = env.reset(k_reset, env_params)
        carry_obs = jnp.where(done, r_obs, next_obs)
        carry_state = jax.tree.map(
            lambda a, b: jnp.where(done, a, b), r_state, next_state
        )
        metrics = {
            "reward": reward, "terminated": term, "truncated": trunc, "obs": next_obs
        }
        return (carry_state, carry_obs, rng), metrics

    _, metrics = jax.lax.scan(step, (state, obs, rng), jnp.arange(total_timesteps))
    return metrics


def dqn_train(rng, env, env_params, obs_dim, action_dim, total_timesteps):
    """Run a DQN agent (epsilon-greedy, hard target updates) with auto-reset."""
    rng, k_init, k_reset = jax.random.split(rng, 3)
    params = init_mlp(k_init, [obs_dim, HIDDEN_SIZE, HIDDEN_SIZE, action_dim])
    target = params
    opt_state = optimizer.init(params)
    buffer = buffer_init(obs_dim)
    obs, state = env.reset(k_reset, env_params)

    def step(carry, t):
        params, target, opt_state, buffer, state, obs, rng = carry
        rng, k_a, k_expl, k_step, k_reset, k_sample = jax.random.split(rng, 6)

        greedy = jnp.argmax(mlp(params, obs)).astype(jnp.int32)
        rand_a = jax.random.randint(k_a, (), 0, action_dim, dtype=jnp.int32)
        action = jnp.where(jax.random.uniform(k_expl) < EPSILON, rand_a, greedy)

        next_obs, next_state, reward, term, trunc, _ = env.step(
            k_step, state, action, env_params
        )
        buffer = buffer_add(buffer, obs, action, reward, next_obs, term)

        done = term | trunc
        r_obs, r_state = env.reset(k_reset, env_params)
        carry_obs = jnp.where(done, r_obs, next_obs)
        carry_state = jax.tree.map(
            lambda a, b: jnp.where(done, a, b), r_state, next_state
        )

        def do_train(params, opt_state):
            b_obs, b_a, b_r, b_nobs, b_term = buffer_sample(buffer, k_sample)

            def loss_fn(p):
                q = jnp.take_along_axis(mlp(p, b_obs), b_a[:, None], axis=-1)
                q_a = q.squeeze(-1)
                bootstrap = jnp.max(mlp(target, b_nobs), axis=-1) * (1.0 - b_term)
                target_q = b_r + GAMMA * bootstrap
                return jnp.mean((q_a - jax.lax.stop_gradient(target_q)) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            return optax.apply_updates(params, updates), opt_state, loss

        can_train = (t >= LEARNING_STARTS) & (buffer.size >= BATCH_SIZE)
        no_train = lambda p, o: (p, o, jnp.float32(0.0))
        params, opt_state, _ = jax.lax.cond(
            can_train, do_train, no_train, params, opt_state
        )
        target = jax.lax.cond(t % TARGET_REFRESH == 0, lambda: params, lambda: target)

        carry = (params, target, opt_state, buffer, carry_state, carry_obs, rng)
        metrics = {
            "reward": reward, "terminated": term, "truncated": trunc, "obs": next_obs
        }
        return carry, metrics

    carry0 = (params, target, opt_state, buffer, state, obs, rng)
    _, metrics = jax.lax.scan(step, carry0, jnp.arange(total_timesteps))
    return metrics


def run(train_fn, n_seeds):
    """Run one agent for n_seeds seeds; returns metrics dict of [n_seeds, T] arrays."""
    keys = jax.vmap(jax.random.key)(jnp.arange(n_seeds))
    out = jax.jit(jax.vmap(train_fn))(keys)
    return {k: np.asarray(v) for k, v in out.items()}
