"""Standalone benchmark: DQN vs. a uniform-random agent on Pinball ``easy``.

Runs both agents for 30 seeds (each is one ``jax.vmap`` over seeds), then plots
their mean episodic return over time with 95% bootstrap confidence bands and
writes ``benchmark_dqn.pdf``. No experiment harness, no results database — a
single self-contained file.

Run with::

    uv run --group benchmark python benchmark_dqn.py

Pinball reward is -1 per step, so an episode's return is -(its length): reaching
the goal early scores higher (closer to 0); an episode that hits the 1000-step
cutoff scores -1000.
"""

from __future__ import annotations

import time
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from pinball_jax import Pinball, PinballParams

# --- configuration ----------------------------------------------------------

SETTING = "easy"
EPISODE_CUTOFF = 1_000
TOTAL_TIMESTEPS = 100_000
N_SEEDS = 30

# DQN hyperparameters (from coresets pinball_1000/large.json).
LR = 0.002
BUFFER_SIZE = 10_000
BATCH_SIZE = 32
LEARNING_STARTS = 1_000
TARGET_REFRESH = 100          # hard target-network copy every N steps
GAMMA = 0.99
EPSILON = 0.1                 # constant epsilon-greedy
HIDDEN_SIZE = 32

env = Pinball(SETTING)
env_params = PinballParams(max_steps_in_episode=EPISODE_CUTOFF)
ACTION_DIM = env.action_space(env_params).n
OBS_DIM = int(np.prod(env.observation_space(env_params).shape))
optimizer = optax.adam(LR)


# --- Q-network (a plain MLP as a list of (W, b) params) ---------------------

def init_mlp(key, sizes):
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        w = jax.random.normal(k, (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in)  # He init
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
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    terminated: jax.Array
    pos: jax.Array      # number of transitions ever added
    size: jax.Array     # number currently stored (<= BUFFER_SIZE)


def buffer_init():
    z = jnp.zeros((BUFFER_SIZE, OBS_DIM), dtype=jnp.float32)
    return Buffer(
        obs=z, action=jnp.zeros(BUFFER_SIZE, jnp.int32), reward=jnp.zeros(BUFFER_SIZE),
        next_obs=z, terminated=jnp.zeros(BUFFER_SIZE, bool),
        pos=jnp.int32(0), size=jnp.int32(0),
    )


def buffer_add(b, obs, action, reward, next_obs, terminated):
    i = b.pos % BUFFER_SIZE
    return Buffer(
        obs=b.obs.at[i].set(obs), action=b.action.at[i].set(action),
        reward=b.reward.at[i].set(reward), next_obs=b.next_obs.at[i].set(next_obs),
        terminated=b.terminated.at[i].set(terminated),
        pos=b.pos + 1, size=jnp.minimum(b.size + 1, BUFFER_SIZE),
    )


def buffer_sample(b, key):
    idx = jax.random.randint(key, (BATCH_SIZE,), 0, b.size)
    return b.obs[idx], b.action[idx], b.reward[idx], b.next_obs[idx], b.terminated[idx]


# --- one agent-environment interaction, per seed ----------------------------
# Each returns per-timestep metrics {reward, terminated, truncated}; vmapping
# over the rng key runs N_SEEDS of them at once.

def _reset(key):
    return env.reset(key, env_params)


def _step_env(key, state, action):
    return env.step(key, state, action, env_params)


def random_train(rng):
    rng, k = jax.random.split(rng)
    obs, state = _reset(k)

    def step(carry, _):
        state, obs, rng = carry
        rng, k_a, k_step, k_reset = jax.random.split(rng, 4)
        action = jax.random.randint(k_a, (), 0, ACTION_DIM, dtype=jnp.int32)
        next_obs, next_state, reward, term, trunc, _ = _step_env(k_step, state, action)
        done = term | trunc
        r_obs, r_state = _reset(k_reset)
        next_obs = jnp.where(done, r_obs, next_obs)
        next_state = jax.tree.map(lambda a, b: jnp.where(done, a, b), r_state, next_state)
        return (next_state, next_obs, rng), {"reward": reward, "terminated": term, "truncated": trunc}

    _, metrics = jax.lax.scan(step, (state, obs, rng), jnp.arange(TOTAL_TIMESTEPS))
    return metrics


def dqn_train(rng):
    rng, k_init, k_reset = jax.random.split(rng, 3)
    params = init_mlp(k_init, [OBS_DIM, HIDDEN_SIZE, HIDDEN_SIZE, ACTION_DIM])
    target = params
    opt_state = optimizer.init(params)
    buffer = buffer_init()
    obs, state = _reset(k_reset)

    def step(carry, t):
        params, target, opt_state, buffer, state, obs, rng = carry
        rng, k_a, k_expl, k_step, k_reset, k_sample = jax.random.split(rng, 6)

        greedy = jnp.argmax(mlp(params, obs)).astype(jnp.int32)
        rand_a = jax.random.randint(k_a, (), 0, ACTION_DIM, dtype=jnp.int32)
        action = jnp.where(jax.random.uniform(k_expl) < EPSILON, rand_a, greedy)

        next_obs, next_state, reward, term, trunc, _ = _step_env(k_step, state, action)
        buffer = buffer_add(buffer, obs, action, reward, next_obs, term)

        done = term | trunc
        r_obs, r_state = _reset(k_reset)
        next_obs = jnp.where(done, r_obs, next_obs)
        next_state = jax.tree.map(lambda a, b: jnp.where(done, a, b), r_state, next_state)

        def do_train(params, opt_state):
            b_obs, b_a, b_r, b_nobs, b_term = buffer_sample(buffer, k_sample)

            def loss_fn(p):
                q_a = jnp.take_along_axis(mlp(p, b_obs), b_a[:, None], axis=-1).squeeze(-1)
                target_q = b_r + GAMMA * jnp.max(mlp(target, b_nobs), axis=-1) * (1.0 - b_term)
                return jnp.mean((q_a - jax.lax.stop_gradient(target_q)) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            return optax.apply_updates(params, updates), opt_state, loss

        can_train = (t >= LEARNING_STARTS) & (buffer.size >= BATCH_SIZE)
        params, opt_state, _ = jax.lax.cond(
            can_train, do_train, lambda p, o: (p, o, jnp.float32(0.0)), params, opt_state
        )
        target = jax.lax.cond(t % TARGET_REFRESH == 0, lambda: params, lambda: target)

        carry = (params, target, opt_state, buffer, next_state, next_obs, rng)
        return carry, {"reward": reward, "terminated": term, "truncated": trunc}

    carry0 = (params, target, opt_state, buffer, state, obs, rng)
    _, metrics = jax.lax.scan(step, carry0, jnp.arange(TOTAL_TIMESTEPS))
    return metrics


def run(train_fn):
    """Run one agent for N_SEEDS seeds; returns metrics dict of [N_SEEDS, T] arrays."""
    keys = jax.vmap(jax.random.key)(jnp.arange(N_SEEDS))
    out = jax.jit(jax.vmap(train_fn))(keys)
    return {k: np.asarray(v) for k, v in out.items()}


# --- return-over-time analysis (episodic return; no smoothing) --------------

GRID = np.linspace(0, TOTAL_TIMESTEPS, 500)


def episode_returns(reward, terminated, truncated):
    """(end_timestep, return) for each completed episode in one run."""
    done = (terminated + truncated) > 0
    ends = np.flatnonzero(done)
    if ends.size == 0:
        return np.array([]), np.array([])
    cumr = np.cumsum(np.asarray(reward, dtype=float))
    prev = np.concatenate(([0.0], cumr[ends[:-1]]))
    return ends, cumr[ends] - prev


def seed_grids(metrics):
    """[N_SEEDS, len(GRID)] of each seed's return interpolated onto GRID (NaN outside)."""
    grids = []
    for i in range(N_SEEDS):
        ends, rets = episode_returns(metrics["reward"][i], metrics["terminated"][i], metrics["truncated"][i])
        grids.append(np.interp(GRID, ends, rets, left=np.nan, right=np.nan) if ends.size
                     else np.full(GRID.shape, np.nan))
    return np.vstack(grids)


def bootstrap_mean_ci(stack, n_boot=10_000, lo=2.5, hi=97.5, seed=0):
    """Mean and percentile-bootstrap CI over seeds, where all seeds are present."""
    n, m = stack.shape
    valid = (~np.isnan(stack)).sum(axis=0) == n
    mean = np.full(m, np.nan)
    ci_lo = np.full(m, np.nan)
    ci_hi = np.full(m, np.nan)
    sub = stack[:, valid]
    mean[valid] = sub.mean(axis=0)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, sub.shape[1]))
    for s in range(0, n_boot, 1000):                       # chunked to bound memory
        e = min(s + 1000, n_boot)
        boot[s:e] = sub[rng.integers(0, n, size=(e - s, n))].mean(axis=1)
    ci_lo[valid], ci_hi[valid] = np.percentile(boot, [lo, hi], axis=0)
    return mean, ci_lo, ci_hi


def make_plot(dqn_metrics, random_metrics, path):
    fig, ax = plt.subplots(figsize=(9, 6))   # 2:3 height:width
    for label, color, metrics in [("DQN", "tab:blue", dqn_metrics),
                                  ("Random Agent", "tab:red", random_metrics)]:
        mean, ci_lo, ci_hi = bootstrap_mean_ci(seed_grids(metrics), n_boot=10_000)
        m = ~np.isnan(mean)
        ax.fill_between(GRID[m], ci_lo[m], ci_hi[m], color=color, alpha=0.2)   # light matched band
        ax.plot(GRID[m], mean[m], lw=2.5, color=color, label=label)           # thick mean

    ax.set_title(f"DQN vs. Random Agent on Pinball {SETTING}  (mean ± 95% bootstrap CI, n={N_SEEDS} seeds)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Return", rotation=0, ha="right", va="center", labelpad=12)
    ax.set_ylim(-1000, 0)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    for p in (path, path.replace(".pdf", ".png")):   # PDF (vector) + PNG (renders on GitHub)
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"saved {p}")


def main():
    for name, fn in [("DQN", dqn_train), ("random", random_train)]:
        t = time.perf_counter()
        globals()[f"{name}_metrics"] = run(fn)
        print(f"{name}: {N_SEEDS} seeds x {TOTAL_TIMESTEPS} steps in {time.perf_counter() - t:.1f}s")
    make_plot(DQN_metrics, random_metrics, "benchmark_dqn.pdf")  # noqa: F821


if __name__ == "__main__":
    main()
