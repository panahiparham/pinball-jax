"""Standalone benchmark: DQN vs. a uniform-random agent on Pinball ``easy``.

Runs both agents for 30 seeds (each is one ``jax.vmap`` over seeds), then plots
their mean episodic return over time with 95% bootstrap confidence bands,
alongside one seed's state-occupancy heatmap over its *entire training
lifetime* for each agent. Writes ``benchmark_dqn.pdf``.

The agent/environment interaction code (Q-network, replay buffer, random and
DQN training loops) lives in ``benchmark_common.py``, shared with
``benchmark_throughput.py``.

Run with::

    uv run --group benchmark python benchmark_dqn.py

Pinball reward is -1 per step, so an episode's return is -(its length): reaching
the goal early scores higher (closer to 0); an episode that hits the 1000-step
cutoff scores -1000.
"""

from __future__ import annotations

import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

import benchmark_common as bm
from pinball_jax import Pinball, PinballParams
from pinball_jax.visualization import HeatmapAnimator, Trajectory

# --- configuration ----------------------------------------------------------

SETTING = "easy"
EPISODE_CUTOFF = 1_000
TOTAL_TIMESTEPS = 100_000
N_SEEDS = 30
HEATMAP_SEED = 0  # which of the N_SEEDS training runs to plot as a lifetime heatmap

env = Pinball(SETTING)
env_params = PinballParams(max_steps_in_episode=EPISODE_CUTOFF)
ACTION_DIM = env.action_space(env_params).n
OBS_DIM = int(np.prod(env.observation_space(env_params).shape))


# --- lifetime state-occupancy (one seed's full training history) ------------


def lifetime_trajectory(metrics, seed_idx=HEATMAP_SEED) -> Trajectory:
    """One seed's full training-lifetime state history, as a `Trajectory`."""
    obs_seq = metrics["obs"][seed_idx]
    terminated = metrics["terminated"][seed_idx].astype(bool)
    return Trajectory(
        x=obs_seq[:, 0], y=obs_seq[:, 1], xdot=obs_seq[:, 2], ydot=obs_seq[:, 3],
        terminated=terminated,
    )


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
    """[N_SEEDS, len(GRID)]: each seed's return interpolated onto GRID (NaN outside)."""
    grids = []
    for i in range(N_SEEDS):
        ends, rets = episode_returns(
            metrics["reward"][i], metrics["terminated"][i], metrics["truncated"][i]
        )
        grids.append(
            np.interp(GRID, ends, rets, left=np.nan, right=np.nan)
            if ends.size else np.full(GRID.shape, np.nan)
        )
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


def make_plot(dqn_metrics, random_metrics, random_traj, dqn_traj, path):
    """1x3 figure: learning curves, then each agent's lifetime occupancy heatmap."""
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1.6, 1, 1]}
    )

    curve_ax = axes[0]
    for label, color, metrics in [("DQN", "tab:blue", dqn_metrics),
                                  ("Random Agent", "tab:red", random_metrics)]:
        mean, ci_lo, ci_hi = bootstrap_mean_ci(seed_grids(metrics), n_boot=10_000)
        m = ~np.isnan(mean)
        curve_ax.fill_between(GRID[m], ci_lo[m], ci_hi[m], color=color, alpha=0.2)
        curve_ax.plot(GRID[m], mean[m], lw=2.5, color=color, label=label)  # thick mean

    curve_ax.set_title("Learning curves")
    curve_ax.set_xlabel("Timestep")
    curve_ax.set_ylabel("Return", rotation=0, ha="right", va="center", labelpad=12)
    curve_ax.set_ylim(-1000, 0)
    curve_ax.grid(False)
    curve_ax.spines["top"].set_visible(False)
    curve_ax.spines["right"].set_visible(False)
    curve_ax.legend(loc="lower right", frameon=False)

    for ax, traj, title in [
        (axes[1], random_traj, "Random Agent lifetime state occupancy"),
        (axes[2], dqn_traj, "DQN lifetime state occupancy"),
    ]:
        animator = HeatmapAnimator(ax, env, traj, bins=40)
        animator.update(len(traj.x) - 1)  # HeatmapAnimator styles the axes itself
        ax.set_title(title)

    fig.tight_layout()
    for p in (path, path.replace(".pdf", ".png")):  # PDF (vector) + PNG (for GitHub)
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"saved {p}")


def main():
    """Trains both agents, then writes the learning-curve/occupancy plot."""
    random_train = partial(
        bm.random_train, env=env, env_params=env_params,
        action_dim=ACTION_DIM, total_timesteps=TOTAL_TIMESTEPS,
    )
    dqn_train = partial(
        bm.dqn_train, env=env, env_params=env_params,
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, total_timesteps=TOTAL_TIMESTEPS,
    )

    t = time.perf_counter()
    dqn_metrics = bm.run(dqn_train, N_SEEDS)
    elapsed = time.perf_counter() - t
    print(f"DQN: {N_SEEDS} seeds x {TOTAL_TIMESTEPS} steps in {elapsed:.1f}s")

    t = time.perf_counter()
    random_metrics = bm.run(random_train, N_SEEDS)
    elapsed = time.perf_counter() - t
    print(f"random: {N_SEEDS} seeds x {TOTAL_TIMESTEPS} steps in {elapsed:.1f}s")

    dqn_traj = lifetime_trajectory(dqn_metrics)
    random_traj = lifetime_trajectory(random_metrics)

    make_plot(dqn_metrics, random_metrics, random_traj, dqn_traj, "benchmark_dqn.pdf")


if __name__ == "__main__":
    main()
