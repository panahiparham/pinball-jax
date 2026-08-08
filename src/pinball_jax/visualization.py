"""Visualization utilities for inspecting agent behavior on Pinball.

Optional feature: requires ``matplotlib`` (``uv sync --group viz``). This module
is not imported by ``pinball_jax/__init__.py``, so the core package stays
matplotlib-free.

Transitions are captured with :class:`TransitionRecorder`, whose ``record``
method is a small wrapper around ``jax.experimental.io_callback``: a host-side
side effect that can be called from inside a jitted interaction loop (a
``lax.while_loop``/``fori_loop``/``scan`` body) without breaking ``jit``.
:func:`record_rollout` is a worked example, running one policy-driven episode
and streaming every transition to a recorder. Wiring is opt-in and works under
``jit`` but not ``vmap`` (regular ``io_callback`` semantics).

From the resulting :class:`Trajectory`, three artifacts can be produced:

* :func:`save_behavior_gif` - an animated GIF of the ball moving through the
  obstacle course.
* :func:`save_occupancy_heatmap` - a static 2D histogram of visited
  ``(x, y)`` positions, aggregated over the velocity dimensions.
* :func:`save_occupancy_heatmap_gif` - the same histogram, animated as it
  accumulates over the course of the episode.

None of these draw a title; callers composing a figure of their own add
whatever titles they want.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Polygon

from pinball_jax.pinball import Pinball, PinballParams, PinballState

BACKGROUND_COLOR = "white"
OBSTACLE_COLOR = "gray"
BALL_COLOR = "blue"
TARGET_COLOR = "red"


class Trajectory(NamedTuple):
    """One recorded episode, as per-step numpy arrays (including the reset state)."""

    x: np.ndarray
    y: np.ndarray
    xdot: np.ndarray
    ydot: np.ndarray
    terminated: np.ndarray


class TransitionRecorder:
    """Accumulates transitions on the host via a JAX-compatible callback.

    Call ``record(state, terminated)`` right after ``env.step`` inside a
    jitted interaction loop; each call round-trips to the host through
    ``jax.experimental.io_callback`` (ordered, so transitions land in order)
    and appends to an in-memory buffer. Retrieve the result with
    :meth:`trajectory`.
    """

    def __init__(self) -> None:
        self._x: list[float] = []
        self._y: list[float] = []
        self._xdot: list[float] = []
        self._ydot: list[float] = []
        self._terminated: list[bool] = []

    def _append(self, x, y, xdot, ydot, terminated) -> None:
        self._x.append(float(x))
        self._y.append(float(y))
        self._xdot.append(float(xdot))
        self._ydot.append(float(ydot))
        self._terminated.append(bool(terminated))

    def record(self, state: PinballState, terminated: jax.Array) -> None:
        """Record one transition. Safe to call from inside a jitted loop body."""
        jax.experimental.io_callback(
            self._append, None, state.x, state.y, state.xdot, state.ydot, terminated, ordered=True
        )

    def trajectory(self) -> Trajectory:
        """Snapshot everything recorded so far."""
        return Trajectory(
            x=np.asarray(self._x),
            y=np.asarray(self._y),
            xdot=np.asarray(self._xdot),
            ydot=np.asarray(self._ydot),
            terminated=np.asarray(self._terminated, dtype=bool),
        )

    def reset(self) -> None:
        """Clear all recorded transitions."""
        self._x.clear()
        self._y.clear()
        self._xdot.clear()
        self._ydot.clear()
        self._terminated.clear()


def record_rollout(
    env: Pinball,
    key: jax.Array,
    recorder: TransitionRecorder,
    params: PinballParams | None = None,
    policy: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    max_steps: int = 400,
) -> None:
    """Run one jitted episode, streaming every transition to ``recorder``.

    ``policy(key, obs) -> action`` defaults to uniform-random over
    ``env.action_space().n``. The episode stops early once the target is
    reached or ``max_steps`` is hit, mirroring ``terminated``/``truncated``
    from :meth:`Pinball.step`.
    """
    params = params if params is not None else PinballParams()
    n_actions = env.action_space(params).n
    policy_fn = policy or (lambda key, obs: jax.random.randint(key, (), 0, n_actions))

    @jax.jit
    def run(key):
        key, k_reset = jax.random.split(key)
        _, state0 = env.reset(k_reset, params)
        recorder.record(state0, jnp.asarray(False))

        def cond(carry):
            _, _, done, i = carry
            return (~done) & (i < max_steps)

        def body(carry):
            state, key, _done, i = carry
            key, k_policy, k_step = jax.random.split(key, 3)
            obs = jnp.stack([state.x, state.y, state.xdot, state.ydot]).astype(jnp.float32)
            action = policy_fn(k_policy, obs)
            _, next_state, _, terminated, truncated, _ = env.step(k_step, state, action, params)
            recorder.record(next_state, terminated)
            return (next_state, key, terminated | truncated, i + 1)

        return jax.lax.while_loop(cond, body, (state0, key, jnp.asarray(False), 0))

    run(key)


def _obstacle_polygons(env: Pinball) -> list[np.ndarray]:
    """Reconstruct each obstacle's vertex loop from the env's baked edge arrays."""
    edge_p0 = np.asarray(env.edge_p0)
    edge_mask = np.asarray(env.edge_mask)
    return [edge_p0[o][edge_mask[o]] for o in range(edge_p0.shape[0])]


def _style_axes(ax: Axes) -> None:
    """Square [0, 1]^2 axes, y flipped to match Pinball's on-screen convention.

    No ticks, tick labels, or axis labels. Call last, after other artists have
    been added (``imshow`` otherwise resets the limits it sets here).
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_facecolor(BACKGROUND_COLOR)


def draw_obstacles(ax: Axes, env: Pinball, **kwargs) -> None:
    """Draw ``env``'s obstacles as filled polygons (default: gray)."""
    style = {"facecolor": OBSTACLE_COLOR, "edgecolor": "black", "linewidth": 0.5, "zorder": 2}
    style.update(kwargs)
    for pts in _obstacle_polygons(env):
        ax.add_patch(Polygon(pts, closed=True, **style))


def draw_target(ax: Axes, env: Pinball, **kwargs) -> None:
    """Draw ``env``'s target region as a filled circle (default: red)."""
    tx, ty = float(env.target[0]), float(env.target[1])
    style = {"facecolor": TARGET_COLOR, "edgecolor": "black", "zorder": 5}
    style.update(kwargs)
    ax.add_patch(Circle((tx, ty), float(env.target_rad), **style))


def _ball_radius(env: Pinball) -> float:
    # Floor the drawn radius so a visually tiny ball_rad still shows up as a marker.
    return max(float(env.ball_rad), 0.012)


class BehaviorAnimator:
    """Draws the static scene once, then updates a ball marker + trail per frame."""

    def __init__(self, ax: Axes, env: Pinball, trajectory: Trajectory, trail: bool = True) -> None:
        self.trajectory = trajectory
        draw_obstacles(ax, env)
        draw_target(ax, env)
        self.trail_line = (
            ax.plot([], [], color="steelblue", linewidth=1, alpha=0.6, zorder=4)[0] if trail else None
        )
        self.ball = Circle(
            (trajectory.x[0], trajectory.y[0]), _ball_radius(env), facecolor=BALL_COLOR, edgecolor="black", zorder=6
        )
        ax.add_patch(self.ball)
        _style_axes(ax)

    def update(self, step: int) -> list:
        idx = min(step, len(self.trajectory.x) - 1)
        self.ball.center = (self.trajectory.x[idx], self.trajectory.y[idx])
        artists = [self.ball]
        if self.trail_line is not None:
            self.trail_line.set_data(self.trajectory.x[: idx + 1], self.trajectory.y[: idx + 1])
            artists.append(self.trail_line)
        return artists


def occupancy_histogram(trajectory: Trajectory, bins: int = 40) -> np.ndarray:
    """2D histogram of visited ``(x, y)``, aggregated over the velocity dimensions.

    Returns an array shaped ``(bins, bins)`` indexed ``[y_bin, x_bin]``, ready
    for ``imshow(..., origin="lower", extent=(0, 1, 0, 1))``.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts, _, _ = np.histogram2d(trajectory.x, trajectory.y, bins=[edges, edges])
    return counts.T


class HeatmapAnimator:
    """Draws the static scene once, then updates a cumulative occupancy heatmap per frame."""

    def __init__(self, ax: Axes, env: Pinball, trajectory: Trajectory, bins: int = 40, cmap: str = "viridis") -> None:
        self.trajectory = trajectory
        self.bins = bins
        self.image = ax.imshow(
            np.zeros((bins, bins)), origin="lower", extent=(0, 1, 0, 1), cmap=cmap, vmin=0, aspect="equal", zorder=0
        )
        draw_obstacles(ax, env)
        draw_target(ax, env)
        _style_axes(ax)

    def update(self, step: int) -> list:
        idx = min(step, len(self.trajectory.x) - 1)
        partial = self.trajectory._replace(x=self.trajectory.x[: idx + 1], y=self.trajectory.y[: idx + 1])
        counts = occupancy_histogram(partial, bins=self.bins)
        self.image.set_data(counts)
        self.image.set_clim(0, max(counts.max(), 1))
        return [self.image]


def save_behavior_gif(
    env: Pinball, trajectory: Trajectory, path: str, fps: int = 15, figsize=(4, 4), dpi: int = 100, trail: bool = True
) -> None:
    """Save an animated GIF of ``trajectory``'s ball moving through ``env``."""
    fig, ax = plt.subplots(figsize=figsize)
    animator = BehaviorAnimator(ax, env, trajectory, trail=trail)
    anim = FuncAnimation(fig, animator.update, frames=len(trajectory.x), interval=1000 / fps, blit=False)
    anim.save(str(path), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def save_occupancy_heatmap(
    env: Pinball, trajectory: Trajectory, path: str, bins: int = 40, cmap: str = "viridis", figsize=(4, 4), dpi: int = 100
) -> None:
    """Save a static occupancy heatmap of ``trajectory`` over ``env``'s obstacle course."""
    fig, ax = plt.subplots(figsize=figsize)
    counts = occupancy_histogram(trajectory, bins=bins)
    ax.imshow(counts, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, vmin=0, aspect="equal", zorder=0)
    draw_obstacles(ax, env)
    draw_target(ax, env)
    _style_axes(ax)
    fig.savefig(str(path), dpi=dpi, facecolor=BACKGROUND_COLOR)
    plt.close(fig)


def save_occupancy_heatmap_gif(
    env: Pinball,
    trajectory: Trajectory,
    path: str,
    bins: int = 40,
    fps: int = 15,
    cmap: str = "viridis",
    figsize=(4, 4),
    dpi: int = 100,
) -> None:
    """Save an animated GIF of ``trajectory``'s occupancy heatmap accumulating over time."""
    fig, ax = plt.subplots(figsize=figsize)
    animator = HeatmapAnimator(ax, env, trajectory, bins=bins, cmap=cmap)
    anim = FuncAnimation(fig, animator.update, frames=len(trajectory.x), interval=1000 / fps, blit=False)
    anim.save(str(path), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
