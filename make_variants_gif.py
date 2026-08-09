"""Generate the 2x5 random-policy behavior + occupancy GIF across all 5 variants.

For each bundled config, records one uniform-random-policy rollout via
`pinball_jax.visualization.record_rollout`, then animates it as a column: top
row is `BehaviorAnimator` (ball + trail), bottom row is `HeatmapAnimator` (the
same rollout's occupancy heatmap, accumulating alongside it). Seeds 42-46
(one per config) produce the committed GIF.

Run with::

    uv run --group viz python make_variants_gif.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")  # avoid macOS retina 2x savefig scaling; also faster/headless
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from pinball_jax import Pinball
from pinball_jax.visualization import (
    BehaviorAnimator,
    HeatmapAnimator,
    TransitionRecorder,
    record_rollout,
)

CONFIGS = ["empty", "box", "easy", "medium", "hard"]
SEED = 42
MAX_STEPS = 400
STRIDE = 3  # subsample steps per rendered frame, for a small, fast-to-render gif
BINS = 40
OUTPUT_PATH = Path("pinball_variants.gif")


def main() -> None:
    """Renders and saves the 2x5 GIF."""
    envs = {}
    trajectories = {}
    for i, name in enumerate(CONFIGS):
        env = Pinball(name)
        recorder = TransitionRecorder()
        record_rollout(env, jax.random.key(SEED + i), recorder, max_steps=MAX_STEPS)
        traj = recorder.trajectory()
        envs[name] = env
        trajectories[name] = traj
        print(f"{name}: {len(traj.x)} steps, terminated={bool(traj.terminated.any())}")

    n_frames = max((len(trajectories[n].x) + STRIDE - 1) // STRIDE for n in CONFIGS)

    fig, axes = plt.subplots(2, len(CONFIGS), figsize=(15, 6))

    behavior_animators = []
    heatmap_animators = []
    for col, name in enumerate(CONFIGS):
        env, traj = envs[name], trajectories[name]

        behavior_ax = axes[0, col]
        behavior_animators.append(BehaviorAnimator(behavior_ax, env, traj))
        behavior_ax.set_title(name)  # only titles in the figure: top row, config names

        heatmap_ax = axes[1, col]
        heatmap_animators.append(HeatmapAnimator(heatmap_ax, env, traj, bins=BINS))

    fig.patch.set_facecolor("white")
    fig.tight_layout()

    def update(frame):
        step = frame * STRIDE
        artists = []
        for animator in behavior_animators + heatmap_animators:
            artists += animator.update(step)
        return artists

    anim = FuncAnimation(fig, update, frames=n_frames, interval=40, blit=False)
    anim.save(OUTPUT_PATH, writer=PillowWriter(fps=12), dpi=90)

    size_bytes = OUTPUT_PATH.stat().st_size
    print(f"Output: {OUTPUT_PATH}")
    print(f"Frames: {n_frames}")
    print(f"Size: {size_bytes} bytes ({size_bytes / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
