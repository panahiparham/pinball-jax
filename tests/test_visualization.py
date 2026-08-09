"""End-to-end tests for pinball_jax.visualization."""

from __future__ import annotations

import jax
import pytest

pytest.importorskip("matplotlib")

from PIL import Image

from pinball_jax.pinball import Pinball
from pinball_jax.visualization import (
    BehaviorAnimator,
    HeatmapAnimator,
    Trajectory,
    TransitionRecorder,
    occupancy_histogram,
    record_rollout,
    save_behavior_gif,
    save_occupancy_heatmap,
    save_occupancy_heatmap_gif,
)


@pytest.fixture
def trajectory() -> Trajectory:
    env = Pinball("box")
    recorder = TransitionRecorder()
    record_rollout(env, jax.random.key(0), recorder, max_steps=25)
    return recorder.trajectory()


def test_record_rollout_captures_an_episode(trajectory: Trajectory) -> None:
    assert 2 <= len(trajectory.x) <= 26  # +1 for the reset frame
    fields = (trajectory.x, trajectory.y, trajectory.xdot, trajectory.ydot)
    assert len({f.shape for f in fields}) == 1
    assert trajectory.terminated.dtype == bool
    # Ball starts at the "box" config's start position.
    assert trajectory.x[0] == pytest.approx(0.2, abs=1e-6)
    assert trajectory.y[0] == pytest.approx(0.9, abs=1e-6)


def test_recorder_reset_clears_state() -> None:
    env = Pinball("empty")
    recorder = TransitionRecorder()
    record_rollout(env, jax.random.key(0), recorder, max_steps=5)
    assert len(recorder.trajectory().x) > 0

    recorder.reset()
    assert len(recorder.trajectory().x) == 0


def test_occupancy_histogram_conserves_visit_count(trajectory: Trajectory) -> None:
    counts = occupancy_histogram(trajectory, bins=10)
    assert counts.shape == (10, 10)
    assert counts.min() >= 0
    assert counts.sum() == pytest.approx(len(trajectory.x))


def test_save_behavior_gif_writes_an_animated_gif(
    tmp_path, trajectory: Trajectory
) -> None:
    env = Pinball("box")
    path = tmp_path / "behavior.gif"
    save_behavior_gif(env, trajectory, str(path), fps=10)

    assert path.exists() and path.stat().st_size > 0
    with Image.open(path) as im:
        assert im.format == "GIF"
        assert im.n_frames == len(trajectory.x)


def test_save_occupancy_heatmap_writes_a_static_image(
    tmp_path, trajectory: Trajectory
) -> None:
    env = Pinball("box")
    path = tmp_path / "heatmap.png"
    save_occupancy_heatmap(env, trajectory, str(path))

    assert path.exists() and path.stat().st_size > 0
    with Image.open(path) as im:
        assert im.format == "PNG"


def test_save_occupancy_heatmap_gif_writes_an_animated_gif(
    tmp_path, trajectory: Trajectory
) -> None:
    env = Pinball("box")
    path = tmp_path / "heatmap.gif"
    save_occupancy_heatmap_gif(env, trajectory, str(path), fps=10, bins=8)

    assert path.exists() and path.stat().st_size > 0
    with Image.open(path) as im:
        assert im.format == "GIF"
        # Pillow's GIF writer collapses byte-identical consecutive frames (common
        # here since the normalized heatmap can render the same while occupancy
        # is concentrated in a single bin), so frame count is an upper bound, not
        # an equality.
        assert 1 < im.n_frames <= len(trajectory.x)


def test_drawing_functions_add_no_ticks_labels_or_titles(
    trajectory: Trajectory,
) -> None:
    """The feature itself never sets ticks/labels/titles; only composing callers do."""
    import matplotlib.pyplot as plt

    env = Pinball("box")
    fig, (ax1, ax2) = plt.subplots(1, 2)
    BehaviorAnimator(ax1, env, trajectory)
    HeatmapAnimator(ax2, env, trajectory, bins=8)

    for ax in (ax1, ax2):
        assert ax.get_xticks().size == 0
        assert ax.get_yticks().size == 0
        assert ax.get_title() == ""
        assert ax.get_xlabel() == ""
        assert ax.get_ylabel() == ""
    plt.close(fig)
