"""JAX implementation of the Pinball reinforcement learning environment.

A ball moves under cardinal-direction impulses and drag, bouncing off polygon
walls/obstacles, and the episode terminates when the ball reaches a circular
target. The domain geometry (ball radius, target, start positions, obstacles)
is read from a ``.cfg`` config file at construction time and baked into
fixed-shape JAX constants, so ``reset``/``step`` are fully JIT-able and
vmap-able.

This is a faithful translation of the classic numpy ``PinballModel`` (originally
by Pierre-Luc Bacon). Implementation notes on the physics:

* Each action runs 20 physics substeps. An impulse is applied only on the first
  substep (and clipped to the velocity range); drag and boundary clamping happen
  once, after the substeps, and are skipped if the ball reached the target
  (mirroring the reference's early return).
* Collision detection is vectorized over all obstacle edges. Per obstacle we
  only need the *number* of intercepting edges: ``>=1`` means a collision,
  ``>=2`` means a corner hit (velocity is negated). The reference's
  ``_select_edge`` tie-break is dead code (its result feeds only the
  non-corner branch, which never runs when selection happens), so it is
  omitted.
* Across obstacles: exactly one colliding obstacle applies its reflection;
  more than one negates the velocity.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp

from pinball_jax.gym_env import DiscreteActionSpace, ObservationSpace

OBSERVATION_SHAPE = (4,)
NUM_ACTIONS = 5
DEFAULT_MAX_STEPS_IN_EPISODE = 1000

DRAG = 0.995
SUBSTEPS = 20
# Action -> (delta_xdot, delta_ydot) before the /5 impulse scaling.
# 0=ACC_X, 1=ACC_Y, 2=DEC_X, 3=DEC_Y, 4=ACC_NONE.
ACTION_EFFECTS = jnp.asarray(
    [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.0, 0.0]]
)


class PinballParams(NamedTuple):
    max_steps_in_episode: int = DEFAULT_MAX_STEPS_IN_EPISODE


class PinballState(NamedTuple):
    x: jax.Array
    y: jax.Array
    xdot: jax.Array
    ydot: jax.Array
    timestep: jax.Array


class _PinballObservationSpace:
    @property
    def shape(self) -> tuple[int, ...]:
        return OBSERVATION_SHAPE

    @property
    def dtype(self) -> jnp.dtype:
        return jnp.float32


class _PinballActionSpace:
    @property
    def n(self) -> int:
        return NUM_ACTIONS


# --------------------------------------------------------------------------- #
# Host-side config parsing and geometry baking (plain Python, run once at
# construction; produces static-shaped arrays).
# --------------------------------------------------------------------------- #


def _read_config(config: str | Path) -> str:
    """Resolve ``config`` as a filesystem path or a bundled config name."""
    path = Path(config)
    if not path.exists():
        path = Path(__file__).parent / "configs" / f"{config}.cfg"
    if not path.exists():
        raise FileNotFoundError(
            f"Pinball config {config!r} not found (looked for a file path and a "
            f"bundled config name under {Path(__file__).parent / 'configs'})"
        )
    return path.read_text()


def _parse_config(text: str):
    """Parse a Pinball ``.cfg`` file into obstacles/target/start/ball_rad."""
    obstacles: list[list[tuple[float, float]]] = []
    target_pos: tuple[float, float] | None = None
    target_rad = 0.01
    ball_rad = 0.01
    start_pts: list[tuple[float, float]] = []

    for line in text.splitlines():
        tokens = line.strip().split()
        if not tokens:
            continue
        head = tokens[0]
        if head == "polygon":
            coords = list(map(float, tokens[1:]))
            obstacles.append(list(zip(coords[0::2], coords[1::2])))
        elif head == "target":
            target_pos = (float(tokens[1]), float(tokens[2]))
            target_rad = float(tokens[3])
        elif head == "start":
            coords = list(map(float, tokens[1:]))
            start_pts = list(zip(coords[0::2], coords[1::2]))
        elif head == "ball":
            ball_rad = float(tokens[1])

    if target_pos is None:
        raise ValueError("config is missing a 'target' line")
    if not start_pts:
        raise ValueError("config is missing a 'start' line")
    if not obstacles:
        raise ValueError("config has no 'polygon' obstacles")

    return obstacles, target_pos, target_rad, start_pts, ball_rad


def _bake_geometry(obstacles: list[list[tuple[float, float]]]):
    """Pad polygon obstacles into fixed-shape edge arrays.

    Each polygon with ``V`` vertices contributes ``V`` closed edges. Arrays are
    padded to ``E = max V`` edges per obstacle. Padded edges carry
    ``edge_mask=False`` and a nonzero direction so the projection denominator is
    never zero. Bounding boxes are computed from real vertices only.
    """
    import numpy as np

    num_obstacles = len(obstacles)
    max_edges = max(len(pts) for pts in obstacles)

    edge_p0 = np.zeros((num_obstacles, max_edges, 2), dtype=np.float64)
    # Default padded direction is (1, 0) so edge . edge == 1 (never zero).
    edge_p1 = np.zeros((num_obstacles, max_edges, 2), dtype=np.float64)
    edge_p1[..., 0] = 1.0
    edge_mask = np.zeros((num_obstacles, max_edges), dtype=bool)
    bbox_min = np.zeros((num_obstacles, 2), dtype=np.float64)
    bbox_max = np.zeros((num_obstacles, 2), dtype=np.float64)

    for o, pts in enumerate(obstacles):
        arr = np.asarray(pts, dtype=np.float64)
        num_vertices = len(pts)
        bbox_min[o] = arr.min(axis=0)
        bbox_max[o] = arr.max(axis=0)
        for k in range(num_vertices):
            edge_p0[o, k] = arr[k]
            edge_p1[o, k] = arr[(k + 1) % num_vertices]
            edge_mask[o, k] = True

    return edge_p0, edge_p1, edge_mask, bbox_min, bbox_max


# --------------------------------------------------------------------------- #
# Vectorized physics helpers.
# --------------------------------------------------------------------------- #


def _angle(v1: jax.Array, v2: jax.Array) -> jax.Array:
    """Reference angle difference between vectors, wrapped to ``[0, 2*pi)``.

    Note the argument order to ``arctan2`` is ``(x, y)`` (matching the numpy
    reference), which is the reverse of the usual convention.
    """
    diff = jnp.arctan2(v1[..., 0], v1[..., 1]) - jnp.arctan2(v2[..., 0], v2[..., 1])
    return jnp.where(diff < 0, diff + 2 * jnp.pi, diff)


class Pinball:
    """Pinball environment adhering to the ``GymEnv`` protocol.

    :param config: A bundled config name (e.g. ``"box"``, ``"empty"``,
        ``"easy"``, ``"medium"``, ``"hard"``) or a path to a ``.cfg`` file.
    """

    def __init__(self, config: str | Path) -> None:
        obstacles, target_pos, target_rad, start_pts, ball_rad = _parse_config(
            _read_config(config)
        )
        edge_p0, edge_p1, edge_mask, bbox_min, bbox_max = _bake_geometry(obstacles)

        self.config = str(config)
        # Baked JAX constants (float32 by default, float64 when x64 is enabled).
        self.edge_p0 = jnp.asarray(edge_p0)
        self.edge_p1 = jnp.asarray(edge_p1)
        self.edge_mask = jnp.asarray(edge_mask)
        self.bbox_min = jnp.asarray(bbox_min)
        self.bbox_max = jnp.asarray(bbox_max)
        self.ball_rad = jnp.asarray(ball_rad, dtype=self.edge_p0.dtype)
        self.target = jnp.asarray(target_pos, dtype=self.edge_p0.dtype)
        self.target_rad = jnp.asarray(target_rad, dtype=self.edge_p0.dtype)
        self.start_pts = jnp.asarray(start_pts, dtype=self.edge_p0.dtype)

    # -- spaces ------------------------------------------------------------- #

    def observation_space(self, params: PinballParams | None = None) -> ObservationSpace:
        del params
        return _PinballObservationSpace()

    def action_space(self, params: PinballParams | None = None) -> DiscreteActionSpace:
        del params
        return _PinballActionSpace()

    # -- collision physics -------------------------------------------------- #

    def _resolve_collision(self, pos: jax.Array, vel: jax.Array) -> tuple[jax.Array, jax.Array]:
        """One collision-resolution pass at ``pos`` with velocity ``vel``.

        Returns ``(new_vel[2], ncollision)`` where ``ncollision`` is the number
        of obstacles the ball collides with this substep.
        """
        r = self.ball_rad

        # Per-edge intercept test, vmapped over [O, E].
        edge = self.edge_p1 - self.edge_p0            # [O, E, 2]
        diff = pos - self.edge_p0                     # [O, E, 2]
        denom = jnp.sum(edge * edge, axis=-1)         # [O, E]
        safe_denom = jnp.where(denom > 0, denom, 1.0)
        scalar_proj = jnp.clip(jnp.sum(diff * edge, axis=-1) / safe_denom, 0.0, 1.0)
        closest = self.edge_p0 + edge * scalar_proj[..., None]  # [O, E, 2]
        dist2 = jnp.sum((pos - closest) ** 2, axis=-1)          # [O, E]
        within = dist2 <= r * r

        ball_to_obstacle = closest - pos              # [O, E, 2]
        vel_b = jnp.broadcast_to(vel, ball_to_obstacle.shape)
        angle = _angle(ball_to_obstacle, vel_b)       # [O, E]
        angle = jnp.where(angle > jnp.pi, 2 * jnp.pi - angle, angle)
        not_moving_away = angle <= jnp.pi / 1.99

        # Bounding-box rejection, per obstacle.
        bbox_pass = (
            (pos[0] - r <= self.bbox_max[:, 0])
            & (pos[0] + r >= self.bbox_min[:, 0])
            & (pos[1] - r <= self.bbox_max[:, 1])
            & (pos[1] + r >= self.bbox_min[:, 1])
        )  # [O]

        edge_hit = within & not_moving_away & self.edge_mask & bbox_pass[:, None]

        # Per-obstacle aggregation.
        count = jnp.sum(edge_hit, axis=1)             # [O]
        collided = count >= 1
        double = count >= 2
        sel = jnp.argmax(edge_hit, axis=1)            # [O] first hit edge (0 if none)
        sel_p0 = jnp.take_along_axis(self.edge_p0, sel[:, None, None], axis=1)[:, 0, :]
        sel_p1 = jnp.take_along_axis(self.edge_p1, sel[:, None, None], axis=1)[:, 0, :]

        # Reflection effect (non-corner branch), per obstacle.
        obstacle_vec = sel_p1 - sel_p0                # [O, 2]
        obstacle_vec = jnp.where((obstacle_vec[:, 0] < 0)[:, None], -obstacle_vec, obstacle_vec)
        theta = _angle(jnp.broadcast_to(vel, obstacle_vec.shape), obstacle_vec) - jnp.pi
        theta = jnp.where(theta < 0, theta + 2 * jnp.pi, theta)
        neg_x = jnp.broadcast_to(jnp.asarray([-1.0, 0.0], dtype=obstacle_vec.dtype), obstacle_vec.shape)
        theta = theta + _angle(neg_x, obstacle_vec)
        theta = jnp.where(theta > 2 * jnp.pi, theta - 2 * jnp.pi, theta)
        speed = jnp.sqrt(vel[0] ** 2 + vel[1] ** 2)
        reflect = jnp.stack([speed * jnp.cos(theta), speed * jnp.sin(theta)], axis=-1)  # [O, 2]

        # Corner hit within a single obstacle negates the velocity.
        effect = jnp.where(double[:, None], -vel[None, :], reflect)  # [O, 2]

        # Cross-obstacle resolution.
        ncollision = jnp.sum(collided)
        dxdy = jnp.sum(jnp.where(collided[:, None], effect, 0.0), axis=0)  # [2]
        new_vel = jnp.where(
            ncollision == 1,
            dxdy,
            jnp.where(ncollision > 1, -vel, vel),
        )
        return new_vel, ncollision

    def _take_action(self, state: PinballState, action: jax.Array) -> PinballState:
        """Run one action (``SUBSTEPS`` physics substeps) from ``state``."""
        r = self.ball_rad
        impulse = ACTION_EFFECTS[action].astype(r.dtype)  # [2]

        def body(i, carry):
            x, y, xdot, ydot, done = carry
            was_done = done

            # Impulse (clipped) only on the first substep.
            imp_x = jnp.clip(xdot + impulse[0] / 5.0, -1.0, 1.0)
            imp_y = jnp.clip(ydot + impulse[1] / 5.0, -1.0, 1.0)
            xdot_i = jnp.where(i == 0, imp_x, xdot)
            ydot_i = jnp.where(i == 0, imp_y, ydot)

            # Move the ball by one increment.
            x1 = x + xdot_i * r / 20.0
            y1 = y + ydot_i * r / 20.0

            new_vel, ncollision = self._resolve_collision(
                jnp.stack([x1, y1]), jnp.stack([xdot_i, ydot_i])
            )
            nvx, nvy = new_vel[0], new_vel[1]

            # On the final substep, a single collision triggers an extra move.
            extra = (i == SUBSTEPS - 1) & (ncollision == 1)
            x2 = jnp.where(extra, x1 + nvx * r / 20.0, x1)
            y2 = jnp.where(extra, y1 + nvy * r / 20.0, y1)

            ended = jnp.sqrt((x2 - self.target[0]) ** 2 + (y2 - self.target[1]) ** 2) < self.target_rad

            # Freeze all updates once the episode has ended.
            x_out = jnp.where(was_done, x, x2)
            y_out = jnp.where(was_done, y, y2)
            vx_out = jnp.where(was_done, xdot, nvx)
            vy_out = jnp.where(was_done, ydot, nvy)
            done_out = was_done | ended
            return (x_out, y_out, vx_out, vy_out, done_out)

        x, y, xdot, ydot, done = jax.lax.fori_loop(
            0, SUBSTEPS, body, (state.x, state.y, state.xdot, state.ydot, jnp.asarray(False))
        )

        # Drag and boundary clamping, skipped if the episode ended (the
        # reference early-returns before applying them).
        xdot = jnp.where(done, xdot, xdot * DRAG)
        ydot = jnp.where(done, ydot, ydot * DRAG)
        bx = jnp.where(x > 1.0, 0.95, jnp.where(x < 0.0, 0.05, x))
        by = jnp.where(y > 1.0, 0.95, jnp.where(y < 0.0, 0.05, y))
        x = jnp.where(done, x, bx)
        y = jnp.where(done, y, by)

        return PinballState(x=x, y=y, xdot=xdot, ydot=ydot, timestep=state.timestep)

    # -- protocol ----------------------------------------------------------- #

    def reset(
        self,
        key: jax.Array,
        params: PinballParams | None = None,
    ) -> tuple[jax.Array, PinballState]:
        del params
        idx = jax.random.randint(key, (), 0, self.start_pts.shape[0])
        start = self.start_pts[idx]
        x, y = start[0], start[1]
        zero = jnp.zeros((), dtype=self.start_pts.dtype)
        state = PinballState(
            x=x, y=y, xdot=zero, ydot=zero, timestep=jnp.zeros((), dtype=jnp.int32)
        )
        obs = jnp.stack([x, y, zero, zero]).astype(jnp.float32)
        return obs, state

    def step(
        self,
        key: jax.Array,
        state: PinballState,
        action: jax.Array,
        params: PinballParams | None = None,
    ) -> tuple[jax.Array, PinballState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        del key
        params = params if params is not None else PinballParams()

        moved = self._take_action(state, action)
        timestep = state.timestep + 1
        next_state = moved._replace(timestep=timestep)

        obs = jnp.stack([moved.x, moved.y, moved.xdot, moved.ydot]).astype(jnp.float32)
        reward = jnp.asarray(-1.0, dtype=jnp.float32)
        distance = jnp.sqrt((moved.x - self.target[0]) ** 2 + (moved.y - self.target[1]) ** 2)
        terminated = distance < self.target_rad
        truncated = timestep >= params.max_steps_in_episode
        info: dict[str, jax.Array] = {}

        return obs, next_state, reward, terminated, truncated, info
