"""
helper.py
=========
Pure functions and small stateful helpers used by the environment, the trainer
and the logger.  Everything here is numpy-only (no gym / pettingzoo / pygame),
so it is cheap to import and easy to unit-test.

Groups of helpers:
  * geometry        : distances, clamping, grid discretisation / normalisation
  * football logic  : nearest player, shot-on-target test
  * statistics      : PossessionTracker
  * reward shaping  : potential-based reward components
"""

from __future__ import annotations  #Forward Reference
import numpy as np
import config as C


# --------------------------------------------------------------------------- #
# Geometry                                                                     #
# --------------------------------------------------------------------------- #
def clamp(v, lo, hi):   # Limit ground to 500 y  800 x   lo:bottom :hi top
    """Clamp a scalar (or array) to the [lo, hi] range."""
    return np.minimum(np.maximum(v, lo), hi)


def distance(a, b) -> float:
    """Euclidean distance between two 2-D points."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def speed(vel) -> float:
    """Euclidean magnitude of a velocity vector."""
    vel = np.asarray(vel, dtype=np.float64)
    return float(np.hypot(vel[0], vel[1]))   #sqrt(x^2 + y^2)


def grid_cell(x, y, cols, rows,
              field_w=C.FIELD_W, field_h=C.FIELD_H):
    """Map a continuous (x, y) to an integer grid cell (cx, cy)."""
    cx = int(x / field_w * cols)
    cy = int(y / field_h * rows)
    cx = int(clamp(cx, 0, cols - 1))
    cy = int(clamp(cy, 0, rows - 1))
    return cx, cy


def grid_norm(x, y, cols, rows,
              field_w=C.FIELD_W, field_h=C.FIELD_H):
    """Discretise to a grid cell then normalise the cell index to [0, 1]."""
    cx, cy = grid_cell(x, y, cols, rows, field_w, field_h)
    nx = cx / (cols - 1) if cols > 1 else 0.0
    ny = cy / (rows - 1) if rows > 1 else 0.0
    return nx, ny


def fine_norm(pos):
    """Normalised position on the fine (20x30) grid."""
    return grid_norm(pos[0], pos[1], C.FINE_COLS, C.FINE_ROWS)


def coarse_norm(pos):
    """Normalised position on the coarse (6x12) grid."""
    return grid_norm(pos[0], pos[1], C.COARSE_COLS, C.COARSE_ROWS)


def reflect_into_bounds(pos, vel, field_w=C.FIELD_W, field_h=C.FIELD_H,
                        radius=C.BALL_RADIUS):
    """
    Reflect the y component off the top / bottom walls and keep the point
    inside the field vertically.  Returns (pos, vel) (modified copies).
    Left / right handling (posts vs goal) is done by the engine because it
    needs the goal geometry.
    """
    pos = np.array(pos, dtype=np.float64)
    vel = np.array(vel, dtype=np.float64)
    if pos[1] <= radius:
        pos[1] = radius
        vel[1] = -vel[1]
    elif pos[1] >= field_h - radius:
        pos[1] = field_h - radius
        vel[1] = -vel[1]
    return pos, vel


# --------------------------------------------------------------------------- #
# Football logic                                                               #
# --------------------------------------------------------------------------- #
def nearest_index(point, positions, candidate_indices):
    """
    Return the index (from candidate_indices) of the player closest to `point`.
    Returns None if there are no candidates.
    """
    best, best_d = None, float("inf")
    for i in candidate_indices:
        d = distance(point, positions[i])
        if d < best_d:
            best_d, best = d, i
    return best



def shot_velocity(ball_pos, goal_x, goal_center_y, rng,
                  shot_speed=C.SHOT_SPEED, y_spread=C.SHOT_Y_SPREAD):
    """
    Velocity vector for a shot aimed at the opponent goal with a random y
    offset (so it is not always dead-centre).
    """
    target_y = goal_center_y + rng.uniform(-y_spread, y_spread)
    direction = np.array([goal_x - ball_pos[0], target_y - ball_pos[1]],
                         dtype=np.float64)
    n = np.linalg.norm(direction)
    if n < 1e-6:
        direction = np.array([np.sign(goal_x - ball_pos[0]) or 1.0, 0.0])
        n = 1.0
    return direction / n * shot_speed


def pass_velocity(from_pos, to_pos, pass_speed=C.PASS_SPEED):
    """Velocity vector for a pass from one point toward another."""
    direction = np.array([to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]],
                         dtype=np.float64)
    n = np.linalg.norm(direction)
    if n < 1e-6:
        return np.zeros(2, dtype=np.float64)
    return direction / n * pass_speed


# --------------------------------------------------------------------------- #
# Statistics                                                                   #
# --------------------------------------------------------------------------- #
class PossessionTracker:
    """Counts the number of steps each team owned the ball and reports %."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.blue_steps = 0
        self.red_steps = 0

    def update(self, owner, blue_all=C.BLUE_ALL, red_all=C.RED_ALL):
        """`owner` is a player index or -1 (free)."""
        if owner in blue_all:
            self.blue_steps += 1
        elif owner in red_all:
            self.red_steps += 1

    def percent(self):
        """Return (blue_percent, red_percent). Free-ball steps are excluded."""
        total = self.blue_steps + self.red_steps
        if total == 0:
            return 50.0, 50.0
        b = 100.0 * self.blue_steps / total
        return b, 100.0 - b

#----------------------------------------------
#------------limit vertical movement-----------
#----------------------------------------------
# ── Y-range limits per role ───────────────────────────────────────────────
Y_LIMITS_BY_SLOT = [
    (300, 500),   # slot 0 — up agent
    (150, 350),   # slot 1 — middle agent
    (0,   200),   # slot 2 — down agent
    (100, 400),   # slot 3 — central agent
]
def get_y_limits(idx):
    """Return (y_min, y_max) for vertical clamping based on player role."""
    import config as C
    blue_field = list(C.BLUE_FIELD)
    
    # Only enforce strict physical lanes for the Blue team
    if idx in blue_field:
        slot = blue_field.index(idx)
        return Y_LIMITS_BY_SLOT[slot]
        
    # For Red players and Goalkeepers, return the full field limits.
    # red_control.py will now handle Red's specific limits.
    return C.PLAYER_RADIUS, C.FIELD_H - C.PLAYER_RADIUS