#config.py

from dataclasses import dataclass, field
import numpy as np

# --------------------------------------------------------------------------- #
# Field geometry                                                               #
# --------------------------------------------------------------------------- #
FIELD_W = 800.0
FIELD_H = 500.0
FIELD_DIAG = float(np.hypot(FIELD_W, FIELD_H))

# Two grid resolutions used only for the observation discretisation.
# Convention: (cols -> x axis, rows -> y axis)
FINE_COLS, FINE_ROWS = 30, 20      # 20x30 grid  (ball + own position)
COARSE_COLS, COARSE_ROWS = 12, 6   # 6x12 grid   (team-mates + opponents)

# --------------------------------------------------------------------------- #
# Goals                                                                        #
# --------------------------------------------------------------------------- #
GOAL_HEIGHT = 120# pre 160                                 # mouth height in pixels
GOAL_Y_MIN = FIELD_H / 2.0 - GOAL_HEIGHT / 2.0       # 170
GOAL_Y_MAX = FIELD_H / 2.0 + GOAL_HEIGHT / 2.0       # 330
GOAL_CENTER_Y = FIELD_H / 2.0

LEFT_GOAL_X = 0.0          # red's goal  -> blue scores here
RIGHT_GOAL_X = FIELD_W     # blue's goal -> red scores here
# goal post


VIBRATION_RADIUS = 15 
# ------------------------------
GAMMA = 0.999 # discount factor

# --------------------------------------------------------------------------- #
# Players                                                                      #
# --------------------------------------------------------------------------- #
# Internal player indices (fixed ordering used everywhere):
#   0,1,2,3 -> blue field players  (the RL agents)
#   4       -> blue goalkeeper      (automatic)
#   5,6,7,8 -> red field players    (rule-based)
#   9       -> red goalkeeper        (automatic)
N_PLAYERS = 10
BLUE_FIELD = (0, 1, 2, 3)
BLUE_GK    = 4
RED_FIELD  = (5, 6, 7, 8)
RED_GK     = 9
BLUE_ALL   = (0, 1, 2, 3, 4)
RED_ALL    = (5, 6, 7, 8, 9)
GK_INDICES = (BLUE_GK, RED_GK)

# Goalkeeper x lines (never leave the goal line).
BLUE_GK_X = 780.0
RED_GK_X = 20.0
GK_Y_MIN = GOAL_Y_MIN
GK_Y_MAX = GOAL_Y_MAX

PLAYER_RADIUS = 10.0
PLAYER_SPEED = 5        # pixels / step for a field player move action
GK_SPEED = PLAYER_SPEED * 1.1  # pixels / step the keeper can shift in y
DIAG = 1.0 / np.sqrt(2.0)      # diagonal move scaling so speed stays constant

# ----------- Red difficulty curriculum ------------------------------------ #
RED_SPEED_START   = 0.3   # red moves at 30% of player speed at episode 1
RED_SPEED_END     = 1.0   # red reaches full speed at RED_CURRICULUM_EPISODES
RED_CURRICULUM_EP = 1000  # how many episodes to ramp over
#--------------Stamina---------------------------------------------------
STAMINA_DRAIN     = 0.04
STAMINA_RECOVER   = 0.04
STAMINA_SPEED_MIN = 0.4
STAMINA_MAX       = 1.0
# --------------------------------------------------------------------------- #
# Ball physics                                                                 #
# --------------------------------------------------------------------------- #
FRICTION = 0.95            # vx,vy multiplied by this every step while free
STOP_SPEED = 3             # if |vx|+|vy| < this the ball stops
SHOT_SPEED = 20.0          # initial speed of a shot
PASS_SPEED = 15.0          # initial speed of a pass
SHOT_Y_SPREAD = 50.0       # +/- random pixels added to the aimed y of a shot
BALL_RADIUS = 5.0

# --------------------------------------------------------------------------- #
# Capturing / collisions                                                       #
# --------------------------------------------------------------------------- #
CAPTURE_RADIUS = 20.0          # a player within this distance can grab the ball
CAPTURE_MOVING_SPEED = 20      # a *moving* free ball is grabbable below this speed
GK_CAPTURE_MULT = 1.3          # keepers may grab a ball moving up to 1.3x as fast
KICK_COOLDOWN_STEPS = 5        # steps the kicker cannot re-capture its own kick
#-----------------------------------------------------------------------------
#vision
#-----------------------------------------------------------------------------

OPP_VISION_RADIUS = 100.0   # opponents beyond this many pixels aren't observed in detail
# --------------------------------------------------------------------------- #
# Actions                                                                      #
# --------------------------------------------------------------------------- #
# Discrete action ids for a field player.
# --------------------------------------------------------------------------- #
# Actions                                                                      #
# --------------------------------------------------------------------------- #
# Discrete action ids for a field player.
A_UP         = 0
A_DOWN       = 1
A_LEFT       = 2
A_RIGHT      = 3
A_UP_LEFT    = 4
A_UP_RIGHT   = 5
A_DOWN_LEFT  = 6
A_DOWN_RIGHT = 7
A_SHOOT      = 8
A_STAY       = 9

# Relative passing actions
A_PASS_A     = 10   # pass to mates[0]
A_PASS_B     = 11   # pass to mates[1]
A_PASS_C     = 12   # pass to mates[2]

N_ACTIONS    = 13

MOVE_ACTIONS = (A_UP, A_DOWN, A_LEFT, A_RIGHT,
                A_UP_LEFT, A_UP_RIGHT, A_DOWN_LEFT, A_DOWN_RIGHT)
PASS_ACTIONS = (A_PASS_A, A_PASS_B, A_PASS_C)


# (dx, dy) unit move vectors. Note: +y points DOWN (pygame convention).
MOVE_VECTORS = {
    A_UP:         (0.0,  -1.0),
    A_DOWN:       (0.0,   1.0),
    A_LEFT:       (-1.0,  0.0),
    A_RIGHT:      (1.0,   0.0),
    A_UP_LEFT:    (-DIAG, -DIAG),
    A_UP_RIGHT:   ( DIAG, -DIAG),
    A_DOWN_LEFT:  (-DIAG,  DIAG),
    A_DOWN_RIGHT: ( DIAG,  DIAG),
    A_STAY:       (0.0,   0.0),
}

# --------------------------------------------------------------------------- #
# Observation                                                                  #
# --------------------------------------------------------------------------- #
# Layout (length 43):
#   [0]       I have the ball
#   [1:3]     ball position   fine-grid  remapped [-1,1]
#   [3:5]     ball velocity   normalised by SHOT_SPEED, clamped [-1,1]
#   [5:7]     self position   fine-grid  remapped [-1,1]
#   [7]       self combined stamina factor [0.0, 1.0]
#   [8]       teammate A has the ball
#   [9:11]    teammate A position  coarse-grid remapped [-1,1]
#   [11]      teammate A combined stamina factor
#   [12]      teammate B has the ball
#   [13:15]   teammate B position  coarse-grid remapped [-1,1]
#   [15]      teammate B combined stamina factor
#   [16]      teammate C has the ball
#   [17:19]   teammate C position  coarse-grid remapped [-1,1]
#   [19]      teammate C combined stamina factor
#   [20]      red field 0 has the ball
#   [21:23]   red field 0 position coarse-grid remapped [-1,1]
#   [23]      red field 0 combined stamina factor
#   [24]      red field 1 has the ball
#   [25:27]   red field 1 position coarse-grid remapped [-1,1]
#   [27]      red field 1 combined stamina factor
#   [28]      red field 2 has the ball
#   [29:31]   red field 2 position coarse-grid remapped [-1,1]
#   [31]      red field 2 combined stamina factor
#   [32]      red field 3 has the ball
#   [33:35]   red field 3 position coarse-grid remapped [-1,1]
#   [35]      red field 3 combined stamina factor
#   [36]      ball is loose
#   [37]      score difference normalised [-1,1]
#   [38]      time fraction [0,1]
#   [39:43]   agent id one-hot vector (length 4)

#speed of each agent 8  x , y =16  [43:61]
# 3 tactic obs
OBS_DIM = 69      #aware of tactic
#5 for array that tells blue if we see the red or not
SCORE_NORM = 20.0  # score diff divided by this then clipped to [-1, 1]  #pre 10

# --------------------------------------------------------------------------- #
# Episode                                                                      #
# --------------------------------------------------------------------------- #
STEPS_PER_SECOND = 30
EPISODE_SECONDS = 120          # 2 minutes
MAX_STEPS = STEPS_PER_SECOND * EPISODE_SECONDS    # 3600

# --------------------------------------------------------------------------- #
# Reward shaping                                                               #
# --------------------------------------------------------------------------- #
GOAL_TOTAL     =  2.6
REWARD_BALL_C  =  0.0
REWARD_CHASE_C =  0.00
REWARD_TURNOVER_C = 0.0
# --------------------------------------------------------------------------- #
# Initial formations (continuous coords) at kickoff                            #
# --------------------------------------------------------------------------- #
BLUE_FORMATION = np.array([
    [560.0, 400.0],
    [660.0, 250.0],
    [560.0, 100.0],
    [460.0, 250.0],
], dtype=np.float64)

RED_FORMATION = np.array([
    [240.0, 400.0],
    [140.0, 250.0],
    [240.0, 100.0],
    [340.0, 250.0],
], dtype=np.float64)

BLUE_GK_START = np.array([BLUE_GK_X, GOAL_CENTER_Y], dtype=np.float64)
RED_GK_START  = np.array([RED_GK_X,  GOAL_CENTER_Y], dtype=np.float64)
BALL_START    = np.array([FIELD_W / 2.0, FIELD_H / 2.0], dtype=np.float64)

# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #
RENDER_FPS    = STEPS_PER_SECOND
COLOR_BG      = (28, 120, 55)
COLOR_LINE    = (235, 235, 235)
COLOR_BLUE    = (40, 110, 230)
COLOR_RED     = (220, 60, 60)
COLOR_GK_BLUE = (15, 60, 150)
COLOR_GK_RED  = (150, 30, 30)
COLOR_BALL    = (250, 245, 230)
COLOR_TEXT    = (255, 255, 255)


@dataclass(frozen=True)
class Config:
    field_w:               float = FIELD_W
    field_h:               float = FIELD_H
    max_steps:             int   = MAX_STEPS
    player_speed:          float = PLAYER_SPEED
    gk_speed:              float = GK_SPEED
    friction:              float = FRICTION
    stop_speed:            float = STOP_SPEED
    shot_speed:            float = SHOT_SPEED
    pass_speed:            float = PASS_SPEED
    shot_y_spread:         float = SHOT_Y_SPREAD
    capture_radius:        float = CAPTURE_RADIUS
    capture_moving_speed:  float = CAPTURE_MOVING_SPEED
    gk_capture_mult:       float = GK_CAPTURE_MULT
    kick_cooldown:         int   = KICK_COOLDOWN_STEPS
    reward_ball_c:         float = REWARD_BALL_C
    reward_chase_c:        float = REWARD_CHASE_C
    reward_turnover_c:     float = REWARD_TURNOVER_C
    goal_total:            float = GOAL_TOTAL
    blue_formation: np.ndarray = field(default_factory=lambda: BLUE_FORMATION.copy())
    red_formation:  np.ndarray = field(default_factory=lambda: RED_FORMATION.copy())
    gamma:                 float = GAMMA

DEFAULT_CONFIG = Config()