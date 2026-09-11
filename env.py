"""
env.py
======
The 4v4 grid-football simulation.

Two layers:
  * FootballEngine  -- pure numpy physics + game engine.  Holds *all* state,
                       advances one tick from (blue_field_actions,
                       red_field_actions), runs the goalkeepers automatically,
                       and exposes everything needed to build observations,
                       rewards and a render.  No gym/pettingzoo/pygame imports,
                       so it is unit-testable on its own.
  * GridFootballEnv -- a PettingZoo ParallelEnv wrapping the engine.  Agents are
                       the three blue field players ("blue_0/1/2").  The blue GK
                       is automatic, the red team is rule-based (red_control),
                       both run inside the engine.

The env is deliberately self-contained and side-effect free per instance, so
many copies can be created and stepped in parallel for training.



"""

from __future__ import annotations
import numpy as np
import collections
from reward import RewardCalculator
import config as C
import helper as H



#----------------------------------------------------------------------------
class AssistTracker:
    def __init__(self):
        self._last = None
        self._pass_from = None

    def reset(self):
        self._last = None
        self._pass_from = None

    def on_touch(self, player_idx: int, is_blue_field: bool):
        """Non-pass touch: pickup, rebound, tackle, red touch."""
        if not is_blue_field:
            self._last = None
            self._pass_from = None
            return
        if player_idx == self._last:
            return
        self._last = player_idx
        self._pass_from = None      # got ball, but NOT via a pass

    def on_pass_touch(self, passer_idx: int, receiver_idx: int):
        """Only called for a completed blue field-to-field pass."""
        self._last = receiver_idx
        self._pass_from = passer_idx

    def on_goal(self) -> tuple[int | None, int | None]:
        scorer   = self._last
        assister = self._pass_from if self._pass_from != scorer else None
        return scorer, assister
# =========================================================================== 
#  ENGINE                                                                      
# =========================================================================== 
class FootballEngine:
    """Pure-numpy physics and rules engine for one match."""

    def __init__(self, config: C.Config = C.DEFAULT_CONFIG, seed=None):
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        # ADD THIS: Create a detached, instance-specific copy of the red formation
        self.local_red_formation = self.cfg.red_formation.copy()

        # --- dynamic state (allocated in reset) ---
        self.pos = np.zeros((C.N_PLAYERS, 2), dtype=np.float64)
        self.stamina = np.ones(C.N_PLAYERS, dtype=np.float64)
        self.distance_run = np.zeros(C.N_PLAYERS, dtype=np.float64)  # for csv run hist
        #self.pos_history = [collections.deque(maxlen=10) for _ in range(C.N_PLAYERS)]
        self.window_start_pos = self.pos.copy()
        self.move_count = np.zeros(C.N_PLAYERS, dtype=np.int32)
        
        self.ball_pos = np.zeros(2, dtype=np.float64)
        self.ball_vel = np.zeros(2, dtype=np.float64)
        self.player_vel = np.zeros((C.N_PLAYERS, 2), dtype=np.float64)   # track velocity
        self.owner = -1                 # -1 free, else player index
        self.steps = 0
        self.red_tactic = "A"
        # --- scoreboard / stats ---
        self.blue_goals = 0
        self.red_goals = 0
        self.blue_shots = 0
        self.blue_sot = 0
        self.red_shots = 0
        self.red_sot = 0
        self.poss = H.PossessionTracker()

        # --- kick bookkeeping ---
        self.last_kicker = -1
        self.kick_cd = 0
        self.last_kick_was_pass = False
        # --- pass stats ---
        self.blue_passes = 0
        self.blue_passes_success = 0
        self.blue_assists = 0
        self.red_passes  = 0
        self.red_passes_success  = 0
        # --- shot-in-flight tracking (outcome-based SoT) ---
        self.shot_in_flight = False
        self.shot_shooter   = -1
        self.shot_team      = None
        self.rew = RewardCalculator(config)
        self.assist_tracker = AssistTracker()      
        self.last_blue_owner = -1        #who lose the ball
        self.reset()


    # ------------------------------------------------------------------ #
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.red_tactic = "A"
        self.blue_goals = self.red_goals = 0
        self.blue_shots = self.blue_sot = 0
        self.red_shots  = self.red_sot  = 0
        self.blue_passes = self.blue_passes_success = 0
        self.blue_assists = 0  #added for p blue ass
        self.red_passes  = self.red_passes_success  = 0
        self.distance_run[:] = 0.0  # run hist csv
        self.poss.reset()
        self.rew.reset()

        # ADD THIS: Dictionary to track pass edges (passer, receiver) -> count
        self.pass_edges = collections.defaultdict(int)

        self._kickoff()
        return
#---------------------------------------------------------
    def get_combined_stamina_mult(self, idx: int) -> float:
        dist_ratio = min(self.distance_run[idx] / 15000.0, 1.0)
        total_stamina_speed_mult = 1.0 - (0.6 * dist_ratio)
        short_term_stamina_mult = C.STAMINA_SPEED_MIN + (1.0 - C.STAMINA_SPEED_MIN) * self.stamina[idx]
        return float(total_stamina_speed_mult * short_term_stamina_mult)
#---------------------------------------------------------
    def _kickoff(self):
        """Place everyone in their starting formation, ball dead in the centre."""
        self.pos[list(C.BLUE_FIELD)] = self.cfg.blue_formation
        self.pos[C.BLUE_GK] = C.BLUE_GK_START

        self.pos[list(C.RED_FIELD)] = self.local_red_formation

        self.pos[C.RED_GK] = C.RED_GK_START
        self.ball_pos[:] = C.BALL_START
        self.ball_vel[:] = 0.0
        self.player_vel[:] = 0.0

        self.stamina[:] = C.STAMINA_MAX
        self.window_start_pos = self.pos.copy()
        self.move_count[:] = 0


        self.owner = -1
        self.last_kicker = -1
        self.kick_cd = 0
        self.shot_in_flight    = False
        self.shot_shooter      = -1
        self.shot_team         = None
        self.last_kick_was_pass = False
        self.rew.refresh_prev_distances(self.ball_pos, self.pos)
        self.assist_tracker.reset() 
        self.last_blue_owner = -1
        self.turnover_from_pass = False  # pass cause lose -no negative rew
        self.last_pass_pair = (-1, -1)

    # ------------------------------------------------------------------ #
    #  Player movement                                                    #
    # ------------------------------------------------------------------ #
    def _move_field_player(self, idx, action):
        if action in C.MOVE_VECTORS:
            # 1. Save starting position
            old_x, old_y = self.pos[idx, 0], self.pos[idx, 1]
            dx, dy = C.MOVE_VECTORS[action]

            # 2. Calculate Stamina Multipliers
            # Total game stamina multiplier (goes from 1.0 down to 0.4 based on distance)
            dist_ratio = min(self.distance_run[idx] / 15000.0, 1.0)
            total_stamina_speed_mult = 1.0 - (0.6 * dist_ratio)  
            
            # Short-term stamina multiplier (burst/recovery)
            short_term_stamina_mult = C.STAMINA_SPEED_MIN + (1.0 - C.STAMINA_SPEED_MIN) * self.stamina[idx]
            
            # --- HIERARCHY SPEED LOGIC ---
            # Normal speed is the player's base speed adjusted ONLY by their total match fatigue
            normal_speed = self.cfg.player_speed * total_stamina_speed_mult
            
            # Present speed is the normal speed modified by their immediate, short-term breath/stamina
            present_speed = normal_speed * short_term_stamina_mult
            
            # Reduce speed if the player is currently dribbling the ball
            if self.owner == idx:
                present_speed *= 0.75  
            
            # 3. Apply raw movement using present_speed
            self.pos[idx, 0] += dx * present_speed
            self.pos[idx, 1] += dy * present_speed
            
            # 4. Apply boundaries (clamping)
            self.pos[idx, 0] = H.clamp(self.pos[idx, 0], C.PLAYER_RADIUS, C.FIELD_W - C.PLAYER_RADIUS)
            y_lo, y_hi = H.get_y_limits(idx)
            self.pos[idx, 1] = H.clamp(self.pos[idx, 1], y_lo, y_hi)
            
            # 5. Apply goal exclusion zone
            if self._in_goal_exclusion_zone(idx, self.pos[idx, 0], self.pos[idx, 1]):
                self.pos[idx, 0] = old_x
                self.pos[idx, 1] = old_y

            # 6. Calculate actual distance run using the final, validated position
            self.distance_run[idx] += H.distance(self.pos[idx], [old_x, old_y])
#-------------------------------------------------------------------        
    def _move_goalkeeper(self, gk_idx, gk_x):
        """Keeper tracks the ball in y only, clamped to the goal mouth."""
        target_y = H.clamp(self.ball_pos[1], C.GK_Y_MIN, C.GK_Y_MAX)
        diff = target_y - self.pos[gk_idx, 1]
        step = float(H.clamp(diff, -self.cfg.gk_speed, self.cfg.gk_speed))
        self.pos[gk_idx, 1] += step
        self.pos[gk_idx, 0] = gk_x
        self.pos[gk_idx, 1] = H.clamp(self.pos[gk_idx, 1], C.GK_Y_MIN, C.GK_Y_MAX)

    def _in_goal_exclusion_zone(self, idx, x, y, margin=40):
        """Returns True if (x, y) is within `margin` of either goal rectangle."""
        in_y = (C.GOAL_Y_MIN - margin) <= y <= (C.GOAL_Y_MAX + margin)
        near_left  = x <= (C.LEFT_GOAL_X  + margin) and in_y
        near_right = x >= (C.RIGHT_GOAL_X - margin) and in_y
        return near_left or near_right
    # ------------------------------------------------------------------ #
    #  Shooting & passing                                                 #
    # ------------------------------------------------------------------ #
    def _field_teammates(self, idx):
        """Three field team-mates of `idx`, sorted (used for the three pass actions)."""
        group = C.BLUE_FIELD if idx in C.BLUE_FIELD else C.RED_FIELD
        return [p for p in group if p != idx]

    def _do_shot(self, idx):
        attack_goal_x = C.LEFT_GOAL_X if idx in C.BLUE_ALL else C.RIGHT_GOAL_X
        
        # Calculate distance from the ball to the center of the target goal
        dist_to_goal = H.distance(self.ball_pos, [attack_goal_x, C.GOAL_CENTER_Y])
        
        # Define our minimum and maximum variance
        min_spread = self.cfg.shot_y_spread  # 50.0
        max_spread = C.FIELD_H / 2.0         # 250.0 (covers the whole 500 height)
        
        # Calculate the dynamic spread based on distance
        if dist_to_goal <= 200.0:
            active_spread = min_spread
        elif dist_to_goal >= 600.0:
            active_spread = max_spread
        else:
            # Smoothly scale the spread for distances between 200 and 600
            # Formula: min_spread + fraction * (max_spread - min_spread)
            fraction = (dist_to_goal - 200.0) / (600.0 - 200.0)
            active_spread = min_spread + fraction * (max_spread - min_spread)
            
        vel = H.shot_velocity(self.ball_pos, attack_goal_x, C.GOAL_CENTER_Y,
                              self.rng, self.cfg.shot_speed, active_spread)
        self.ball_vel[:] = vel
#----------------------------------------------------
        
        self.owner = -1
        self.last_kicker = idx
        self.kick_cd = self.cfg.kick_cooldown
        self.shot_in_flight    = True
        self.shot_shooter      = idx
        self.shot_team         = 'blue' if idx in C.BLUE_ALL else 'red'
        self.last_kick_was_pass = False
        self.turnover_from_pass = False  # no turnover from pass
        if idx in C.BLUE_ALL:
            self.blue_shots += 1
        else:
            self.red_shots += 1

    def _do_pass(self, idx, target_idx):
        vel = H.pass_velocity(self.pos[idx], self.pos[target_idx],
                              self.cfg.pass_speed)
        self.ball_vel[:] = vel
        self.owner = -1
        self.last_kicker = idx
        self.kick_cd = self.cfg.kick_cooldown
        self.turnover_from_pass = True 
        self.last_kick_was_pass = (idx not in C.GK_INDICES)   # GK distribution is not a "pass"
        if idx in C.BLUE_FIELD:
            self.blue_passes += 1
        elif idx in C.RED_FIELD:
            self.red_passes += 1

    # ------------------------------------------------------------------ #
    #  Ball physics                                                       #
    # ------------------------------------------------------------------ #
    def _update_ball(self):
        """Advance a free ball: integrate, apply friction, resolve walls/posts.
        Returns 'blue', 'red' or None depending on whether a goal was scored."""
        self.ball_pos += self.ball_vel
        self.ball_vel *= self.cfg.friction
        if abs(self.ball_vel[0]) + abs(self.ball_vel[1]) < self.cfg.stop_speed:
            self.ball_vel[:] = 0.0

        # top / bottom walls
        self.ball_pos, self.ball_vel = H.reflect_into_bounds(
            self.ball_pos, self.ball_vel)

        scorer = None
        # left side (red goal -> blue scores)
        if self.ball_pos[0] <= C.LEFT_GOAL_X + C.BALL_RADIUS:
            if C.GOAL_Y_MIN <= self.ball_pos[1] <= C.GOAL_Y_MAX:
                scorer = "blue"
            else:  # post / wall reflection
                self.ball_pos[0] = C.LEFT_GOAL_X + C.BALL_RADIUS
                self.ball_vel[0] = -self.ball_vel[0]
        # right side (blue goal -> red scores)
        elif self.ball_pos[0] >= C.RIGHT_GOAL_X - C.BALL_RADIUS:
            if C.GOAL_Y_MIN <= self.ball_pos[1] <= C.GOAL_Y_MAX:
                scorer = "red"
            else:
                self.ball_pos[0] = C.RIGHT_GOAL_X - C.BALL_RADIUS
                self.ball_vel[0] = -self.ball_vel[0]
        return scorer

    # ------------------------------------------------------------------ #
    #  Capturing                                                          #
    # ------------------------------------------------------------------ #
    def _capture(self):
        sp = H.speed(self.ball_vel)
        if self.owner == -1:
            candidates = []
            for p in range(C.N_PLAYERS):
                if self.kick_cd > 0 and p == self.last_kicker:
                    continue
                thr = self.cfg.capture_moving_speed
                if p in C.GK_INDICES:
                    thr *= self.cfg.gk_capture_mult
                if (H.distance(self.pos[p], self.ball_pos) <= self.cfg.capture_radius
                        and sp <= thr):
                    candidates.append(p)
            if candidates:
                self.owner = int(self.rng.choice(candidates))
                self.ball_vel[:] = 0.0
                self.ball_pos[:] = self.pos[self.owner]

                is_completed_pass = (
                    self.last_kick_was_pass
                    and self.last_kicker != -1
                    and self.owner != self.last_kicker
                )
                kicker_blue  = is_completed_pass and (self.last_kicker in C.BLUE_ALL)
                catcher_blue = is_completed_pass and (self.owner in C.BLUE_ALL)

                if (is_completed_pass and kicker_blue and catcher_blue
                        and self.last_kicker in C.BLUE_FIELD
                        and self.owner in C.BLUE_FIELD):
                    self.assist_tracker.on_pass_touch(self.last_kicker, self.owner)
                else:
                    self.assist_tracker.on_touch(
                        self.owner,
                        is_blue_field=(self.owner in C.BLUE_FIELD)
                    )


                if is_completed_pass:

                    # ADD THIS: Record the exact pass pair
                    self.pass_edges[(self.last_kicker, self.owner)] += 1


                    if kicker_blue and catcher_blue:
                        self.blue_passes_success += 1
                    elif not kicker_blue and not catcher_blue:
                        self.red_passes_success += 1
                    
           


                        
                self.last_kick_was_pass = False
                return self.owner
        else:
            others = [p for p in range(C.N_PLAYERS)
                    if p != self.owner
                    and H.distance(self.pos[p], self.ball_pos) <= self.cfg.capture_radius]
            if others:
                old_owner = self.owner   # ← ADD this line first
                pool = [self.owner] + others
                self.owner = int(self.rng.choice(pool))
                # ↑ original code ends here
                # ↓ ADD immediately after new owner is chosen
                if self.owner != old_owner:
                    self.assist_tracker.on_touch(
                        self.owner,
                        is_blue_field=(self.owner in C.BLUE_FIELD)
                    )
                # ↑ ADD ends here
            self.ball_vel[:] = 0.0
            self.ball_pos[:] = self.pos[self.owner]
        return -1

    def _resolve_sot(self, on_target: bool):
        """Close a tracked shot: update SoT counters and give reward if on target."""
        if not self.shot_in_flight:
            return
        if on_target:
            if self.shot_team == 'blue':
                self.blue_sot += 1
                self.rew.on_shot_on_target(self.shot_shooter)

            else:
                self.red_sot += 1
        self.shot_in_flight = False
        self.shot_shooter   = -1
        self.shot_team      = None                       

    def _goalkeeper_distribute(self):
        """If a keeper owns the ball, immediately pass to the nearest field mate."""
        if self.owner in C.GK_INDICES:
            mates = C.BLUE_FIELD if self.owner in C.BLUE_ALL else C.RED_FIELD
            target = H.nearest_index(self.pos[self.owner], self.pos, mates)
            if target is not None:
                self._do_pass(self.owner, target)


    #-----------------------------------------------new rule to capture and mirror the ball
    def _resolve_player_collisions(self):
        if self.owner != -1:
            return
        sp = H.speed(self.ball_vel)
        collision_dist = C.PLAYER_RADIUS + C.BALL_RADIUS
        for p in range(C.N_PLAYERS):
            if self.kick_cd > 0 and p == self.last_kicker:
                continue
            dist = H.distance(self.pos[p], self.ball_pos)
            if dist > collision_dist:
                continue
            thr = self.cfg.capture_moving_speed
            if p in C.GK_INDICES:
                thr *= self.cfg.gk_capture_mult
            if sp <= thr:
                continue  # slow enough → let _capture() grab it
            # too fast → mirror bounce
            normal = (self.ball_pos - self.pos[p]) / dist if dist > 1e-6 else np.array([1.0, 0.0])
            self.ball_vel -= 2.0 * float(np.dot(self.ball_vel, normal)) * normal
            self.ball_pos[:] = self.pos[p] + normal * (collision_dist + 1.0)
            if self.shot_in_flight:
                self._resolve_sot(on_target=False)
            break
    # ------------------------------------------------------------------ #
    #  One simulation tick                                                #
    # ------------------------------------------------------------------ #
    def step(self, blue_actions, red_actions):

        self.rew.begin_step()
        blue_actions = [int(a) for a in blue_actions]
        red_actions = [int(a) for a in red_actions]

        if self.kick_cd > 0:
            self.kick_cd -= 1
            
        # --- 1. ADD THIS: Save position before movement ---
        old_pos = self.pos.copy()

        # 1) goalkeepers track the ball
        self._move_goalkeeper(C.BLUE_GK, C.BLUE_GK_X)
        self._move_goalkeeper(C.RED_GK, C.RED_GK_X)

        # 2) field players act (move OR shoot OR pass OR stay)
        for slot, idx in enumerate(C.BLUE_FIELD):
            self._apply_field_action(idx, blue_actions[slot])
        for slot, idx in enumerate(C.RED_FIELD):
            self._apply_field_action(idx, red_actions[slot])

        # --- 2. ADD THIS: Calculate the actual velocity (current pos - old pos) ---
        self.player_vel = self.pos - old_pos

        # keep an owned ball glued to its owner after movement
        if self.owner != -1:
            self.ball_pos[:] = self.pos[self.owner]

        # 3) ball physics (only meaningful while free)
        scorer = None
        if self.owner == -1:
            scorer = self._update_ball()
        if scorer is None and self.owner == -1:       # ← ADD
            self._resolve_player_collisions() 

        # 4) capture / contested possession + outcome-based SoT: GK save
        if scorer is None:
            capturer = self._capture()
            if self.shot_in_flight and capturer in C.GK_INDICES:
                opp_gk = C.RED_GK if self.shot_team == 'blue' else C.BLUE_GK
                self._resolve_sot(on_target=(capturer == opp_gk))
            elif self.shot_in_flight and capturer != -1:
                self._resolve_sot(on_target=False)
            self._goalkeeper_distribute()

        # 5) outcome-based SoT: goal scored by the shooter
        if scorer is not None and self.shot_in_flight:
            self._resolve_sot(on_target=(scorer == self.shot_team))

        # 6) goals + shaping
        if scorer is not None:
            if scorer == "blue":
                self.blue_goals += 1
                #self.rew.on_goal_scored()
                scorer_player, assister_player = self.assist_tracker.on_goal()
                blue_field    = list(C.BLUE_FIELD)
                scorer_slot   = blue_field.index(scorer_player)   if scorer_player   in blue_field else None
                assister_slot = blue_field.index(assister_player) if assister_player in blue_field else None
                if assister_slot is not None:
                    self.blue_assists += 1    # <-- ADD THIS LINE
                self.rew.on_goal_scored(scorer_slot, assister_slot)
            else:
                self.red_goals += 1
                self.rew.on_goal_conceded()
            self._kickoff()          # resets prev distances internally
        else:
            self.rew.shaping(self.owner, self.pos, self.ball_pos)
            self.rew.chase_shaping(self.owner, self.pos, self.ball_pos)


        # 7) stats
        
        # --- NEW TURNOVER TRACKING LOGIC ---
        if self.owner in C.BLUE_FIELD:
            # Remember the last blue agent to have the ball
            self.last_blue_owner = self.owner
            self.turnover_from_pass = False  #  Clear exemption if they safely control/dribble it
        elif self.owner == C.BLUE_GK:
            # Clear it if the Blue Goalkeeper gets it (so field agents aren't punished for the GK's mistakes)
            self.last_blue_owner = -1
            self.turnover_from_pass = False  
        elif self.owner in C.RED_ALL:
    # If red gets the ball and a blue agent was the last owner...
            if self.last_blue_owner != -1:
                self.rew.on_turnover(self.last_blue_owner)   # now also penalizes intercepted passes
                self.last_blue_owner = -1  # Reset so they only get penalized once per turnover
                self.turnover_from_pass = False
            
            #Reset the pass memory because Red took possession
            self.last_pass_pair = (-1, -1)                                    
        # -----------------------------------

        self.poss.update(self.owner)
        self.steps += 1
        return self.rew.end_step()

    #def _apply_field_action(self, idx, action, shooter_on_target):
    def _apply_field_action(self, idx, action):
        if action in C.MOVE_VECTORS:                 # move + stay
            self._move_field_player(idx, action)
            #------------------------------------------------

            if action == C.A_STAY:
                self.stamina[idx] = min(C.STAMINA_MAX, self.stamina[idx] + C.STAMINA_RECOVER)
            else:
                self.move_count[idx] += 1
                net_dist = H.distance(self.pos[idx], self.window_start_pos[idx])

                if net_dist > C.VIBRATION_RADIUS:
                    max_net    = self.cfg.player_speed * 10
                    drain_ratio = min(net_dist / max_net, 1.0)
                    self.stamina[idx] = max(0.0, self.stamina[idx] - C.STAMINA_DRAIN * drain_ratio)

                if self.move_count[idx] >= 10:
                    self.move_count[idx] = 0
                    self.window_start_pos[idx] = self.pos[idx].copy()
            #------------------------------------------------
        elif action == C.A_SHOOT:
            if self.owner == idx:
                self._do_shot(idx)
        elif action in C.PASS_ACTIONS:
            if self.owner == idx:
                # Deduce which teammate to pass to (0 for A, 1 for B, 2 for C)
                mate_slot = action - C.A_PASS_A 
                
                # Get the teammates in the exact same order used for observations
                group = C.BLUE_FIELD if idx in C.BLUE_ALL else C.RED_FIELD
                mates = [p for p in group if p != idx]
                
                target_idx = mates[mate_slot]
                self._do_pass(idx, target_idx)
                    
                

    # ------------------------------------------------------------------ #
    #  Helpers for the wrapper                                            #
    # ------------------------------------------------------------------ #
    def episode_stats(self):
        bp, rp = self.poss.percent()
        return {
            "blue_goals": self.blue_goals,
            "red_goals": self.red_goals,
            "blue_shots": self.blue_shots,
            "blue_shots_on_target": self.blue_sot,
            "blue_passes": self.blue_passes,
            "blue_passes_success": self.blue_passes_success,
            "blue_assists": self.blue_assists,
            "red_shots": self.red_shots,
            "red_shots_on_target": self.red_sot,
            "red_passes": self.red_passes,
            "red_passes_success": self.red_passes_success,
            "blue_possess_percent": round(bp, 2),
            "red_possess_percent": round(rp, 2),
            "total_reward": round(self.rew.total_reward, 4),
            "reward_components": {k: round(v, 4) for k, v in self.rew.reward_components.items()},
            "agent_rewards": [round(r, 4) for r in self.rew.agent_total_rewards],
            "agent_distance_run": [round(float(self.distance_run[i]), 1) for i in range(C.N_PLAYERS)],
            "agent_total_stamina": [round(max(0.25, 1.0 - 0.60 * (self.distance_run[i] /15000.0)), 2) for i in range(C.N_PLAYERS)],
            "pass_edges": dict(self.pass_edges),
            "red_tactic": self.red_tactic     
            }


# =========================================================================== #
#  PETTINGZOO WRAPPER                                                          #
# =========================================================================== #
def _import_pettingzoo():
    from pettingzoo import ParallelEnv
    from gymnasium import spaces
    return ParallelEnv, spaces


try:
    _ParallelEnv, _spaces = _import_pettingzoo()
    _PZ_AVAILABLE = True
except Exception:                       # allow engine-only use without deps
    _ParallelEnv, _spaces = object, None
    _PZ_AVAILABLE = False


class GridFootballEnv(_ParallelEnv):
    """PettingZoo ParallelEnv exposing the four blue field players as agents."""

    metadata = {"render_modes": ["human", "rgb_array"],
                "name": "grid_football_4v4_v0", "is_parallelizable": True}

    def __init__(self, render_mode=None, config: C.Config = C.DEFAULT_CONFIG,
                 red_controller=None, seed=None):
        if not _PZ_AVAILABLE:
            raise ImportError(
                "pettingzoo and gymnasium are required for GridFootballEnv. "
                "Install with: pip install pettingzoo gymnasium")
        super().__init__()
        self.cfg = config
        self.render_mode = render_mode
        self.engine = FootballEngine(config=config, seed=seed)

        # red controller: callable(engine) -> length-4 int array of red actions
        if red_controller is None:
            from red_control import RedController
            red_controller = RedController(seed=seed)   # <--- Removed the () at the end
        self.red_controller = red_controller

        self.possible_agents = ["blue_0", "blue_1", "blue_2", "blue_3"]
        self.agents = list(self.possible_agents)
        self._agent_to_idx = {"blue_0": 0, "blue_1": 1, "blue_2": 2, "blue_3": 3}

        self._obs_space = _spaces.Box(low=-1.0, high=1.0, shape=(C.OBS_DIM,), dtype=np.float32)

        self._act_space = _spaces.Discrete(C.N_ACTIONS)

        # pygame handles (lazy)
        self._screen = None
        self._clock = None
        self._font = None
        #
        self._small  = None 
        self._last_obs = None
    # gymnasium / pettingzoo space accessors -------------------------------- #
    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    # ----------------------------------------------------------------- #
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.red_controller.reset(seed=seed)
            self.engine.local_red_formation[:] = self.red_controller.get_formation()
            self.engine.reset(seed=seed)
        else:
            self.red_controller.reset()
            self.engine.local_red_formation[:] = self.red_controller.get_formation()
            self.engine.reset()

        # Sync red tactic to the engine immediately for step 0
        self.engine.red_tactic = self.red_controller.tactic

        self.agents = list(self.possible_agents)
        self.current_speed = np.zeros(C.N_PLAYERS, dtype=np.float64)
        obs = self._all_obs()
        self._last_obs = obs
        infos = {a: {"action_mask": self._mask(self._agent_to_idx[a])}
                 for a in self.agents}
        return obs, infos

    def step(self, actions):
        # blue actions from the policy, red actions from the rule-based controller
        blue = [int(actions.get(a, C.A_STAY)) for a in self.possible_agents]
        red = self.red_controller(self.engine)

        # Sync the formation in case a mid-game tactic change triggers a kickoff
        self.engine.local_red_formation[:] = self.red_controller.get_formation()

        red = [int(x) for x in np.asarray(red).reshape(-1)[:4]]
        if len(red) < 4:
            red = (red + [C.A_STAY] * 4)[:4]

        rewards_arr = self.engine.step(blue, red)

        truncated = self.engine.steps >= self.cfg.max_steps
        obs = self._all_obs()
        self._last_obs = obs
        rewards = {a: float(rewards_arr[self._agent_to_idx[a]]) for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: truncated for a in self.agents}
        infos = {a: {"action_mask": self._mask(self._agent_to_idx[a])}
                 for a in self.agents}

        if self.render_mode == "human":
            self.render()

        if truncated:
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    # ----------------------------------------------------------------- #
    #  Observation construction                                          #
    # ----------------------------------------------------------------- #
    def _mask(self, idx):
        """Action mask: with the ball -> everything (except self-pass); otherwise move + stay only."""
        player_idx = list(C.BLUE_FIELD)[idx]        # slot 0/1/2/3 → actual player index
        mask = np.zeros(C.N_ACTIONS, dtype=np.int8)
        
        for a in C.MOVE_ACTIONS:
            mask[a] = 1
        mask[C.A_STAY] = 1
        
        if self.engine.owner == player_idx:
            mask[C.A_SHOOT] = 1
            

            # Enable all passing actions
            for a in C.PASS_ACTIONS:
                mask[a] = 1
                
        return mask        

    def action_masks(self):
        return {a: self._mask(self._agent_to_idx[a]) for a in self.possible_agents}
    

#----- obs--------

    def _obs_for(self, idx):
        e = self.engine
        o = np.zeros(C.OBS_DIM, dtype=np.float32)
        k = 0
        own = e.owner
        
        # --- 1. Self data ---
        o[k] = 1.0 if own == idx else 0.0;                                     k += 1
        o[k:k+2] = np.array(H.fine_norm(e.ball_pos),    dtype=np.float32)*2-1; k += 2
        o[k:k+2] = np.clip(np.array(e.ball_vel, dtype=np.float32) / C.SHOT_SPEED, -1.0, 1.0); k += 2
        o[k:k+2] = np.array(H.fine_norm(e.pos[idx]),    dtype=np.float32)*2-1; k += 2
        o[k]     = e.get_combined_stamina_mult(idx);                          k += 1
        o[k:k+2] = np.clip(np.array(e.player_vel[idx], dtype=np.float32) / self.cfg.player_speed, -1.0, 1.0); k += 2
        
        # --- 2. Teammate data ---
        mates = [p for p in C.BLUE_FIELD if p != idx]
        for p in mates:
            o[k] = 1.0 if own == p else 0.0;                                   k += 1
            o[k:k+2] = np.array(H.coarse_norm(e.pos[p]),dtype=np.float32)*2-1; k += 2
            o[k] = e.get_combined_stamina_mult(p);                             k += 1
            o[k:k+2] = np.clip(np.array(e.player_vel[p], dtype=np.float32) / self.cfg.player_speed, -1.0, 1.0); k += 2

        # --- 3. Opponent field players data (distance-gated per agent) ---
        for p in list(C.RED_FIELD):
            visible = H.distance(e.pos[idx], e.pos[p]) <= C.OPP_VISION_RADIUS
            o[k] = 1.0 if visible else 0.0;                                    k += 1
            if visible:
                o[k] = 1.0 if own == p else 0.0;                               k += 1
                o[k:k+2] = np.array(H.coarse_norm(e.pos[p]),dtype=np.float32)*2-1; k += 2
                o[k] = e.get_combined_stamina_mult(p);                         k += 1
                o[k:k+2] = np.clip(np.array(e.player_vel[p], dtype=np.float32) / self.cfg.player_speed, -1.0, 1.0); k += 2
            else:
                k += 6   # has_ball, pos(2), stamina, vel(2) all left at 0.0

        # --- 4. Opponent goalkeeper (distance-gated per agent) ---
        gk_visible = H.distance(e.pos[idx], e.pos[C.RED_GK]) <= C.OPP_VISION_RADIUS
        o[k] = 1.0 if gk_visible else 0.0;                                     k += 1
        if gk_visible:
            o[k:k+2] = np.array(H.coarse_norm(e.pos[C.RED_GK]), dtype=np.float32)*2-1
        k += 2   # position slots reserved either way (left 0.0 if not visible)

        # --- 5. Game state and ID ---
        o[k] = 1.0 if own == -1 else 0.0;                                      k += 1
        o[k] = float(H.clamp((e.blue_goals - e.red_goals)/C.SCORE_NORM,-1.,1.));k += 1
        o[k] = e.steps / self.cfg.max_steps;                                    k += 1
        o[k + int(idx)] = 1.0;                                                  k += 4

        # --- 6. Red's current tactic, one-hot (A/B/C) ---
        _TACTIC_ONE_HOT = {"A": 0, "B": 1, "C": 2}
        tactic_idx = _TACTIC_ONE_HOT.get(getattr(e, "red_tactic", "A"), 0)
        o[k + tactic_idx] = 1.0;                                                k += 3

        assert k == C.OBS_DIM, f"obs wrote {k} slots, expected {C.OBS_DIM}"
        return o

    def _all_obs(self):
        return {a: self._obs_for(self._agent_to_idx[a]) for a in self.possible_agents}

    def episode_stats(self):
        return self.engine.episode_stats()

    # ----------------------------------------------------------------- #
    #  Rendering (pygame)                                                #
    # ----------------------------------------------------------------- #
    def render(self):
        if self.render_mode is None:
            return None
        import pygame
        if self._screen is None:
            pygame.init()
            pygame.display.set_caption("Grid Football 5v5")
            flags = 0 if self.render_mode == "human" else pygame.HIDDEN
            self._screen = pygame.display.set_mode((int(C.FIELD_W), int(C.FIELD_H) + 60 + 120), flags)
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("consolas", 20)
            self._small = pygame.font.SysFont("consolas", 16)

        e = self.engine
        scr = self._screen
        scr.fill((10, 10, 10))
        pitch = pygame.Rect(0, 60, int(C.FIELD_W), int(C.FIELD_H))
        pygame.draw.rect(scr, C.COLOR_BG, pitch)

        def Y(v):  # shift pitch down by the 60px scoreboard strip
            return int(v) + 60

        # midline + centre circle
        pygame.draw.line(scr, C.COLOR_LINE, (C.FIELD_W // 2, Y(0)),
                         (C.FIELD_W // 2, Y(C.FIELD_H)), 2)
        pygame.draw.circle(scr, C.COLOR_LINE,
                           (int(C.FIELD_W // 2), Y(C.FIELD_H // 2)), 60, 2)
        # goals
        for gx in (C.LEFT_GOAL_X, C.RIGHT_GOAL_X):
            pygame.draw.line(scr, (255, 255, 0), (int(gx), Y(C.GOAL_Y_MIN)),
                             (int(gx), Y(C.GOAL_Y_MAX)), 5)

        # players
        for i in range(C.N_PLAYERS):
            x, y = int(e.pos[i, 0]), Y(e.pos[i, 1])
            # --- ADD THIS: Circle of sight for Blue field players ---
            if i in C.BLUE_FIELD:
                pygame.draw.circle(scr, (173, 216, 230), (x, y), int(C.OPP_VISION_RADIUS), 1)
            # --------------------------------------------------------
            if i in C.BLUE_FIELD:
                col = C.COLOR_BLUE
            elif i == C.BLUE_GK:
                col = C.COLOR_GK_BLUE
            elif i in C.RED_FIELD:
                col = C.COLOR_RED
            else:
                col = C.COLOR_GK_RED
            pygame.draw.circle(scr, col, (x, y), int(C.PLAYER_RADIUS))
            if i == e.owner:
                pygame.draw.circle(scr, (255, 255, 255), (x, y),
                                   int(C.PLAYER_RADIUS) + 3, 2)
            label = (str(i) if i in C.BLUE_FIELD or i in C.RED_FIELD else "G")
            tag = self._small.render(label, True, (255, 255, 255))
            scr.blit(tag, (x - 5, y - 8))
            #------------------Stamina graphic-----------------------------------
            # stamina bar (field players only)
            if i in C.BLUE_FIELD or i in C.RED_FIELD:
                bar_w  = 30
                bar_h  = 4
                bar_x  = x - bar_w // 2
                1
                # --- NEW Total Game Stamina Bar (Bottom) ---
                tot_bar_y = y - int(C.PLAYER_RADIUS) - 6
                # Background
                pygame.draw.rect(scr, (60, 60, 60), (bar_x, tot_bar_y, bar_w, bar_h))
                
                # Visual stamina drops from 1.0 to 0.4 at 10000px   
                dist_ratio = min(e.distance_run[i] / 15000.0, 1.0)
                tot_stamina_visual = 1.0 - (0.60 * dist_ratio)
                
                # Fill portion (Cyan color)
                pygame.draw.rect(scr, (0, 200, 255), (bar_x, tot_bar_y, int(bar_w * tot_stamina_visual), bar_h))

                # --- Short-term Stamina Bar (Top) ---
                bar_y  = y - int(C.PLAYER_RADIUS) - 12
                # Background 
                pygame.draw.rect(scr, (60, 60, 60), (bar_x, bar_y, bar_w, bar_h))

                # fill colour: green → yellow → red
                st = e.stamina[i]
                if st > 0.5:
                    bar_col = (int(255 * (1 - st) * 2), 255, 0)   
                else:
                    bar_col = (255, int(255 * st * 2), 0)           

                # filled portion
                pygame.draw.rect(scr, bar_col, (bar_x, bar_y, int(bar_w * st), bar_h))
            #-----------------------------------------------------------

        # ball
        pygame.draw.circle(scr, C.COLOR_BALL,
                           (int(e.ball_pos[0]), Y(e.ball_pos[1])),
                           int(C.BALL_RADIUS))

        # scoreboard
        # scoreboard
        bp, rp = e.poss.percent()
        secs = e.steps / C.STEPS_PER_SECOND
        score = self._font.render(
            f"BLUE {e.blue_goals} - {e.red_goals} RED", True, C.COLOR_TEXT)
        meta = self._small.render(
            f"t={secs:5.1f}s  poss B/R {bp:4.1f}/{rp:4.1f}%  "
            f"shots(SoT) B {e.blue_shots}({e.blue_sot})  R {e.red_shots}({e.red_sot})  "
            f"pass(ok) B {e.blue_passes}({e.blue_passes_success})  R {e.red_passes}({e.red_passes_success})",
            True, C.COLOR_TEXT)
        scr.blit(score, (C.FIELD_W // 2 - 90, 6))
        scr.blit(meta,  (10, 34))

        # --- ADD THIS: Bottom Observation Dashboard ---
        if self._last_obs is not None:
            ui_y = int(C.FIELD_H) + 60
            pygame.draw.line(scr, C.COLOR_LINE, (0, ui_y), (C.FIELD_W, ui_y), 2)
            
            # Loop through 4 columns (1 for each blue agent)
            for idx, agent_id in enumerate(self.possible_agents):
                col_x = idx * (int(C.FIELD_W) // 4)
                obs_arr = self._last_obs.get(agent_id)
                if obs_arr is None: continue
                
                # Title for the column
                title = self._font.render(f"[{agent_id.upper()}]", True, (0, 200, 255))
                scr.blit(title, (col_x + 10, ui_y + 8))
                
                # In your observation structure, Red player data begins at index 28, taking 7 slots each
                red_idx = 28
                for r_i in range(4):
                    is_visible = obs_arr[red_idx] > 0.5
                    
                    if is_visible:
                        # Index + 2 is x, Index + 3 is y
                        rx, ry = obs_arr[red_idx + 2], obs_arr[red_idx + 3]
                        text = f"R{r_i}: X={rx:+.2f} Y={ry:+.2f}"
                        color = (255, 120, 120) # Light red
                    else:
                        text = f"R{r_i}: Not visible"
                        color = (120, 120, 120) # Greyed out
                    
                    surf = self._small.render(text, True, color)
                    scr.blit(surf, (col_x + 10, ui_y + 35 + (r_i * 18)))
                    
                    red_idx += 7
        # ----------------------------------------------

        if self.render_mode == "human":
            for _ in pygame.event.get():
                pass
            pygame.display.flip()

            
            self._clock.tick(C.RENDER_FPS)
            return None
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(scr)), axes=(1, 0, 2))

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None


# Convenience factory expected by some PettingZoo tooling.
def parallel_env(**kwargs):
    return GridFootballEnv(**kwargs)