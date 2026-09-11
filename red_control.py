from __future__ import annotations
import numpy as np
import config as C
import helper as H


class RedController:
    """
    Rule-based controller for the 4 red field players, with 3 switchable tactics.

    ── Formations ─────────────────────────────────────────────────────────
      A  2-2   : 6 & 7 defend (capped just behind halfway), 5 & 8 attack.
      B  3-1   : 6/7/8 defend (may push up to x = 550), 5 is the lone striker.
                 "Aggressive" — even while sitting in a back three, this
                 tactic never just parks the ball: the defending line holds
                 higher, a defender who wins the ball in space will carry it
                 forward instead of always laying it off, and the striker
                 sprints into the box rather than waiting to be found.
      C  1-3   : 5/7/8 attack, 6 defends but may advance to x = 720.
                 "Attacking" — the three forward players stagger their runs
                 (one bursts in behind, the other holds width) so the
                 carrier always has two live passing angles, not one.

    ── Tactic switching ───────────────────────────────────────────────────
    Every TACTIC_REVIEW_STEPS ticks (and on reset) the score is checked:
      red ahead   → prefers B   (protect the lead)
      blue ahead  → prefers C   (chase the game)
      level       → prefers A
    The preferred tactic is taken with probability SWITCH_PROB (0.80);
    otherwise one of the two remaining tactics is picked at random.

    ── Roles within a tactic ──────────────────────────────────────────────
    DEFENDERS
      • Has the ball        → normally passes out immediately (prefers an
                              attacker, and prefers whoever has the more
                              open sight of goal). In an "aggressive" tactic
                              a defender with a clear lane in front of them
                              will sometimes carry the ball forward instead.
      • Nearest to the ball → press the carrier / loose ball directly.
      • Others              → sit in the intercept lane between ball and goal.
      • Ball far and safe   → hold a line behind the ball (higher up the
                              pitch for "aggressive" tactics), clamped to
                              the tactic's max_x and the player's own y-lane.
    ATTACKERS
      • Carrier             → shoot if close AND the angle on goal is open;
                              if the angle is too tight, look for a mate
                              with a better angle, or dribble sideways to
                              open it up themselves. Otherwise dribble
                              forward — juking away from the nearest
                              defender blocking the direct line — or pass,
                              weighting mates both by how advanced they are
                              and by how good their shooting angle is.
      • Best zone match     → press the blue carrier (unless STRIKER_PRESSES
                              is off for this tactic).
      • Others              → man-mark blue field players, or (in the
                              "attacking" tactic) make staggered support
                              runs instead of a single generic one.

    ── Chaos ──────────────────────────────────────────────────────────────
      CHAOS_PROB — random move; HESITATE_PROB — A_STAY for one tick.
    """

    # ══════════════════════════════════════════════════════════════════════
    #  TACTIC TABLE
    #  defenders       : {player_idx: max_x that player may advance to}
    #  striker_presses : may an attacker leave shape to press the carrier?
    #  striker_max_x   : optional cap on how far a non-defender may push up
    #  aggressive      : back line still hunts the goal (tactic B)
    #  attacking       : forward three use staggered support runs (tactic C)
    #  defender_lag    : how far behind the ball the defensive line sits
    #                    (overrides DEFENDER_LAG when present)
    # ══════════════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════════════
    #  TACTIC TABLE
    #  defenders       : {player_idx: max_x that player may advance to}
    #  striker_presses : may an attacker leave shape to press the carrier?
    #  striker_max_x   : optional cap on how far a non-defender may push up
    #  aggressive      : back line still hunts the goal (tactic B)
    #  attacking       : forward three use staggered support runs (tactic C)
    #  defender_lag    : how far behind the ball the defensive line sits
    #                    (overrides DEFENDER_LAG when present)
    # ══════════════════════════════════════════════════════════════════════
    TACTICS = {
        "A": {
            # 2-2 Balanced setup
            # 2 Defenders (6, 7) and 2 Attackers (5, 8)
            "defenders":       {6: C.FIELD_W / 2 - 20, 7: C.FIELD_W / 2 - 20},
            "striker_presses": True,
            # Limit from middle y: 5 & 6 take top half (0-250), 7 & 8 take bottom half (250-500)
            "y_limits":        {5: (0, 250), 6: (0, 250), 7: (250, 500), 8: (250, 500)},
            "formation":       [
                [340.0, 125.0], # 5: Top Attacker
                [240.0, 125.0], # 6: Top Defender
                [240.0, 375.0], # 7: Bottom Defender
                [340.0, 375.0]  # 8: Bottom Attacker
            ], 
        },
        "B": {
            # 3-1 Defensive setup
            "defenders":       {6: 600.0, 7: 600.0, 8: 600.0},
            "striker_presses": True,
            # Striker (5) locked strictly to the middle y-axis
            "y_limits":        {5: (200, 300), 6: (0, 200), 7: (150, 350), 8: (300, 500)},
            "formation":       [
                [340.0, 250.0], # 5: Central Striker
                [140.0, 100.0], # 6: Top Defender
                [140.0, 250.0], # 7: Central Defender
                [140.0, 400.0]  # 8: Bottom Defender
            ], 
        },
        "C": {
            # 1-3 Attacking setup
            # 1 Defender (6) and 3 Attackers (5, 7, 8)
            "defenders":       {6: 700.0},
            "striker_presses": True,
            "y_limits":        {5: (0, 200), 6: (100, 400), 7: (150, 350), 8: (300, 500)},
            "formation":       [
                [340.0, 100.0], # 5: Top Attacker
                [140.0, 250.0], # 6: Central Defender
                [340.0, 250.0], # 7: Central Attacker
                [340.0, 400.0]  # 8: Bottom Attacker
            ], 
        },
    }
    START_TACTIC        = "A"

    # ── tactic switching ──────────────────────────────────────────────────
    TACTIC_REVIEW_STEPS = 300        # re-evaluate the score every N ticks
    SWITCH_PROB         = 0.80       # chance of taking the preferred tactic
    LEAD_MARGIN         = 1          # goals needed to count as "winning"

    # ── attacker thresholds ───────────────────────────────────────────────
    SHOOT_DIST           = 200
    SUPPORT_X_LAG        = 200
    NOISE_RADIUS         = 35
    SHADOW_NOISE         = 55
    PRESS_NOISE          = 20
    PASS_PROB            = 0.10      # chance a carrier passes instead of dribbling
    PASS_BEST_PROB       = 0.70      # chance a pass goes to the most advanced mate

    # ── shooting-angle / dribbling thresholds ───────────────────────────────
    MIN_SHOOT_ANGLE       = 0.35     # ~20°: below this the keeper covers it easily
    ANGLE_IMPROVE_MARGIN  = 0.12     # a mate must beat our own angle by this much
    ANGLE_WEIGHT          = 150.0    # converts radians of "openness" into an
                                     # x-equivalent bonus when scoring passes
    DRIBBLE_JUKE_DIST     = 90       # react to a defender inside this range
    DRIBBLE_JUKE_Y        = 70       # how far sideways a juke moves the target

    # ── defender thresholds ───────────────────────────────────────────────
    DEFENDER_HOME_X      = 150       # minimum x to hold when deep
    DEFENDER_LAG         = 120       # default: how far behind the ball the line sits
    DEFENDER_CHASE_DIST  = 130       # chase a loose ball only within this range
    DEFENDER_PRESS_DIST  = 220       # nearest defender presses within this range
    DEFENDER_PASS_BIAS   = 300.0     # x-bonus that makes attackers preferred targets
    DEFENDER_CARRY_DIST  = 150       # a "clear lane" needs no blue player closer than this
    DEFENDER_CARRY_PROB  = 0.25      # chance an aggressive defender carries instead of passing

    # ── chaos dials ───────────────────────────────────────────────────────
    CHAOS_PROB           = 0.04
    HESITATE_PROB        = 0.05

    # ── curriculum (kept for compatibility) ───────────────────────────────
    SPEED_START          = 1.0
    SPEED_END            = 1.0
    CURRICULUM_EP        = 1000
    DELAY_MIN            = 0
    DELAY_MAX            = 0

    # ═════════════════════════════════════════════════════════════════════
    def __init__(self, seed=None):
        self.rng            = np.random.default_rng(seed)
        self.episode       = 0
        self.step_count    = 0
        self.speed_mult    = self.SPEED_START
        self.kickoff_delay = 0
        self.tactic        = self.START_TACTIC
        self.tactic_counts = {t: 0 for t in self.TACTICS}   # for logging
        self.last_score    = (0, 0)                         # (red, blue) last seen

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.episode   += 1
        t = min(self.episode / self.CURRICULUM_EP, 1.0)
        self.speed_mult = (self.SPEED_START
                           + t * (self.SPEED_END - self.SPEED_START))
        self.kickoff_delay = int(
            self.rng.integers(self.DELAY_MIN, self.DELAY_MAX + 1))
        # Randomly choose between "A", "B", or "C" at the start of each match
        self.tactic     = str(self.rng.choice(list(self.TACTICS.keys())))
        self.last_score = (0, 0)

    # ═════════════════════════════════════════════════════════════════════
    #  TACTIC MANAGEMENT
    # ═════════════════════════════════════════════════════════════════════
    @staticmethod
    def _get_score(engine) -> tuple[int, int]:
        """Return (red_goals, blue_goals) — confirmed against engine.py."""
        return int(engine.red_goals), int(engine.blue_goals)

    def _review_tactic(self, engine) -> None:
        red, blue = self._get_score(engine)

        if red - blue >= self.LEAD_MARGIN:
            preferred = "B"          # winning → shut the game down
        elif blue - red >= self.LEAD_MARGIN:
            preferred = "C"          # losing  → throw players forward
        else:
            preferred = "A"          # level   → balanced

        if self.rng.random() < self.SWITCH_PROB:
            self.tactic = preferred
        else:
            others = [t for t in self.TACTICS if t != preferred]
            self.tactic = str(self.rng.choice(others))

    def set_tactic(self, name: str) -> None:
        """Force a tactic (useful for evaluation / ablations)."""
        if name not in self.TACTICS:
            raise ValueError(f"unknown tactic {name!r}")
        self.tactic = name

    @property
    def _defenders(self) -> dict:
        return self.TACTICS[self.tactic]["defenders"]

    @property
    def _striker_presses(self) -> bool:
        return self.TACTICS[self.tactic]["striker_presses"]

    def _max_x(self, idx: int) -> float:
        return float(self._defenders[idx])
    
    def _y_limits(self, idx: int) -> tuple[float, float]:
        """Get tactic-specific y limits for a red player."""
        return self.TACTICS[self.tactic]["y_limits"][idx]

    def get_formation(self) -> np.ndarray:
        """Return the starting formation for the current tactic."""
        return np.array(self.TACTICS[self.tactic]["formation"], dtype=np.float64)
    
    def _striker_cap(self):
        """Optional x cap for non-defenders in the current tactic (or None)."""
        return self.TACTICS[self.tactic].get("striker_max_x")

    # ═════════════════════════════════════════════════════════════════════
    #  MAIN LOOP
    # ═════════════════════════════════════════════════════════════════════
    def act(self, engine) -> np.ndarray:
        self.step_count += 1
        if self.step_count <= self.kickoff_delay:
            engine.red_tactic = self.tactic
            return np.full(4, C.A_STAY, dtype=np.int64)

        # DISABLED: Do not change tactic based on score during the match
        # current_score = self._get_score(engine)
        # if current_score != self.last_score:
        #     self._review_tactic(engine)
        #     self.last_score = current_score

        self.tactic_counts[self.tactic] += 1
        engine.red_tactic = self.tactic   # exposed for env.py's observation encoding

        red_list      = list(C.RED_FIELD)
        actions       = np.full(4, C.A_STAY, dtype=np.int64)
        chaos_roll    = self.rng.random(4)
        hesitate_roll = self.rng.random(4)

        defenders     = self._defenders
        attacker_idxs = [i for i in red_list if i not in defenders]

        pressing_slot   = self._best_presser_slot(engine, red_list, attacker_idxs)
        press_def_idx   = self._press_defender_idx(engine, defenders)

        for slot, idx in enumerate(red_list):

            # ── global chaos ──────────────────────────────────────────
            if chaos_roll[slot] < self.CHAOS_PROB:
                move_pool     = list(C.MOVE_VECTORS.keys()) + [C.A_STAY]
                actions[slot] = int(self.rng.choice(move_pool))
                continue

            # ── hesitation ────────────────────────────────────────────
            if hesitate_roll[slot] < self.HESITATE_PROB:
                actions[slot] = C.A_STAY
                continue

            # ── defenders ─────────────────────────────────────────────
            if idx in defenders:
                # Other mark if ball is deep (x < 400) and Blue has possession
                if engine.ball_pos[0] < 400 and engine.owner in C.BLUE_ALL:
                    rank = list(defenders.keys()).index(idx)
                    actions[slot] = self._shadow_blue(engine, idx, rank)
                    continue

                actions[slot] = self._defender_action(
                    engine, idx, is_presser=(idx == press_def_idx))
                continue

            # ── attackers ─────────────────────────────────────────────
            if engine.owner == idx:
                actions[slot] = self._with_ball(engine, idx)

            elif engine.owner in C.BLUE_ALL:
                if slot == pressing_slot and self._striker_presses:
                    actions[slot] = self._press(engine, idx)
                else:
                    rank = self._offball_rank(slot, pressing_slot, red_list,
                                              attacker_idxs)
                    actions[slot] = self._shadow_blue(engine, idx, rank)

            elif engine.owner in C.RED_FIELD:
                actions[slot] = self._support_run(engine, idx, engine.owner)

            else:
                # ball free
                if slot == pressing_slot:
                    target = engine.ball_pos + self._jitter(self.PRESS_NOISE)
                    actions[slot] = self._move_toward(engine.pos[idx], target)
                else:
                    actions[slot] = self._second_ball(engine, idx)

        return actions

    # ═════════════════════════════════════════════════════════════════════
    #  DEFENDERS
    # ═════════════════════════════════════════════════════════════════════
    def _press_defender_idx(self, engine, defenders: dict) -> int:
        """
        Which defender (if any) leaves the lane to press the ball directly.
        Only one at a time, and only when the ball is genuinely threatening.
        """
        threatening = engine.owner in C.BLUE_ALL or engine.owner == -1
        if not threatening or not defenders:
            return -1

        dists = [(H.distance(engine.pos[i], engine.ball_pos), i)
                 for i in defenders]
        d, idx = min(dists, key=lambda t: t[0])
        return idx if d < self.DEFENDER_PRESS_DIST else -1

    def _defender_action(self, engine, idx: int, is_presser: bool) -> int:
        """
        Four-state defender logic:
          1. Has the ball        → pass out (or carry, for an aggressive tactic).
          2. Nearest & in range  → press the ball directly.
          3. Blue has ball, or loose ball nearby → intercept lane.
          4. Otherwise           → hold the defensive line.
        """
        if engine.owner == idx:
            return self._defender_pass(engine, idx)

        if is_presser:
            target = engine.ball_pos + self._jitter(self.PRESS_NOISE)
            target = np.asarray(target, dtype=float)
            target[0] = float(np.clip(target[0], C.PLAYER_RADIUS, self._max_x(idx)))
            return self._move_toward(engine.pos[idx], target)

        dist_to_ball = H.distance(engine.pos[idx], engine.ball_pos)
        ball_loose   = engine.owner == -1

        if engine.owner in C.BLUE_ALL or (ball_loose
                                          and dist_to_ball < self.DEFENDER_CHASE_DIST):
            return self._defender_intercept(engine, idx)

        return self._defender_hold(engine, idx)

    def _defender_pass(self, engine, idx: int) -> int:
        pos = engine.pos[idx]
        cfg = self.TACTICS[self.tactic]

        # --- NEW TACTIC B RULE ---
        if self.tactic == "B":
            if self.rng.random() < 0.5:
                return C.A_SHOOT  # 50% chance to shoot
            else:
                # 50% chance to pass to the striker
                mates = [p for p in C.RED_FIELD if p != idx]
                striker_idx = [m for m in mates if m not in self._defenders][0]
                return self._pass_to_mate(mates, striker_idx)
        # -------------------------

        if cfg.get("aggressive") and self.rng.random() < self.DEFENDER_CARRY_PROB:
            blocker = H.nearest_index(pos, engine.pos, C.BLUE_FIELD)
            lane_clear = (blocker is None
                          or H.distance(pos, engine.pos[blocker]) > self.DEFENDER_CARRY_DIST)
            if lane_clear:
                return self._dribble_forward(engine, idx, pos)

        mates = [p for p in C.RED_FIELD if p != idx]
        
        scores = []
        for m in mates:
            if self.tactic == "A" and m in self._defenders:
                scores.append(-9999.0)  
            else:
                bias = 0.0 if m in self._defenders else self.DEFENDER_PASS_BIAS
                scores.append(engine.pos[m][0] + bias + self.ANGLE_WEIGHT * self._shot_angle(engine.pos[m]))
                
        return self._pass_action(scores)

    def _defender_intercept(self, engine, idx: int) -> int:
        """
        Move to a position 40 % of the way between own goal and the ball,
        clamped to this tactic's max_x and the player's own y-lane.
        """
        own_goal  = np.array([C.LEFT_GOAL_X, C.GOAL_CENTER_Y], dtype=float)
        ball      = engine.ball_pos.astype(float)
        intercept = own_goal + 0.4 * (ball - own_goal)
        intercept += self._jitter(self.PRESS_NOISE)

        intercept[0] = float(np.clip(intercept[0], C.PLAYER_RADIUS, self._max_x(idx)))
        y_lo, y_hi   = H.get_y_limits(idx)
        intercept[1] = float(np.clip(intercept[1], y_lo, y_hi))

        return self._move_toward(engine.pos[idx], intercept)

    def _defender_hold(self, engine, idx: int) -> int:
        """Hold a line behind the ball, capped by this tactic's max_x. An
        'aggressive' tactic overrides DEFENDER_LAG to hold a higher line."""
        y_lo, y_hi = self._y_limits(idx)
        lag = self.TACTICS[self.tactic].get("defender_lag", self.DEFENDER_LAG)

        target_x = float(np.clip(engine.ball_pos[0] - lag,
                                 self.DEFENDER_HOME_X,
                                 self._max_x(idx)))
        target = np.array([
            target_x + self.rng.uniform(-20, 20),
            float(np.clip((y_lo + y_hi) / 2 + self.rng.uniform(-30, 30),
                          y_lo, y_hi)),
        ])
        return self._move_toward(engine.pos[idx], target)

    # ═════════════════════════════════════════════════════════════════════
    #  ATTACKERS
    # ═════════════════════════════════════════════════════════════════════
    def _best_presser_slot(self, engine, red_list: list,
                           attacker_idxs: list) -> int:
        """
        Pick the attacker slot that should press.
        Prefers the one whose y-zone contains ball.y; falls back to nearest.
        Returns -1 if this tactic has no attackers at all.
        """
        if not attacker_idxs:
            return -1

        ball_y         = engine.ball_pos[1]
        attacker_slots = [s for s, i in enumerate(red_list) if i in attacker_idxs]

        zone_candidates: list[tuple[float, int]] = []
        for slot in attacker_slots:
            idx = red_list[slot]
            y_lo, y_hi = self._y_limits(idx)
            if y_lo <= ball_y <= y_hi:
                d = H.distance(engine.pos[idx], engine.ball_pos)
                zone_candidates.append((d, slot))

        if zone_candidates:
            return min(zone_candidates, key=lambda t: t[0])[1]

        dists = [(H.distance(engine.pos[red_list[s]], engine.ball_pos), s)
                 for s in attacker_slots]
        return min(dists, key=lambda t: t[0])[1]

    def _press(self, engine, idx: int) -> int:
        target = engine.ball_pos + self._jitter(self.PRESS_NOISE)

        cap = self._striker_cap()
        if cap is not None and idx not in self._defenders:
            target[0] = min(target[0], cap)

        return self._move_toward(engine.pos[idx], target)

    @staticmethod
    def _offball_rank(slot: int, pressing_slot: int, red_list: list,
                      attacker_idxs: list) -> int:
        """Map a non-pressing attacker slot to which blue player it shadows."""
        others = [s for s, i in enumerate(red_list)
                  if i in attacker_idxs and s != pressing_slot]
        return others.index(slot) if slot in others else 0

    def _shadow_blue(self, engine, idx: int, rank: int) -> int:
        blue_without_ball = [p for p in sorted(C.BLUE_FIELD) if p != engine.owner]
        if not blue_without_ball:
            return C.A_STAY
        target_player = blue_without_ball[rank % len(blue_without_ball)]
        target        = engine.pos[target_player].copy().astype(float)
        target       += self._jitter(self.SHADOW_NOISE)
        y_lo, y_hi    = self._y_limits(idx)
        target[1]     = float(np.clip(target[1], y_lo, y_hi))
        
        # --- NEW TACTIC B RULE ---
        # If Blue attacks, the Tactic B striker must stay in the attacking half (x >= 400)
        if self.tactic == "B" and idx not in self._defenders:
            target[0] = float(np.clip(target[0], 400.0, C.FIELD_W))
        # -------------------------
            
        return self._move_toward(engine.pos[idx], target)

    
    def _with_ball(self, engine, idx: int) -> int:
        pos          = engine.pos[idx]
        dist_to_goal = C.RIGHT_GOAL_X - pos[0]
        mates        = [p for p in C.RED_FIELD if p != idx]

        # --- NEW TACTIC RULES ---
        if self.tactic == "A":
            valid_mates = [m for m in mates if m not in self._defenders]
        elif self.tactic == "B":
            valid_mates = []  # Striker never passes in B
        else:
            valid_mates = mates
        # -------------------------

        if dist_to_goal < self.SHOOT_DIST:
            angle = self._shot_angle(pos)
            if angle >= self.MIN_SHOOT_ANGLE:
                return C.A_SHOOT

            best_mate, best_angle = self._best_angle_mate(engine, valid_mates)
            if best_mate is not None and best_angle > angle + self.ANGLE_IMPROVE_MARGIN:
                return self._pass_to_mate(mates, best_mate)

            return self._dribble_for_angle(engine, idx, pos)

        if valid_mates and self.rng.random() < self.PASS_PROB:
            return self._pick_pass(engine, idx, mates)

        return self._dribble_forward(engine, idx, pos)

        # Ensure we have valid forward targets before attempting a standard pass
        if valid_mates and self.rng.random() < self.PASS_PROB:
            return self._pick_pass(engine, idx, mates)

        return self._dribble_forward(engine, idx, pos)

    def _pick_pass(self, engine, idx: int, mates: list) -> int:
        scores = []
        for m in mates:
            # --- NEW TACTIC A RULE ---
            if self.tactic == "A" and m in self._defenders:
                scores.append(-9999.0)  # Prohibit attackers passing backward
            else:
                scores.append(engine.pos[m][0] + self.ANGLE_WEIGHT * self._shot_angle(engine.pos[m]))
        return self._pass_action(scores)

    def _pass_action(self, scores: list) -> int:
        # Filter out prohibited targets so they can't be randomly selected
        valid_slots = [i for i, s in enumerate(scores) if s > -9000]
        
        # Fallback if something goes wrong
        if not valid_slots:
            valid_slots = [0]  

        best_slot = max(valid_slots, key=lambda i: scores[i])
        
        if len(valid_slots) == 1 or self.rng.random() < self.PASS_BEST_PROB:
            chosen = best_slot
        else:
            others = [s for s in valid_slots if s != best_slot]
            chosen = int(self.rng.choice(others))
            
        return C.A_PASS_A + chosen

    def _pass_to_mate(self, mates: list, mate_idx: int) -> int:
        """Same encoding as _pass_action, but for a specific chosen mate."""
        return C.A_PASS_A + mates.index(mate_idx)

    def _support_run(self, engine, idx: int, carrier_idx: int) -> int:
        cfg = self.TACTICS[self.tactic]

        if idx not in self._defenders:
            if cfg.get("aggressive"):
                # 3-1's lone striker never just waits to be found — sprint
                # into the box so there's always a runner for the through-ball.
                target_x   = C.RIGHT_GOAL_X - 150
                y_lo, y_hi = self._y_limits(idx)
                target_y   = (y_lo + y_hi) / 2
                return self._move_toward(engine.pos[idx],
                                         np.array([target_x, target_y]))

            if cfg.get("attacking"):
                return self._attacking_run(engine, idx, carrier_idx)

        carrier_x  = engine.pos[carrier_idx][0]
        y_lo, y_hi = self._y_limits(idx)

        raw_x    = carrier_x + 120 + self.rng.uniform(-40, 40)
        target_x = float(np.clip(raw_x,
                                 carrier_x - 50,
                                 min(carrier_x + self.SUPPORT_X_LAG,
                                     C.RIGHT_GOAL_X - 60)))
        mid_y    = (y_lo + y_hi) / 2
        target_y = float(np.clip(mid_y + self.rng.uniform(-60, 60), y_lo, y_hi))

        return self._move_toward(engine.pos[idx],
                                 np.array([target_x, target_y]))

    def _attacking_run(self, engine, idx: int, carrier_idx: int) -> int:
        """
        1-3 shape: the two off-ball attackers stagger their movement instead
        of both drifting to the same generic support spot, so the carrier
        always has two distinct passing angles — one in behind, one wide.
        """
        y_lo, y_hi = self._y_limits(idx)
        carrier_x  = engine.pos[carrier_idx][0]

        if idx % 2 == 0:
            # run in behind, toward the six-yard area
            target_x = C.RIGHT_GOAL_X - 90
            target_y = float(np.clip(C.GOAL_CENTER_Y + self.rng.uniform(-40, 40),
                                     y_lo, y_hi))
        else:
            # hold width to keep a switch-of-play / cutback angle open
            target_x = float(np.clip(carrier_x + 60, carrier_x - 30,
                                     C.RIGHT_GOAL_X - 40))
            target_y = float(np.clip((y_lo + y_hi) / 2 + self.rng.uniform(-50, 50),
                                     y_lo, y_hi))

        return self._move_toward(engine.pos[idx], np.array([target_x, target_y]))

    def _second_ball(self, engine, idx: int) -> int:
        y_lo, y_hi = self._y_limits(idx)
        target = np.array([
            engine.ball_pos[0] - 50 + self.rng.uniform(-30, 30),
            float(np.clip((y_lo + y_hi) / 2 + self.rng.uniform(-40, 40),
                          y_lo, y_hi)),
        ])
        return self._move_toward(engine.pos[idx], target)

    # ═════════════════════════════════════════════════════════════════════
    #  DRIBBLING & SHOOTING-ANGLE HELPERS
    # ═════════════════════════════════════════════════════════════════════
    def _shot_angle(self, pos) -> float:
        """
        Angle (radians) subtended by the goal mouth as seen from `pos`.
        Bigger = a more open sight of goal (easier for the keeper to be
        beaten); near zero means the shot is coming in almost along the
        goal line and any keeper covers it easily.
        """
        pos = np.asarray(pos, dtype=float)
        top = np.array([C.RIGHT_GOAL_X, C.GOAL_Y_MIN])
        bot = np.array([C.RIGHT_GOAL_X, C.GOAL_Y_MAX])
        v1, v2 = top - pos, bot - pos
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        return float(np.arccos(cos_a))

    def _best_angle_mate(self, engine, mates: list):
        """Return (mate_idx, angle) for whichever mate has the most open
        sight of goal right now. (None, -1.0) if there are no mates."""
        if not mates:
            return None, -1.0
        scored = [(self._shot_angle(engine.pos[m]), m) for m in mates]
        angle, mate = max(scored, key=lambda t: t[0])
        return mate, angle

    def _dribble_for_angle(self, engine, idx: int, pos) -> int:
        """
        Too close/tight-angled to shoot cleanly and no mate is better
        placed: shift toward the goal's centreline before trying again,
        rather than either forcing a bad shot or standing still.
        """
        y_lo, y_hi = self._y_limits(idx)
        target_y   = pos[1] + (C.GOAL_CENTER_Y - pos[1]) * 0.6
        target_y   = float(np.clip(target_y, y_lo, y_hi))
        target     = np.array([pos[0] + 15, target_y])
        return self._move_toward(pos, target)

    def _dribble_forward(self, engine, idx: int, pos) -> int:
        """
        Carry the ball toward goal. If the nearest blue player is close and
        standing between us and the goal, juke sideways away from them
        instead of running straight into the challenge.
        """
        goal_target = np.array([C.RIGHT_GOAL_X, C.GOAL_CENTER_Y], dtype=float)

        blocker = H.nearest_index(pos, engine.pos, C.BLUE_FIELD)
        if blocker is not None:
            bpos  = engine.pos[blocker]
            ahead = bpos[0] > pos[0]
            if ahead and H.distance(pos, bpos) < self.DRIBBLE_JUKE_DIST:
                y_lo, y_hi = self._y_limits(idx)
                lateral    = -1.0 if bpos[1] >= pos[1] else 1.0
                juke_y     = float(np.clip(pos[1] + lateral * self.DRIBBLE_JUKE_Y,
                                           y_lo, y_hi))
                goal_target = np.array([pos[0] + 70, juke_y])

        goal_target = goal_target + self._jitter(self.NOISE_RADIUS)

        cap = self._striker_cap()
        if cap is not None and idx not in self._defenders:
            goal_target[0] = min(goal_target[0], cap)

        return self._move_toward(pos, goal_target)

    # ═════════════════════════════════════════════════════════════════════
    #  UTILITIES
    # ═════════════════════════════════════════════════════════════════════
    def _jitter(self, radius: float) -> np.ndarray:
        angle = self.rng.uniform(0, 2 * np.pi)
        r     = self.rng.uniform(0, radius)
        return np.array([r * np.cos(angle), r * np.sin(angle)])

    def _move_toward(self, from_pos, to_pos) -> int:
        if self.rng.random() > self.speed_mult:
            return C.A_STAY
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        best, best_score = C.A_STAY, -1.0
        for action, (vx, vy) in C.MOVE_VECTORS.items():
            score = dx * vx + dy * vy
            if score > best_score:
                best_score = score
                best       = action
        return best

    def __call__(self, engine) -> np.ndarray:
        return self.act(engine)
