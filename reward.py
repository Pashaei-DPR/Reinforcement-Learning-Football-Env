from __future__ import annotations
import numpy as np
import config as C


class RewardCalculator:
    """
    Reward calculator with potential-based shaping (Ng et al. 1999).

    Shared rewards (all 4 agents receive equally):
        ball shaping   — every step,  F = reward_ball_c * (γ·Φ(s') − Φ(s))
        team concede   — −0.475 each  when red scores (−1.9 total,
                          symmetric to the +1.9 total of a scored goal)

    Individual rewards on goal scored:
        scorer         — +1.0
        other 3        — +0.3 each   (total +1.9)

    Bookkeeping convention: `total_reward` and every entry in
    `reward_components` are in TOTAL-DELIVERED units (summed across the
    4 agents), so  total_reward == goal + concede + shaping  always holds.
    """

    def __init__(self, config: C.Config = C.DEFAULT_CONFIG):
        self.cfg = config
        self._team       = 0.0
        self._individual = [0.0, 0.0, 0.0, 0.0]
        self._prev_phi   = 0.0
        self._prev_chase_phi = [0.0, 0.0, 0.0, 0.0] 
        self.total_reward = 0.0
        self.agent_total_rewards = [0.0, 0.0, 0.0, 0.0]
        self.reward_components = {
            'goal':    0.0,
            'concede': 0.0,
            'shaping': 0.0,
            'chase':   0.0,
            'pass':    0.0,
            'turnover': 0.0,
            'assist':  0.0,  
        }

  

    def reset(self):
        self._team       = 0.0
        self._individual = [0.0, 0.0, 0.0, 0.0]
        self._prev_phi   = 0.0
        self._prev_chase_phi = [0.0, 0.0, 0.0, 0.0]
        self.total_reward = 0.0
        self.agent_total_rewards = [0.0, 0.0, 0.0, 0.0]
        self.reward_components = {
            'goal':    0.0,
            'concede': 0.0,
            'shaping': 0.0,
            'chase':   0.0,
            'pass':    0.0,
            'turnover': 0.0,
            'assist':  0.0,
        }

    # ── Potential function ────────────────────────────────────────────────────
    def _phi(self, ball_pos) -> float:
        d = float(np.hypot(
            ball_pos[0] - C.LEFT_GOAL_X,
            ball_pos[1] - C.GOAL_CENTER_Y,
        ))
        return -d / C.FIELD_DIAG
    #-----chase------
    def _chase_phi(self, agent_pos, ball_pos) -> float:
        """Potential for one agent closing the distance to the ball: -dist/diag."""
        d = float(np.hypot(agent_pos[0] - ball_pos[0], agent_pos[1] - ball_pos[1]))
        return -d / C.FIELD_DIAG
    # ── Called by the engine ──────────────────────────────────────────────────
    def refresh_prev_distances(self, ball_pos, pos):
        self._prev_phi = self._phi(ball_pos)
        blue_field = list(C.BLUE_FIELD)
        for slot, idx in enumerate(blue_field):
            self._prev_chase_phi[slot] = self._chase_phi(pos[idx], ball_pos)

    def shaping(self, owner, pos, ball_pos):
        """Shared ball shaping — goes into _team (each agent receives it)."""
        if self.cfg.reward_ball_c == 0.0:
            return
        phi    = self._phi(ball_pos)
        shaped = self.cfg.reward_ball_c * (self.cfg.gamma * phi - self._prev_phi)
        self._team                        += shaped
        # delivered units: all 4 agents receive `shaped`
        self.reward_components['shaping'] += shaped * 4.0
        self._prev_phi = phi


    def chase_shaping(self, owner, pos, ball_pos):
        if self.cfg.reward_chase_c == 0.0:
            return
        blue_field = list(C.BLUE_FIELD)
        
        for slot, idx in enumerate(blue_field):
            phi = self._chase_phi(pos[idx], ball_pos)
            
            # CHANGED: Now checks C.BLUE_ALL so the GK is included
            eligible = owner not in C.BLUE_ALL
            
            if eligible:
                shaped = self.cfg.reward_chase_c * (
                    self.cfg.gamma * phi - self._prev_chase_phi[slot])
                self._individual[slot]          += shaped
                self.reward_components['chase'] += shaped
            self._prev_chase_phi[slot] = phi

            
    def on_shot_on_target(self, shooter_idx): 

        pass

    def on_pass_success(self, passer_idx, receiver_idx):

        pass


# ── Step boundaries ───────────────────────────────────────────────────────
    def begin_step(self):
        self._team       = 0.0
        self._individual = [0.0, 0.0, 0.0, 0.0]

    # The absolute total value of a goal (both for scoring and conceding) is fixed to 2.6
    @property
    def GOAL_TOTAL(self) -> float:
        return self.cfg.goal_total


    

    def on_goal_scored(self,
                       scorer_idx:   int | None,
                       assister_idx: int | None = None):
        
        # Distribute reward evenly if no specific RL agent scored
        if scorer_idx is None:
            share = self.GOAL_TOTAL / 4.0
            for i in range(4):
                self._individual[i] += share
            self.reward_components['goal'] += self.GOAL_TOTAL
            return

        total_awarded  = 0.0   # everything EXCEPT the assister's share
        assist_awarded = 0.0   # the assister's share only

        for i in range(4):
            if i == scorer_idx:
                # Scorer gets 1.0 if assisted, but is punished down to 0.8 if solo
                reward = 1.0 if assister_idx is not None else 0.8
                self._individual[i] += reward
                total_awarded += reward
            elif i == assister_idx:
                # Assister gets equal glory to the scorer
                reward = 1.0
                self._individual[i] += reward
                assist_awarded += reward        # ← goes to 'assist' ONLY
            else:
                # Bystanders get 0.3 if assisted, but a bumped 0.6 if solo
                reward = 0.3 if assister_idx is not None else 0.6
                self._individual[i] += reward
                total_awarded += reward

        self.reward_components['goal']   += total_awarded
        self.reward_components['assist'] += assist_awarded
        # ---------------------------------------------------------

  
  
  
    def on_goal_conceded(self):
        # Symmetric to a scored goal: −2.6 total, shared equally (−0.65 each)
        share = -self.GOAL_TOTAL / 4.0
        self._team                        += share
        self.reward_components['concede'] += -self.GOAL_TOTAL
    #---------lose ball-----------
    def on_turnover(self, player_idx):
        if self.cfg.reward_turnover_c == 0.0:
            return
        self._individual[player_idx] -= self.cfg.reward_turnover_c
        self.reward_components['turnover'] -= self.cfg.reward_turnover_c



    def end_step(self) -> np.ndarray:
        out  = np.full(4, self._team, dtype=np.float32)
        out += np.array(self._individual, dtype=np.float32)
        self.total_reward += float(out.sum())
        for i in range(4):
            self.agent_total_rewards[i] += float(out[i])
        return out
