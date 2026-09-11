"""
logger.py
=========
EpisodeLogger — writes per-episode stats to CSV and prints to terminal.

CSV columns include per-agent rewards, shot/pass accuracy, and concede reward.
Terminal shows rolling W/D/L, per-agent rewards, accuracy stats, and possession.
"""

from __future__ import annotations
import os
import csv
import math
import collections
import numpy as np

CSV_COLUMNS = [
    "rollout",
    "episode",
    "red_tactic",
    # score
    "blue_goals",
    "red_goals",
    "goal_diff",
    # shots
    "blue_shots",
    "blue_sot",
    "blue_shot_acc",       # sot / shots %
    "red_shots",
    "red_sot",
    "red_shot_acc",
    # passes
    "blue_passes",
    "blue_passes_ok",
    "blue_pass_acc",       # ok / passes %
    "blue_assists",
    "red_passes",
    "red_passes_ok",
    "red_pass_acc",
    # possession
    "blue_poss_pct",
    "red_poss_pct",
    # rewards — team total
    "total_reward",
    "rew_goal",
    "rew_concede",
    "rew_shaping",
    "rew_chase",
    "rew_lose_ball",       
    "rew_pass_success",   
    "rew_assist",          
    # rewards — per agent
    "agent_0_reward",
    "agent_1_reward",
    "agent_2_reward",
    "agent_3_reward",

"p0_dist", "p1_dist", "p2_dist", "p3_dist", "p4_dist",  # Blue Team (0-3 Field, 4 GK)
    "p5_dist", "p6_dist", "p7_dist", "p8_dist", "p9_dist",  # Red Team (5-8 Field, 9 GK)
    # --- NEW: Total Game Stamina ---
    "p0_tot_stam", "p1_tot_stam", "p2_tot_stam", "p3_tot_stam", "p4_tot_stam",
    "p5_tot_stam", "p6_tot_stam", "p7_tot_stam", "p8_tot_stam", "p9_tot_stam",

]



def _acc(num: int, den: int) -> float:
    """Safe percentage: num/den*100, or 0 if den==0."""
    return round(100.0 * num / den, 1) if den > 0 else 0.0


class EpisodeLogger:
    def __init__(self, csv_path: str, window: int = 50):
        self.csv_path = csv_path
        self.window   = window
        self._history: collections.deque = collections.deque(maxlen=window)

        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        is_new       = not os.path.isfile(csv_path)
        self._f      = open(csv_path, "a", newline="")
        self._writer = csv.DictWriter(self._f, fieldnames=CSV_COLUMNS)
        if is_new:
            self._writer.writeheader()
            self._f.flush()

    # ── Row builder ───────────────────────────────────────────────────────────
    def _row(self, rollout: int, episode: int, stats: dict) -> dict:
        rc           = stats.get("reward_components", {})
        agent_rews   = stats.get("agent_rewards", [0.0, 0.0, 0.0, 0.0])

        b_shots = stats.get("blue_shots", 0)
        b_sot   = stats.get("blue_shots_on_target", 0)
        r_shots = stats.get("red_shots", 0)
        r_sot   = stats.get("red_shots_on_target", 0)
        b_pass  = stats.get("blue_passes", 0)
        b_ok    = stats.get("blue_passes_success", 0)
        b_ast   = stats.get("blue_assists", 0)
        r_pass  = stats.get("red_passes", 0)

        r_ok    = stats.get("red_passes_success", 0)
        b_goals = stats.get("blue_goals", 0)
        r_goals = stats.get("red_goals", 0)
        dists = stats.get("agent_distance_run", [0.0, 0.0, 0.0, 0.0])
        tot_stams = stats.get("agent_total_stamina", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        return {
            "rollout":        rollout,
            "episode":        episode,
            "red_tactic":     stats.get("red_tactic", "?"), 
            "blue_goals":     b_goals,
            "red_goals":      r_goals,
            "goal_diff":      b_goals - r_goals,
            "blue_shots":     b_shots,
            "blue_sot":       b_sot,
            "blue_shot_acc":  _acc(b_sot, b_shots),
            "red_shots":      r_shots,
            "red_sot":        r_sot,
            "red_shot_acc":   _acc(r_sot, r_shots),
            "blue_passes":    b_pass,
            "blue_passes_ok": b_ok,
            "blue_pass_acc":  _acc(b_ok, b_pass),
            "blue_assists":   b_ast,
            "red_passes":     r_pass,
            "red_passes_ok":  r_ok,
            "red_pass_acc":   _acc(r_ok, r_pass),
            "blue_poss_pct":  round(stats.get("blue_possess_percent", 50.0), 2),
            "red_poss_pct":   round(stats.get("red_possess_percent",  50.0), 2),
            "total_reward":   round(stats.get("total_reward", 0.0), 4),
            "rew_goal":       round(rc.get("goal",    0.0), 4),
            "rew_concede":    round(rc.get("concede", 0.0), 4),
            "rew_shaping":    round(rc.get("shaping", 0.0), 4),
            "rew_chase": round(rc.get("chase", 0.0), 4),
            "rew_lose_ball":    round(rc.get("turnover", 0.0), 4),     # <-- changed from "lose_ball"
            "rew_pass_success": round(rc.get("pass",     0.0), 4),     # <-- changed from "pass_success"
            "rew_assist":       round(rc.get("assist",   0.0), 4),     # <-- now matches reward.py
            
            "agent_0_reward": round(agent_rews[0] if len(agent_rews) > 0 else 0.0, 4),
            "agent_1_reward": round(agent_rews[1] if len(agent_rews) > 1 else 0.0, 4),
            "agent_2_reward": round(agent_rews[2] if len(agent_rews) > 2 else 0.0, 4),
            "agent_3_reward": round(agent_rews[3] if len(agent_rews) > 3 else 0.0, 4),
            "p0_dist": round(dists[0] if len(dists) > 0 else 0.0, 1),
            "p1_dist": round(dists[1] if len(dists) > 1 else 0.0, 1),
            "p2_dist": round(dists[2] if len(dists) > 2 else 0.0, 1),
            "p3_dist": round(dists[3] if len(dists) > 3 else 0.0, 1),
            "p4_dist": round(dists[4] if len(dists) > 4 else 0.0, 1),
            "p5_dist": round(dists[5] if len(dists) > 5 else 0.0, 1),
            "p6_dist": round(dists[6] if len(dists) > 6 else 0.0, 1),
            "p7_dist": round(dists[7] if len(dists) > 7 else 0.0, 1),
            "p8_dist": round(dists[8] if len(dists) > 8 else 0.0, 1),
            "p9_dist": round(dists[9] if len(dists) > 9 else 0.0, 1),
            # --- NEW: Total Game Stamina ---
            "p0_tot_stam": round(tot_stams[0] if len(tot_stams) > 0 else 1.0, 2),
            "p1_tot_stam": round(tot_stams[1] if len(tot_stams) > 1 else 1.0, 2),
            "p2_tot_stam": round(tot_stams[2] if len(tot_stams) > 2 else 1.0, 2),
            "p3_tot_stam": round(tot_stams[3] if len(tot_stams) > 3 else 1.0, 2),
            "p4_tot_stam": round(tot_stams[4] if len(tot_stams) > 4 else 1.0, 2),
            "p5_tot_stam": round(tot_stams[5] if len(tot_stams) > 5 else 1.0, 2),
            "p6_tot_stam": round(tot_stams[6] if len(tot_stams) > 6 else 1.0, 2),
            "p7_tot_stam": round(tot_stams[7] if len(tot_stams) > 7 else 1.0, 2),
            "p8_tot_stam": round(tot_stams[8] if len(tot_stams) > 8 else 1.0, 2),
            "p9_tot_stam": round(tot_stams[9] if len(tot_stams) > 9 else 1.0, 2),
        
        }
    # ── Formatting helpers ────────────────────────────────────────────────────
    @staticmethod
    def _fmt(v) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "  n/a"
        return f"{v:+.4f}" if abs(v) < 10 else f"{v:+.2f}"

    @staticmethod
    def _bar(pct: float, width: int = 10) -> str:
        """Simple ASCII possession bar, e.g. [████░░░░░░] 43%"""
        filled = round(pct / 100 * width)
        return f"[{'█' * filled}{'░' * (width - filled)}] {pct:.0f}%"

    # ── Public API ────────────────────────────────────────────────────────────
    def log_episode(self, rollout: int, episode: int, stats: dict) -> None:
        row = self._row(rollout, episode, stats)
        self._writer.writerow(row)
        self._f.flush()
        self._history.append(row)

    def print_terminal(
        self,
        episode:   int,
        all_stats: list[dict],
        losses:    dict,
        elapsed_s: float,
    ) -> None:
        n     = len(all_stats)
        mins  = elapsed_s / 60.0

        # ── This rollout ──────────────────────────────────────────────────────
        c_bg   = float(np.mean([s.get("blue_goals",   0) for s in all_stats]))
        c_rg   = float(np.mean([s.get("red_goals",    0) for s in all_stats]))
        c_rew  = float(np.mean([s.get("total_reward", 0) for s in all_stats]))
        c_poss = float(np.mean([s.get("blue_possess_percent", 50) for s in all_stats]))

        wins       = sum(1 for s in all_stats if s.get("blue_goals", 0) > s.get("red_goals", 0))
        draws      = sum(1 for s in all_stats if s.get("blue_goals", 0) == s.get("red_goals", 0))
        losses_n   = n - wins - draws
        result_tag = "W" if wins > losses_n else ("L" if losses_n > wins else "D")

        # shot / pass accuracy this rollout
        b_shots = sum(s.get("blue_shots", 0)            for s in all_stats)
        b_sot   = sum(s.get("blue_shots_on_target", 0)  for s in all_stats)
        b_pass  = sum(s.get("blue_passes", 0)           for s in all_stats)
        b_ok    = sum(s.get("blue_passes_success", 0)   for s in all_stats)

        # per-agent rewards this rollout
        agent_rews = [
            float(np.mean([
                s.get("agent_rewards", [0,0,0,0])[i]
                for s in all_stats
                if len(s.get("agent_rewards", [])) > i
            ]))
            for i in range(4)
        ]
        # per-player average distance run this rollout (10 players total)
        player_dists = [
            float(np.mean([
                s.get("agent_distance_run", [0]*10)[i]
                for s in all_stats
                if len(s.get("agent_distance_run", [])) > i
            ]))
            for i in range(10)
        ]

        # ── Rolling window ────────────────────────────────────────────────────
        hn = len(self._history)
        if hn:
            hist        = list(self._history)
            avg_rew     = float(np.mean([h["total_reward"]  for h in hist]))
            avg_bg      = float(np.mean([h["blue_goals"]    for h in hist]))
            avg_rg      = float(np.mean([h["red_goals"]     for h in hist]))
            wr          = float(np.mean([1 if h["blue_goals"] > h["red_goals"] else 0
                                         for h in hist])) * 100
            dr          = float(np.mean([1 if h["blue_goals"] == h["red_goals"] else 0
                                         for h in hist])) * 100
            avg_poss    = float(np.mean([h["blue_poss_pct"]  for h in hist]))
            avg_shot_a  = float(np.mean([h["blue_shot_acc"]  for h in hist]))
            avg_pass_a  = float(np.mean([h["blue_pass_acc"]  for h in hist]))
            avg_goal_r  = float(np.mean([h["rew_goal"]       for h in hist]))
            avg_shape_r = float(np.mean([h["rew_shaping"]    for h in hist]))
            avg_chase_r = float(np.mean([h["rew_chase"] for h in hist]))
            avg_conc_r  = float(np.mean([h["rew_concede"]    for h in hist]))
            avg_ag_r    = [float(np.mean([h[f"agent_{i}_reward"] for h in hist]))
                           for i in range(4)]
        else:
            avg_rew = avg_bg = avg_rg = wr = dr = 0.0
            avg_poss = avg_shot_a = avg_pass_a = 50.0
            avg_goal_r = avg_shape_r = avg_conc_r = avg_chase_r = 0.0
            avg_ag_r = [0.0, 0.0, 0.0, 0.0]

        pl  = losses.get("policy_loss")
        vl  = losses.get("value_loss")
        ent = losses.get("entropy")


        # --- ADD THIS BLOCK: Count tactics used in this batch ---
        tactics_used = [s.get("red_tactic", "?") for s in all_stats]
        tactic_counts = {t: tactics_used.count(t) for t in sorted(set(tactics_used))}
        tactic_str = " ".join([f"{k}:{v}" for k, v in tactic_counts.items()])
        # ── Print ─────────────────────────────────────────────────────────────
        print("─" * 80)
        print(
            f"  Ep {episode:>6}  │  "
            f"{result_tag} {wins}W-{draws}D-{losses_n}L  "
            f"Score {c_bg:.1f}-{c_rg:.1f}  "
            f"R={c_rew:+.2f}  "
            f"│  {mins:.1f}min"
        )
        print(
            f"  This rollout  │  "
            f"SoT {_acc(b_sot, b_shots):.0f}%  "
            f"Pass {b_ok}/{b_pass} ({_acc(b_ok, b_pass):.0f}%)  "
            f"Poss {self._bar(c_poss, 8)}"
            f"Tac[{tactic_str}]"
        )
        print(
            f"  Agents  A0={agent_rews[0]:+.2f}  "
            f"A1={agent_rews[1]:+.2f}  "
            f"A2={agent_rews[2]:+.2f}  "
            f"A3={agent_rews[3]:+.2f}"
        )
        print(
            f"  Roll({hn:>3})   │  "
            f"{avg_bg:.2f}-{avg_rg:.2f}  "
            f"WR={wr:.0f}% DR={dr:.0f}%  "
            f"R={avg_rew:+.2f}  "
            f"Poss={avg_poss:.0f}%  "
            f"SoT={avg_shot_a:.0f}%  "
            f"Pass={avg_pass_a:.0f}%"
        )
        print(
            f"  Rew breakdown  │  "
            f"goal={avg_goal_r:+.3f}  "
            f"concede={avg_conc_r:+.3f}  "
            f"shaping={avg_shape_r:+.3f}"
        )
        print(
            f"  Agents avg     │  "
            f"A0={avg_ag_r[0]:+.2f}  "
            f"A1={avg_ag_r[1]:+.2f}  "
            f"A2={avg_ag_r[2]:+.2f}  "
            f"A3={avg_ag_r[3]:+.2f}"
        )
        print(
            f"  Loss  π={self._fmt(pl)}  "
            f"v={self._fmt(vl)}  "
            f"H={self._fmt(ent)}"
        )

    def close(self) -> None:
        self._f.close()