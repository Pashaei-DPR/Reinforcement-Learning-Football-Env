"""
main.py
=======
IPPO with MaskablePPO (Stable-Baselines3 + sb3-contrib).
No SuperSuit — uses a self-contained custom VecEnv instead.


Usage
-----
    python main.py              # fresh 
    python main.py --resume     
"""

from __future__ import annotations
import os
import glob
import time
import argparse
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

import config as C


from env import GridFootballEnv
from logger import EpisodeLogger


# ── Suppress oneDNN info noise ────────────────────────────────────────────────
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",  "3")

# ── Hyperparameters ──────────────────────────────────────────────────────────
N_ENVS         = 8             # parallel matches running simultaneously
N_AGENTS       = 4             # blue field players per match
N_SLOTS        = N_ENVS * N_AGENTS   # total VecEnv slots = 32

N_STEPS        = C.MAX_STEPS   # steps per slot per update = 1 full episode
BATCH_SIZE     = 4800         # divides rollout exactly: 3600*32 = 115200 = 24 * 4800
N_EPOCHS       = 4
GAE_LAMBDA     = 0.95
CLIP_RANGE     = 0.2
ENT_COEF       = 0.03 #0.01#01  0.03 till 1700 roloput   -from rollout 1 till 1700 the ent was 0.03  then from 1700 till 2500 was 0.01  now 0,001 till 3300


VF_COEF        = 0.5
MAX_GRAD_NORM  = 0.5
LR             = 3e-4
SAVE_EVERY     = 100
SAVE_DIR       = "checkpoints"
LOG_PATH       = "training_log.csv"
TOTAL_STEPS    = 500_000_000

AGENTS = ["blue_0", "blue_1", "blue_2", "blue_3"]


# ── Custom VecEnv ─────────────────────────────────────────────────────────────
class FootballVecEnv(VecEnv):
    """
    Wraps N_ENVS independent GridFootballEnv instances as one VecEnv.
    Total slots = N_ENVS * N_AGENTS = 32.
    MaskablePPO trains a single shared policy across all 32 slots.

    Slot layout:
    slots  0, 1, 2, 3  →  env 0  (blue_0, blue_1, blue_2, blue_3)
    slots  4, 5, 6, 7  →  env 1
    slots  8, 9,10,11  →  env 2
    slots 12,13,14,15  →  env 3
    slots 16,17,18,19  →  env 4
    slots 20,21,22,23  →  env 5
    slots 24,25,26,27  →  env 6
    slots 28,29,30,31  →  env 7

    Each env resets independently when its episode ends.
    Because all envs run for the same MAX_STEPS, in practice they all
    finish simultaneously — giving 8 complete episodes per PPO update.
    """

    def __init__(self, start_seed: int = 42):  
        # 2. Pass a unique, offset seed to each environment instance
        self.par_envs = [
            GridFootballEnv(render_mode=None, seed=start_seed + i) 
            for i in range(N_ENVS)
        ]
        obs_space     = self.par_envs[0].observation_space(AGENTS[0])
        act_space     = self.par_envs[0].action_space(AGENTS[0])
        super().__init__(N_SLOTS, obs_space, act_space)

        self._actions: np.ndarray | None = None
        self._masks   = np.ones((N_SLOTS, act_space.n), dtype=bool)
        self._infos   = [{} for _ in range(N_SLOTS)]

    # ── VecEnv required interface ─────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        all_obs = []
        for e, env in enumerate(self.par_envs):
            obs_dict, info_dict = env.reset()
            self._store_env_masks(e, info_dict)
            all_obs.extend([obs_dict[a] for a in AGENTS])
        return np.stack(all_obs).astype(np.float32)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        all_obs   = []
        all_rews  = []
        all_dones = []
        all_infos = []

        for e, env in enumerate(self.par_envs):
            base        = e * N_AGENTS
            action_dict = {a: int(self._actions[base + i]) for i, a in enumerate(AGENTS)}

            obs_dict, rew_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)

            obs   = [obs_dict[a] for a in AGENTS]
            rews  = [rew_dict.get(a, 0.0) for a in AGENTS]
            dones = [trunc_dict.get(a, False) or term_dict.get(a, False)
                     for a in AGENTS]
            infos = [{**info_dict.get(a, {})} for a in AGENTS]

            if dones[0]:   # all 4 agents in this env finish together
                # collect stats BEFORE reset so they aren't wiped
                stats = env.episode_stats()
                for i in range(N_AGENTS):
                    infos[i]["episode_stats"]        = stats
                    infos[i]["terminal_observation"] = np.array(obs[i],
                                                                dtype=np.float32)
                    # episode ended by time limit, not a true terminal state:
                    # tells SB3 to bootstrap gamma * V(terminal_observation)
                    infos[i]["TimeLimit.truncated"]  = True
                # auto-reset this env
                new_obs_dict, new_info_dict = env.reset()
                obs = [new_obs_dict[a] for a in AGENTS]
                self._store_env_masks(e, new_info_dict)
            else:
                self._store_env_masks(e, info_dict)

            all_obs.extend(obs)
            all_rews.extend(rews)
            all_dones.extend(dones)
            all_infos.extend(infos)

        self._infos = all_infos
        return (
            np.stack(all_obs).astype(np.float32),
            np.array(all_rews, dtype=np.float32),
            np.array(all_dones),
            all_infos,
        )

    def close(self) -> None:
        for env in self.par_envs:
            env.close()

    def render(self, mode: str = "human"):
        return self.par_envs[0].render()

    def seed(self, seed=None):
        # 3. Update to properly register seeds if requested by SB3
        if seed is None:
            return [None] * self.num_envs
        for i, env in enumerate(self.par_envs):
            env.reset(seed=seed + i)
        return [seed + i for i in range(self.num_envs)]

    # ── Action masking (required by MaskablePPO) ──────────────────────────────
    def action_masks(self) -> np.ndarray:
        """Return current valid-action masks, shape (N_SLOTS, n_actions)."""
        return self._masks.copy()

    # ── VecEnv attribute helpers ───────────────────────────────────────────────
    def has_attr(self, attr_name: str, indices=None):
        return [hasattr(self, attr_name)] * self.num_envs

    def env_method(self, method_name: str, *args, indices=None, **kwargs):
        if method_name == "action_masks":
            return list(self._masks)   # list of N_SLOTS bool arrays (n_actions,)
        method = getattr(self, method_name, None)
        if method:
            return [method(*args, **kwargs)] * self.num_envs
        return [None] * self.num_envs

    def get_attr(self, attr_name: str, indices=None):
        return [getattr(self, attr_name, None)] * self.num_envs

    def set_attr(self, attr_name: str, value, indices=None):
        setattr(self, attr_name, value)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    # ── Internal ──────────────────────────────────────────────────────────────
    def _store_env_masks(self, env_idx: int, info_dict: dict) -> None:
        """Write action masks for the 4 slots that belong to env_idx."""
        base = env_idx * N_AGENTS
        for i, a in enumerate(AGENTS):
            mask = info_dict.get(a, {}).get("action_mask")
            self._masks[base + i] = (mask.astype(bool) if mask is not None
                                     else np.ones(self.action_space.n, dtype=bool))


# ── Callback ──────────────────────────────────────────────────────────────────
class FootballCallback(BaseCallback):
    """
    After every step, checks the anchor slot of each env (slots 0, 4, 8, 12, 16, 20, 24, 28).
    When an env finishes:
      - logs its episode to CSV
    When ALL finished envs in this step are processed:
      - prints one aggregated terminal summary
      - saves checkpoint every SAVE_EVERY episodes
    """

    def __init__(self, ep_logger: EpisodeLogger, verbose: int = 0):
        super().__init__(verbose)
        self.ep_logger = ep_logger
        self.episode   = 0
        self.rollout   = 0
        self.next_save = SAVE_EVERY
        self.t_start   = time.time()

    
    #def _on_step(self) -> bool:
    def _on_step(self) -> bool:
        anchor_slots = [e * N_AGENTS for e in range(N_ENVS)]
        finished: list[tuple[int, dict]] = []

        for slot in anchor_slots:
            if not self.locals["dones"][slot]:
                continue
            stats = self.locals["infos"][slot].get("episode_stats")
            if stats is None:
                continue
            self.episode += 1
            finished.append((self.episode, stats))

        if finished:
            self.rollout += 1
            for ep_num, stats in finished:
                self.ep_logger.log_episode(self.rollout, ep_num, stats)
            finished_stats = [s for _, s in finished]
            lv = self.model.logger.name_to_value
            losses = {
                "policy_loss": lv.get("train/policy_gradient_loss", float("nan")),
                "value_loss":  lv.get("train/value_loss",           float("nan")),
                "entropy":     abs(lv.get("train/entropy_loss",     float("nan"))),
            }
            self.ep_logger.print_terminal(
                self.episode, finished_stats, losses, time.time() - self.t_start
            )
            if self.episode >= self.next_save:
                self._save()
                self.next_save += SAVE_EVERY

        return True
    
    def _save(self) -> None:
        os.makedirs(SAVE_DIR, exist_ok=True)
        end   = (self.episode // SAVE_EVERY) * SAVE_EVERY
        start = end - SAVE_EVERY + 1
        path  = os.path.join(SAVE_DIR, f"m{start}_{end}")
        self.model.save(path)
        print(f"  ✓ Checkpoint → {path}.zip  (ep {self.episode})")

    def _on_training_end(self) -> None:
        self.ep_logger.close()


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def find_latest_checkpoint() -> "str | None":
    if not os.path.isdir(SAVE_DIR):
        return None
    files = glob.glob(os.path.join(SAVE_DIR, "m*_*.zip"))
    if not files:
        return None

    def end_ep(f: str) -> int:
        try:
            return int(os.path.basename(f).split("_")[1].replace(".zip", ""))
        except Exception:
            return 0

    return max(files, key=end_ep)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="IPPO Grid Football — SB3")
    parser.add_argument("--resume", action="store_true",
                        help="Auto-resume from latest checkpoint")
    # 1. Added CLI flag for seed selection
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # 2. Set global seeds across all core libraries
    MASTER_SEED = args.seed
    torch.manual_seed(MASTER_SEED)
    np.random.seed(MASTER_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MASTER_SEED)

    # 3. Pass the MASTER_SEED to your environment wrapper
    env    = FootballVecEnv(start_seed=MASTER_SEED)
    logger = EpisodeLogger(LOG_PATH)
    cb     = FootballCallback(logger)

    ckpt = None
    if args.resume or os.path.isdir(SAVE_DIR):
        ckpt = find_latest_checkpoint()

    if ckpt:
        print(f"Resuming from {ckpt}")
        model = MaskablePPO.load(ckpt, env=env, device="cpu")
        try:
            saved_ep     = int(os.path.basename(ckpt).split("_")[1].replace(".zip", ""))
            cb.episode   = saved_ep
            cb.rollout   = saved_ep // N_ENVS
            cb.next_save = ((saved_ep // SAVE_EVERY) + 1) * SAVE_EVERY
        except Exception:
            pass
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            n_steps       = N_STEPS,
            batch_size    = BATCH_SIZE,
            n_epochs      = N_EPOCHS,
            gamma         = C.GAMMA,
            gae_lambda    = GAE_LAMBDA,
            clip_range    = CLIP_RANGE,
            ent_coef      = ENT_COEF,
            vf_coef       = VF_COEF,
            max_grad_norm = MAX_GRAD_NORM,
            learning_rate = LR,
            policy_kwargs = dict(net_arch=[256, 256]),
            verbose       = 0,
            device        = "cpu",
            seed          = MASTER_SEED,  # <--- Added this line
        )

    total_params = sum(p.numel() for p in model.policy.parameters())
    rollout_size = N_STEPS * N_SLOTS
    print(f"\n{'═'*70}")
    print(f"  IPPO Grid Football — MaskablePPO (SB3)  [{N_ENVS} parallel envs]")
    print(f"  OBS={C.OBS_DIM}  ACTIONS={C.N_ACTIONS}  "
          f"HORIZON={C.MAX_STEPS}  PARAMS={total_params:,}")
    print(f"  envs={N_ENVS}  slots={N_SLOTS}  rollout={rollout_size:,} steps/update")
    print(f"  γ={C.GAMMA}  λ={GAE_LAMBDA}  lr={LR}  clip={CLIP_RANGE}  "
          f"epochs={N_EPOCHS}  batch={BATCH_SIZE}")
    print(f"  log → {LOG_PATH}   checkpoints → {SAVE_DIR}/")
    print(f"  Stop: Ctrl-C  (emergency checkpoint auto-saved)")
    print(f"{'═'*70}\n")

    try:
        model.learn(
            total_timesteps     = TOTAL_STEPS,
            callback            = cb,
            reset_num_timesteps = (ckpt is None),
        )
    except KeyboardInterrupt:
        print("\nTraining stopped.")
        path = os.path.join(SAVE_DIR, f"emergency_ep{cb.episode}")
        os.makedirs(SAVE_DIR, exist_ok=True)
        model.save(path)
        print(f"Emergency save → {path}.zip")
    finally:
        logger.close()
        print(f"Log written to {LOG_PATH}")


if __name__ == "__main__":
    main()