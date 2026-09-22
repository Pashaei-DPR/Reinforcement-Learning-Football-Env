Old paper
------------------------------------------------------------------------------------
Short-Term Gains vs. Long-Term Success: Reward Strategy Design for Reinforcement Learning in Football

https://scholar.google.com/scholar?oi=bibs&cluster=6960816829086480145&btnI=1&hl=en

https://scholar.google.com/citations?view_op=view_citation&hl=en&user=_C4Iif8AAAAJ&sortby=pubdate&citation_for_view=_C4Iif8AAAAJ:cP5_aSlQDF4C

Abstract— Reinforcement learning in complex games like
soccer relies heavily on how you define your reward function
and environment. In this work, we developed a custom 3v3
soccer environment and implemented two RL-based teams with
distinct learning trends: one with a fast convergence but limited
long-term adaptation, and another with a slower yet more
robust learning trajectory. Simulation shows that despite
performing better at the start, the short-term agents fall short
of the performance of the long-term agents in the long run, and
after passing 50% of the episodes, the win rate of long-term
agents rises from 30% in the beginning to 50%.
Keywords— Reinforcement Learning, Multi-agent systems,
Soccer Simulation.


___________________________________________

              latest Version V8
___________________________________________
<img width="995" height="881" alt="image" src="https://github.com/user-attachments/assets/49617df7-6283-4bb5-a8b3-baccab552a40" />

# Grid Football 5v5: Multi-Agent RL Simulation

This repository contains a custom 5v5 grid-football simulation environment built for Multi-Agent Reinforcement Learning (MARL). The project features a pure-NumPy physics engine wrapped in a PettingZoo `ParallelEnv`, trained using Independent Proximal Policy Optimization (IPPO) via Stable-Baselines3's `MaskablePPO`.

## Core Features

* **Fast, Standalone Physics Engine**: The underlying `FootballEngine` handles ball physics, friction, bounding, player movement, collisions, and stamina entirely in NumPy, allowing for highly efficient parallel rollouts without relying on heavy external physics libraries.


* **Asymmetric Actor-Critic Architecture**: Uses a custom Multi-Layer Perceptron (MLP) extractor where the Critic observes the full environment state (69 dimensions, including the opponent's active tactic), while the Actor is blind to the opponent's hidden tactic (66 dimensions).


* **Action Masking**: Agents are prevented from executing invalid actions (such as passing or shooting when they do not possess the ball) through dynamic action masking.


* **Dynamic Rule-Based Opponent**: The opposing Red Team is controlled by a sophisticated rule-based AI that dynamically switches between three tactical formations (A: 2-2 Balanced, B: 3-1 Defensive/Aggressive, C: 1-3 Attacking) based on the game state and score.


* **Advanced Reward Shaping**: Employs potential-based reward shaping for ball positioning and chasing, alongside shared and individual sparse rewards for goals, assists, passes, and turnovers.


* **Stamina System**: Players experience both short-term fatigue (sprint/recovery) and long-term match fatigue (based on total distance run), which dynamically scales their movement speed.



## Installation

Ensure you have Python 3.8+ installed. You will need `torch`, `stable-baselines3`, `sb3-contrib`, `pettingzoo`, `pygame`, and `numpy`.

```bash
pip install numpy torch pettingzoo pygame stable-baselines3 sb3-contrib

```

## Usage

### 1. Training the Agents

To train the Blue Team agents from scratch using MaskablePPO, run the main training script. This utilizes a custom `FootballVecEnv` to run 8 parallel matches (32 total agent slots) efficiently.

```bash
python main.py

```

To resume training from the latest checkpoint:

```bash
python main.py --resume

```

### 2. Watching a Trained Model

You can evaluate and watch a trained agent play against the rule-based Red Team using the PyGame renderer.

```bash
# Auto-loads the latest checkpoint in the checkpoints/ directory
python watch.py

# Watch a specific checkpoint for 5 episodes
python watch.py --model checkpoints/m1_300.zip --episodes 5

# Watch in slow motion to inspect agent behavior
python watch.py --slow

```

### 3. Play Manually (Human vs AI)

Test the mechanics or challenge the rule-based Red Team yourself. The test script allows three players to manually control the Blue Team using a shared keyboard.

```bash
python test.py

```

**Controls**:

* **Blue 0**: `W/A/S/D`
* **Blue 1**: `T/F/G/H`
* **Blue 2**: `U/H/J/K`
* **Blue 3**: `Arrow Keys`
* **Actions (Shared)**: `L-SHIFT` to Shoot, `1`/`2`/`3` to Pass to teammates.

## Environment Details

### Action Space

Each field player has a discrete action space of size 13:

* **0-7**: Movement vectors (Up, Down, Left, Right, and diagonals).


* **8**: Shoot.


* **9**: Stay (recovers short-term stamina).


* **10-12**: Pass to relative teammate A, B, or C.



### Observation Space

The environment provides an egocentric, normalized observation vector.

* **Actor (66 dims)**: Includes self position/velocity/stamina, ball position/velocity, teammate data, distance-gated opponent data (field players and goalkeeper), game time, score difference, and agent ID.


* **Critic (69 dims)**: Includes all Actor observations plus a one-hot encoding of the Red Team's currently active tactic (A, B, or C).



### Opponent AI (Red Team)

The Red Team uses a state-machine rule-based controller `RedController`.

* **Tactic A (2-2)**: Balanced formation with two attackers and two defenders.


* **Tactic B (3-1)**: Protects a lead. Features a deep back line that can carry the ball aggressively, and a lone striker locked strictly to the central y-axis.


* **Tactic C (1-3)**: Chases the game. The attacking trio utilizes staggered support runs to provide multiple passing angles for the ball carrier.



## Project Structure

* `config.py`: Global configuration parameters, field geometry, speeds, and action definitions.


* `env.py`: The `FootballEngine` and PettingZoo `GridFootballEnv` wrapper.


* `helper.py`: NumPy geometry math, speed calculations, clamping, and grid discretizations.


* `red_control.py`: The rule-based AI logic for the opponent team.


* `reward.py`: Reward initialization and shaping logic.


* `logger.py`: Custom `EpisodeLogger` to track and format terminal outputs and CSV training logs (goals, pass accuracy, specific rewards).


* `main.py`: The IPPO training loop utilizing `sb3-contrib`.


* `watch.py`: Evaluation script with a PyGame visualizer.


* `test.py`: Interactive human-play script.







