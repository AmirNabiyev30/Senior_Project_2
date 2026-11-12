# AI Alignment in Robotic Simulations

CSc 59867-E Senior Project II (Fall 2026), City College of New York
Team: Mahdi Mahin, Amir Nabiyev

## What this project is about

When a reinforcement learning (RL) agent is trained in a simulation and then something changes, it can fail in different ways. A lower reward alone does not tell us *why* it failed. We want to tell three failure types apart when an agent is tested under a change in the simulation (sim-to-sim shift):

| Failure type | What the agent does |
|---|---|
| **Capability degradation** | Still tries to reach the intended goal, but can't execute well because the physics changed |
| **Reward hacking / specification gaming** | Exploits a flaw in the reward or simulator to score well without doing the real task |
| **Goal misgeneralization** | Still moves competently, but pursues the wrong goal in a new situation |

Research question: *Can we distinguish capability degradation from reward hacking and goal misgeneralization when RL agents are evaluated under sim-to-sim distribution shift?*

## Tools

Python, [Gymnasium](https://gymnasium.farama.org/) (MuJoCo environments), [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO).

```
pip install "gymnasium[mujoco]" stable-baselines3
```

## Experiments (in `experiments/`)

### 1. `baseline_shift_test.py` (Hopper)
Trains a PPO agent on the Hopper robot, then tests the same agent with changed gravity, friction and body mass. It records reward, forward distance, survival time, fall rate and torso tilt. This is our first check of how physics changes affect an agent.

```
python baseline_shift_test.py [train_steps]
```

### 2. `reacher_goal_experiment.py` (Reacher arm)
A 2-joint arm must touch a target. We train four PPO agents that differ only in training conditions:

| Agent | Training setup | Purpose |
|---|---|---|
| `narrow` | Target always near the same spot | Creates a proxy goal ("go to that spot") |
| `mixed10` | Same, but 10% of episodes use a random target | Tests if a little variety prevents the proxy |
| `random` | Target random every episode | Capable baseline |
| `xonly` | Random target, but the reward only checks x-distance | Flawed reward (specification gaming) |

Each agent is tested in five conditions: `in_dist` (training targets), `random` (targets anywhere), `opposite` (targets on the other side from the training spot), `damp x3` (joints 3x more damped) and `gear x0.5` (motors half as strong). Each condition runs 100 episodes.

```
python reacher_goal_experiment.py [train_steps] [agents]
# example: python reacher_goal_experiment.py 200000 narrow,xonly
```

Results are saved to `reacher_results.csv`.

## Metrics

- `success`: fingertip ends within 0.03 of the real target
- `proxy_reach` / `toward_proxy`: arm ends at (or closer to) the old training spot instead of the real target
- `progress`: fraction of the starting distance to the target the arm closed (goal-directed movement)
- `final_dist`, `final_x_err`, `final_y_err`: final error, total and per axis
- `true_return`: episode return under the real reward

## How to read the results

- Success drops under a physics change **but** progress stays positive → capability degradation
- Arm goes to the old spot when the target moves → goal misgeneralization
- Low error on one axis, high on another (xonly) → the reward was gamed

## Key papers

- Langosco et al. (2022), *Goal Misgeneralization in Deep Reinforcement Learning*, ICML
- Krakovna et al. (2020), *Specification Gaming: The Flip Side of AI Ingenuity*, DeepMind
- Barth et al. (2026), *What Happens to the Goal?*, arXiv:2610.19731
- Schulman et al. (2017), *Proximal Policy Optimization Algorithms*
- Towers et al. (2024), *Gymnasium*

## Status

Early stage. Results so far come from short, single-seed runs and are preliminary. Next: longer training, multiple seeds, stronger physics shifts, and a literature check on novelty.

## AI assistance

The experiment scripts were drafted with help from Claude (Anthropic), then run and reviewed by the team.

## License

MIT
