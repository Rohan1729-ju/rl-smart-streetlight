# RL-Based Smart Streetlight Control

Tabular Q-learning agent that sets streetlight brightness (0/25/50/75/100 %) every
10 minutes to minimise electricity use while keeping the road adequately lit for
the current ambient light and pedestrian activity. Compared against fixed-brightness,
random and dusk-to-dawn baselines in a custom simulator.

**Course:** AI Assignment 2 – Real-Life Use Cases of RL 
**Team:** < Rohan Kumar Duddeda B21AI031>

## Installation
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt                     # numpy, matplotlib (Python 3.9+)
```

## Reproduce all results
```bash
python train.py --episodes 6000 --seed 0   # trains, writes results/q_table.npy + learning_curve.png
python evaluate.py --days 500              # baselines vs agent on 500 unseen days -> metrics.csv, comparison.png, policy_heatmap.png
python sensitivity.py                      # beta sweep (energy vs safety trade-off) -> sensitivity_beta.png
```
Everything is seeded; training takes about a minute on a laptop CPU. The trained Q-table
(`results/q_table.npy`) is committed, so `evaluate.py` runs without retraining.

## Project layout
| Path | Purpose |
|---|---|
| `env/streetlight_env.py` | Custom simulator (day/night curve, cloud, Poisson pedestrians, reward) |
| `agent/q_agent.py` | Tabular Q-learning with epsilon-greedy exploration |
| `policies.py` | Baseline policies + one-day rollout helper |
| `train.py`, `evaluate.py`, `sensitivity.py` | Training, evaluation, reward-weight sweep |
| `results/` | Q-table, metrics, figures |
| `report/` | LaTeX source of the 2-page report |

## RL formulation (summary)
- **State (360 discrete):** (4-hour block ∈ 6) × (ambient bin ∈ 4) × (pedestrian bin ∈ 3) × (previous brightness ∈ 5)
- **Action (5 discrete):** brightness ∈ {0, 25, 50, 75, 100 %}
- **Reward:** `r = -α·b − β·shortfall·(1 + 0.5·min(n_ped,4)) − γ·|b − b_prev|` with α=1, β=3, γ=0.1
- **Algorithm:** Q-learning, lr 0.1, discount 0.95, ε: 1.0 → 0.02 (×0.999 per episode), 6000 episodes

## Results (500 unseen test days)
See `results/metrics.csv` and `results/eval_output.txt`.
