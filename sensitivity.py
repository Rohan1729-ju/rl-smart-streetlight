"""Sweep the visibility-penalty weight beta: shows the energy vs. safety trade-off.
Usage: python sensitivity.py"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from env.streetlight_env import StreetlightEnv
from policies import greedy
from train import train
from evaluate import evaluate

betas = [0.5, 1, 2, 3, 5, 10]
pts = []
for b in betas:
    agent, _, _ = train(episodes=4000, seed=1, beta=b, verbose=False)
    r = evaluate(greedy(agent.Q), 200, env=StreetlightEnv(beta=b))
    pts.append((b, r["energy_kwh"], r["ped_unsafe_pct"]))
    print(f"beta={b:5.1f}  energy={r['energy_kwh']:.3f} kWh/day  ped-unsafe={r['ped_unsafe_pct']:.2f}%")
ref = evaluate(greedy(np.load("results/q_table.npy")), 200)
plt.figure(figsize=(4.2, 3.2))
plt.plot([p[1] for p in pts], [p[2] for p in pts], "o-")
for b, e, u in pts: plt.annotate(f"β={b}", (e, u), fontsize=7, xytext=(3, 3), textcoords="offset points")
plt.xlabel("Energy (kWh/day)"); plt.ylabel("Pedestrian steps under-lit (%)")
plt.title("Effect of visibility weight β"); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("results/sensitivity_beta.png", dpi=200)
