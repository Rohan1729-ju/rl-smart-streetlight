"""Evaluate the trained agent against baselines on identical unseen days.
Usage: python evaluate.py [--days 500]"""
import argparse, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from env.streetlight_env import StreetlightEnv
from policies import fixed, random_policy, dusk_to_dawn, greedy, run_day

TEST_SEED0 = 100_000          # test days never overlap with training randomness


def evaluate(policy, days, env=None):
    env = env or StreetlightEnv()
    rows = [run_day(env, policy, TEST_SEED0 + d) for d in range(days)]
    m = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
    ped = sum(r["ped_steps"] for r in rows)
    return dict(energy_kwh=m["energy"], reward=m["reward"],
                unsafe_pct=100 * m["unsafe"] / StreetlightEnv.STEPS_PER_DAY,
                ped_unsafe_pct=100 * sum(r["ped_unsafe"] for r in rows) / max(ped, 1),
                flicker=m["flicker"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--days", type=int, default=500)
    days = p.parse_args().days
    Q = np.load("results/q_table.npy")
    policies = {"Fixed 100%": fixed(4), "Fixed 50%": fixed(2),
                "Random": random_policy(0), "Dusk-to-dawn": dusk_to_dawn,
                "Q-learning": greedy(Q)}
    res = {n: evaluate(pol, days) for n, pol in policies.items()}

    print(f"{'Policy':14s} {'kWh/day':>8s} {'Reward':>9s} {'Unsafe%':>8s} {'PedUnsafe%':>11s} {'Flicker':>8s}")
    for n, r in res.items():
        print(f"{n:14s} {r['energy_kwh']:8.3f} {r['reward']:9.1f} {r['unsafe_pct']:8.2f} "
              f"{r['ped_unsafe_pct']:11.2f} {r['flicker']:8.1f}")
    with open("results/metrics.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["policy"] + list(next(iter(res.values())).keys()))
        for n, r in res.items(): w.writerow([n] + [f"{v:.4f}" for v in r.values()])

    # ---- bar chart ----
    names = list(res); x = np.arange(len(names))
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    for a, key, t in zip(ax, ["energy_kwh", "ped_unsafe_pct", "reward"],
                         ["Energy (kWh/day)", "Pedestrian steps under-lit (%)", "Avg. daily return"]):
        a.bar(x, [res[n][key] for n in names], color=["#999"] * 4 + ["#1f77b4"])
        a.set_xticks(x); a.set_xticklabels(names, rotation=35, ha="right", fontsize=8); a.set_title(t, fontsize=9)
    plt.tight_layout(); plt.savefig("results/comparison.png", dpi=200)

    # ---- learned policy heatmap (greedy action, previous level = 50%) ----
    fig, ax = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for p_bin, a in enumerate(ax):
        grid = np.full((StreetlightEnv.N_AMBIENT_BINS, StreetlightEnv.N_HOUR_BINS), np.nan)
        for h in range(StreetlightEnv.N_HOUR_BINS):
            for amb in range(StreetlightEnv.N_AMBIENT_BINS):
                s = ((h * 4 + amb) * 3 + p_bin) * 5 + 2
                if np.any(Q[s] != 0):
                    grid[amb, h] = StreetlightEnv.BRIGHTNESS[np.argmax(Q[s])] * 100
        im = a.imshow(grid, origin="lower", vmin=0, vmax=100, cmap="YlOrRd", aspect="auto")
        a.set_xticks(range(6)); a.set_xticklabels([f"{4*i}-{4*i+4}h" for i in range(6)], fontsize=7)
        a.set_title(["No pedestrians", "1-2 pedestrians", "3+ pedestrians"][p_bin], fontsize=9)
        a.set_xlabel("Time of day"); a.set_yticks(range(4))
    ax[0].set_ylabel("Ambient light bin (0=dark)")
    fig.colorbar(im, ax=ax, label="Chosen brightness (%)", fraction=0.02)
    plt.savefig("results/policy_heatmap.png", dpi=200, bbox_inches="tight")
    print("saved metrics.csv, comparison.png, policy_heatmap.png")
