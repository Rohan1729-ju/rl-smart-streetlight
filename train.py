"""Train the Q-learning agent. Usage: python train.py [--episodes 6000] [--seed 0]"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from env.streetlight_env import StreetlightEnv
from agent.q_agent import QAgent


def train(episodes=6000, seed=0, beta=3.0, alpha=1.0, verbose=True):
    env = StreetlightEnv(alpha=alpha, beta=beta, seed=seed)
    agent = QAgent(env.N_STATES, env.N_ACTIONS, seed=seed)
    rewards, energies = [], []
    for ep in range(episodes):
        s = env.reset()
        eps, ret, en, done = agent.epsilon(ep), 0.0, 0.0, False
        while not done:
            a = agent.act(s, eps)
            s2, r, done, info = env.step(a)
            agent.update(s, a, r, s2, done)
            s, ret, en = s2, ret + r, en + info["energy_kwh"]
        rewards.append(ret); energies.append(en)
        if verbose and (ep + 1) % 500 == 0:
            print(f"episode {ep+1:5d}  eps={eps:.3f}  avg reward(last 500)={np.mean(rewards[-500:]):8.2f}")
    return agent, np.array(rewards), np.array(energies)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs("results", exist_ok=True)
    agent, rewards, energies = train(a.episodes, a.seed)
    agent.save("results/q_table.npy")
    np.save("results/train_rewards.npy", rewards)
    ma = np.convolve(rewards, np.ones(100) / 100, mode="valid")
    plt.figure(figsize=(5, 3.2))
    plt.plot(rewards, alpha=0.25, label="per episode")
    plt.plot(np.arange(99, len(rewards)), ma, label="100-ep moving avg")
    plt.xlabel("Training episode (days)"); plt.ylabel("Episode return")
    plt.title("Q-learning learning curve"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig("results/learning_curve.png", dpi=200)
    print("saved results/q_table.npy and learning_curve.png")
