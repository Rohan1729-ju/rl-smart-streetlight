"""Tabular Q-learning agent with epsilon-greedy exploration."""
import numpy as np


class QAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.95,
                 eps_start=1.0, eps_end=0.02, eps_decay=0.999, seed=0):
        self.Q = np.zeros((n_states, n_actions))   # rewards are <= 0, so 0 is optimistic
        self.lr, self.gamma = lr, gamma
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions

    def epsilon(self, episode):
        return max(self.eps_end, self.eps_start * self.eps_decay ** episode)

    def act(self, s, eps=0.0):
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        q = self.Q[s]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))   # random tie-break

    def update(self, s, a, r, s2, done):
        target = r if done else r + self.gamma * self.Q[s2].max()
        self.Q[s, a] += self.lr * (target - self.Q[s, a])

    def save(self, path):
        np.save(path, self.Q)

    def load(self, path):
        self.Q = np.load(path)
