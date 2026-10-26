"""Baseline lighting policies, plus a helper to roll out any policy for one day."""
import numpy as np
from env.streetlight_env import StreetlightEnv


def fixed(level):
    return lambda env, s: level

def random_policy(seed=0):
    rng = np.random.default_rng(seed)
    return lambda env, s: int(rng.integers(StreetlightEnv.N_ACTIONS))

def dusk_to_dawn(env, s):
    """Classic timer/photocell rule: 100 % when dark, off when bright."""
    _, amb_bin, _, _ = StreetlightEnv.decode(s)
    return 4 if amb_bin <= 1 else 0

def greedy(Q):
    return lambda env, s: int(np.argmax(Q[s]))


def run_day(env, policy, seed):
    """Roll out one full day with a fixed weather/pedestrian seed."""
    s = env.reset(seed=seed)
    tot = dict(reward=0.0, energy=0.0, unsafe=0, ped_steps=0, ped_unsafe=0, flicker=0.0)
    done = False
    while not done:
        a = policy(env, s)
        s, r, done, info = env.step(a)
        tot["reward"] += r
        tot["energy"] += info["energy_kwh"]
        tot["flicker"] += info["flicker"]
        bad = info["shortfall"] > 0.2
        tot["unsafe"] += bad
        if info["ped"] > 0:
            tot["ped_steps"] += 1
            tot["ped_unsafe"] += bad
    return tot
