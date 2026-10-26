"""Custom simulated streetlight environment (no external RL libraries).

One episode = one 24-hour day, one step = 10 minutes (144 steps).
The agent picks a lamp brightness level; the reward trades off electricity
use against how well the road is lit for the current pedestrian situation.
"""
import numpy as np


class StreetlightEnv:
    BRIGHTNESS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])  # action -> lamp dimming level
    N_ACTIONS = len(BRIGHTNESS)
    STEPS_PER_DAY = 144          # 24 h * 6 steps/h
    STEP_HOURS = 1.0 / 6.0
    LAMP_LUX = 60.0              # light output at 100 % (simulation units)
    LAMP_KW = 0.1                # 100 W lamp at full brightness

    # discretisation of the observation
    N_HOUR_BINS = 6              # 4-hour blocks
    AMBIENT_EDGES = [10.0, 30.0, 60.0]   # -> 4 ambient bins
    N_AMBIENT_BINS = 4
    N_PED_BINS = 3               # 0 / 1-2 / 3+ pedestrians
    N_STATES = N_HOUR_BINS * N_AMBIENT_BINS * N_PED_BINS * N_ACTIONS  # 360

    def __init__(self, alpha=1.0, beta=3.0, flicker=0.1, seed=None):
        self.alpha, self.beta, self.flicker = alpha, beta, flicker
        self.rng = np.random.default_rng(seed)

    # ---------- day generation ----------
    @staticmethod
    def ped_rate(hour):
        """Expected pedestrians per 10-min step: morning + evening peaks."""
        g = lambda mu, sd: np.exp(-0.5 * ((hour - mu) / sd) ** 2)
        return 0.05 + 1.5 * g(8.0, 1.0) + 2.0 * g(19.5, 1.5) + 0.3 * g(13.0, 2.0)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        hours = np.arange(self.STEPS_PER_DAY) * self.STEP_HOURS
        cloud = self.rng.uniform(0.4, 1.0)                      # today's cloudiness
        sun = np.clip(np.sin(np.pi * (hours - 6.0) / 12.0), 0, None)
        sun[(hours < 6) | (hours > 18)] = 0.0
        self.ambient = np.clip(100.0 * cloud * sun +
                               self.rng.normal(0, 2.0, hours.shape), 0, None)
        self.peds = self.rng.poisson(self.ped_rate(hours))
        self.hours = hours
        self.t = 0
        self.prev_a = 0
        return self._state()

    # ---------- observation ----------
    def _state(self, t=None):
        t = self.t if t is None else t
        t = min(t, self.STEPS_PER_DAY - 1)
        h = min(int(self.hours[t] // 4), self.N_HOUR_BINS - 1)
        a = int(np.searchsorted(self.AMBIENT_EDGES, self.ambient[t], side="right"))
        p = 0 if self.peds[t] == 0 else (1 if self.peds[t] <= 2 else 2)
        return ((h * self.N_AMBIENT_BINS + a) * self.N_PED_BINS + p) * self.N_ACTIONS + self.prev_a

    @staticmethod
    def decode(s):
        s, prev = divmod(s, StreetlightEnv.N_ACTIONS)
        s, p = divmod(s, StreetlightEnv.N_PED_BINS)
        h, a = divmod(s, StreetlightEnv.N_AMBIENT_BINS)
        return h, a, p, prev

    # ---------- dynamics ----------
    def step(self, action):
        b = self.BRIGHTNESS[action]
        amb, ped = self.ambient[self.t], self.peds[self.t]
        required = 20.0 if ped == 0 else 50.0                    # lux needed on the road
        shortfall = max(0.0, required - (amb + b * self.LAMP_LUX)) / required
        risk_w = 1.0 + 0.5 * min(ped, 4)                         # more people -> darkness hurts more
        change = abs(b - self.BRIGHTNESS[self.prev_a])
        reward = -self.alpha * b - self.beta * shortfall * risk_w - self.flicker * change
        info = dict(energy_kwh=b * self.LAMP_KW * self.STEP_HOURS, shortfall=shortfall,
                    ped=int(ped), flicker=change)
        self.prev_a = action
        self.t += 1
        done = self.t >= self.STEPS_PER_DAY
        return self._state(self.t), reward, done, info
