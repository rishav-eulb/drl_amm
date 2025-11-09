"""
Event‑driven RL environment for predictive AMM control
-----------------------------------------------------
This module provides a lightweight, dependency‑free environment that mirrors
the paper's RL loop:

State s_t:
  • v_t            : current valuation in (0,1)
  • vpred_t        : LSTM prediction of forward valuation v′_p
  • exp_load_t     : Monte‑Carlo expected load E[load | v_t]
  • inv_x, inv_y   : (optional) current reserves (normalized)

Actions a_t ∈ {0,1}:
  • 0 = do nothing
  • 1 = inject Gaussian noise ε into the LSTM feature window (exploration)

Transition:
  • The environment advances to the next *event* (price‑based) index.
  • We simulate a pseudo‑arbitrage shift of the cAMM to v_{t+1} (the true next
    valuation), track the inventory drift, and compute the reward.

Reward (paper‑inspired):
  r_t = +1  if  |v_{t+1} − v′_p| + E[load(v_t)]  < β_c
        −1  if  |v_{t+1} − v′_p| + E[load(v_t)]  > β_c
         0  otherwise

Notes
-----
• This is a *training* environment; it abstracts away trade microstructure.
• Expected load is computed via amm.math.expected_load. You supply the set of
  samples for v′ (e.g., bootstrap from recent deltas or a Gaussian around vpred).
• LSTM predictor is any callable: (window[T,D]) -> scalar in (0,1).
• Feature window manager is pluggable; by default we keep a rolling buffer.

Public API
----------
- RLEnvConfig
- RLEnv

Example
-------
from ml.rl_env import RLEnv, RLEnvConfig
from amm.camm import ConfigurableAMM

cfg = RLEnvConfig(beta_c=0.02, sigma_noise=0.02)
env = RLEnv(cfg, events, camm, lstm_predict_fn, window_init)
obs = env.reset()
obs, r, done, info = env.step(1)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np

from amm.math import expected_load, clip01
from amm.camm import ConfigurableAMM
from data.event_stream import Event

# ---------------------------- config --------------------------------------------

@dataclass
class RLEnvConfig:
    beta_c: float = 0.02           # reward threshold
    sigma_noise: float = 0.02      # std for feature noise injection
    samples_per_step: int = 16     # MC samples for expected_load
    lstm_win: int = 50             # length of the LSTM window
    normalize_reserves: bool = True
    seed: int = 0

# ---------------------------- env -----------------------------------------------

class RLEnv:
    def __init__(
        self,
        cfg: RLEnvConfig,
        events: List[Event],
        camm: ConfigurableAMM,
        lstm_predict: Callable[[np.ndarray], float],
        window_init: np.ndarray,
        *,
        price_to_val: Optional[Callable[[float], float]] = None,
    ) -> None:
        """Create an event‑driven environment.

        Parameters
        ----------
        cfg : RLEnvConfig
        events : list of Event(t, v, dv) produced by data.event_stream.make_events
        camm : ConfigurableAMM, used to simulate pseudo‑arbitrage shifts
        lstm_predict : callable that maps current window [T,D] -> v′_p in (0,1)
        window_init : initial feature window [T,D]
        price_to_val : optional mapping for price→valuation (unused if events carry v)
        """
        self.cfg = cfg
        self.base_events = events
        self.camm0 = camm  # reference, we will clone internal state manually when resetting
        self.lstm_predict = lstm_predict
        self.price_to_val = price_to_val
        self.rng = np.random.default_rng(cfg.seed)

        # runtime
        self.window0 = np.array(window_init, dtype=float)
        assert self.window0.ndim == 2 and self.window0.shape[0] == cfg.lstm_win

        self.reset()

    # --------------- helpers -----------------------------------------------------
    def _clone_camm(self) -> ConfigurableAMM:
        # shallow clone of reserves/drift/incentives
        c = ConfigurableAMM(x=float(self.camm0.x), y=float(self.camm0.y), name=self.camm0.name + "_roll")
        c.drift.dx = self.camm0.drift.dx
        c.drift.dy = self.camm0.drift.dy
        if self.camm0.incentives.mu_v is not None:
            c.incentives.mu_v = float(self.camm0.incentives.mu_v)
        if self.camm0.incentives.sigma_v is not None:
            c.incentives.sigma_v = float(self.camm0.incentives.sigma_v)
        return c

    def _make_state(self, v_now: float, vpred: float, exp_load_val: float) -> np.ndarray:
        if self.cfg.normalize_reserves:
            x = self.camm.x / (self.camm.x + self.camm.y)
            y = self.camm.y / (self.camm.x + self.camm.y)
        else:
            x = self.camm.x; y = self.camm.y
        return np.array([v_now, vpred, exp_load_val, x, y], dtype=float)

    def _sample_future_vs(self, vpred: float) -> np.ndarray:
        # sample around vpred with small Gaussian perturbations and clipping to (0,1)
        s = self.cfg.samples_per_step
        if s <= 0:
            return np.array([])
        eps = self.rng.normal(0.0, self.cfg.sigma_noise, size=s)
        v_samp = clip01(vpred + eps)
        return v_samp

    # --------------- gym‑like API ------------------------------------------------
    def reset(self) -> np.ndarray:
        self.idx = 0
        self.events = list(self.base_events)
        self.camm = self._clone_camm()
        self.window = self.window0.copy()
        v0 = float(self.events[self.idx].v)
        vpred = float(self.lstm_predict(self.window))
        expL = float(expected_load(v0, self._sample_future_vs(vpred), self.camm.c))
        self._last_obs = self._make_state(v0, vpred, expL)
        self._done = False
        return self._last_obs.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self._done:
            return self._last_obs.copy(), 0.0, True, {}

        # 0=hold, 1=inject noise into features
        if action == 1:
            noise = self.rng.normal(0.0, self.cfg.sigma_noise, size=self.window.shape)
            self.window = self.window + noise

        # predict forward valuation
        vpred = float(self.lstm_predict(self.window))

        # advance to next event (true next valuation)
        self.idx += 1
        if self.idx >= len(self.events):
            self._done = True
            return self._last_obs.copy(), 0.0, True, {"reason": "end_of_episode"}

        v_now = float(self.events[self.idx - 1].v)
        v_next = float(self.events[self.idx].v)

        # expected load under v_now using samples around prediction
        expL = float(expected_load(v_now, self._sample_future_vs(vpred), self.camm.c))

        # reward
        score = abs(v_next - vpred) + expL
        if score < self.cfg.beta_c:
            reward = 1.0
        elif score > self.cfg.beta_c:
            reward = -1.0
        else:
            reward = 0.0

        # simulate virtual shift to v_next, track drift
        dx, dy = self.camm.pseudo_arbitrage_to(v_next, current_v=v_now)

        # compose next observation
        obs = self._make_state(v_now=v_next, vpred=vpred, exp_load_val=expL)
        self._last_obs = obs

        done = self.idx >= len(self.events) - 1
        self._done = done
        info = {"v_now": v_now, "v_next": v_next, "vpred": vpred, "exp_load": expL, "drift": (dx, dy)}
        return obs.copy(), float(reward), bool(done), info

    # convenience
    @property
    def camm(self) -> ConfigurableAMM:
        return self._camm

    @camm.setter
    def camm(self, val: ConfigurableAMM) -> None:
        self._camm = val

# ---------------------------- demo ----------------------------------------------
if __name__ == "__main__":
    # Tiny smoke test with fake objects
    from amm.camm import ConfigurableAMM
    from data.event_stream import make_events, normalize_price_to_valuation
    import numpy as np

    # synthetic price path
    rng = np.random.default_rng(0)
    price = np.cumprod(1.0 + 0.002 * rng.standard_normal(2000))
    v = normalize_price_to_valuation(price)
    events = make_events(v, beta_v=0.01)

    camm = ConfigurableAMM(x=100.0, y=100.0)

    # dummy predictor that returns EMA of last column in window
    def dummy_pred(win: np.ndarray) -> float:
        x = win[:, -1]
        return float(1 / (1 + np.exp(-x.mean())))  # squash to (0,1)

    window_init = np.zeros((50, 4))

    env = RLEnv(RLEnvConfig(beta_c=0.02, sigma_noise=0.02), events, camm, dummy_pred, window_init)
    obs = env.reset()
    total_r = 0.0
    for _ in range(100):
        obs, r, done, info = env.step(action=1)
        total_r += r
        if done:
            break
    print("steps:", _, "total_r:", total_r, "last info keys:", list(info.keys()))
