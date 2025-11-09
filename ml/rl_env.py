"""
Event‑driven RL environment for predictive AMM control (FIXED)
--------------------------------------------------------------
Fixed clip01 array handling bug.

State s_t:
  • v_t            : current valuation in (0,1)
  • vpred_t        : LSTM prediction of forward valuation v′_p
  • exp_load_t     : Monte‑Carlo expected load E[load | v_t]
  • inv_x, inv_y   : current reserves (normalized)

Actions a_t ∈ {0,1}:
  • 0 = do nothing
  • 1 = inject Gaussian noise ε (conditional on loss exceeding β_c)

Reward (paper Eq. 3):
  ℓ_t = |v_{t+1} − v′_p| + E[load(v_t)]
  r_t = +1  if  ℓ_t < β_c
        -1  if  ℓ_t > β_c
         0  otherwise
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np

from amm.maths import expected_load
from amm.camm import ConfigurableAMM
from data.event_stream import Event

# ---------------------------- helper --------------------------------------------

def clip01(v: float | np.ndarray, eps: float = 1e-9) -> float | np.ndarray:
    """Clip scalar or array into (0,1) range."""
    if isinstance(v, np.ndarray):
        return np.clip(v, eps, 1.0 - eps)
    else:
        return min(1.0 - eps, max(eps, float(v)))


# ---------------------------- config --------------------------------------------

@dataclass
class RLEnvConfig:
    beta_c: float = 0.001          # reward threshold
    sigma_noise: float = 0.02      # std for feature noise injection
    samples_per_step: int = 16     # MC samples for expected_load
    lstm_win: int = 50             # length of the LSTM window
    normalize_reserves: bool = True
    seed: int = 0
    track_losses: bool = True      # track loss components for analysis


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
        self.camm0 = camm
        self.lstm_predict = lstm_predict
        self.price_to_val = price_to_val
        self.rng = np.random.default_rng(cfg.seed)

        # runtime
        self.window0 = np.array(window_init, dtype=float)
        assert self.window0.ndim == 2 and self.window0.shape[0] == cfg.lstm_win

        # diagnostics
        self.loss_history = [] if cfg.track_losses else None
        
        self.reset()

    # --------------- helpers -----------------------------------------------------
    def _clone_camm(self) -> ConfigurableAMM:
        """Shallow clone of AMM state."""
        c = ConfigurableAMM(x=float(self.camm0.x), y=float(self.camm0.y), name=self.camm0.name + "_roll")
        c.drift.dx = self.camm0.drift.dx
        c.drift.dy = self.camm0.drift.dy
        if self.camm0.incentives.mu_v is not None:
            c.incentives.mu_v = float(self.camm0.incentives.mu_v)
        if self.camm0.incentives.sigma_v is not None:
            c.incentives.sigma_v = float(self.camm0.incentives.sigma_v)
        return c

    def _make_state(self, v_now: float, vpred: float, exp_load_val: float) -> np.ndarray:
        """Construct state observation vector."""
        if self.cfg.normalize_reserves:
            total = self.camm.x + self.camm.y + 1e-12
            x = self.camm.x / total
            y = self.camm.y / total
        else:
            # Normalize to initial scale
            x = self.camm.x / (self.camm0.x + 1e-12)
            y = self.camm.y / (self.camm0.y + 1e-12)
        return np.array([v_now, vpred, exp_load_val, x, y], dtype=np.float32)

    def _sample_future_vs(self, vpred: float) -> np.ndarray:
        """Sample future valuations around prediction for load estimation."""
        s = self.cfg.samples_per_step
        if s <= 0:
            return np.array([])
        
        # Generate Gaussian noise
        eps = self.rng.normal(0.0, self.cfg.sigma_noise, size=s)
        
        # Add to prediction and clip to (0,1)
        v_samp = clip01(vpred + eps)  # Now handles array correctly
        return v_samp

    # --------------- gym‑like API ------------------------------------------------
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.idx = 0
        self.events = list(self.base_events)
        self.camm = self._clone_camm()
        self.window = self.window0.copy()
        
        v0 = float(self.events[self.idx].v)
        vpred = float(self.lstm_predict(self.window))
        expL = float(expected_load(v0, self._sample_future_vs(vpred), self.camm.c))
        
        self._last_obs = self._make_state(v0, vpred, expL)
        self._done = False
        
        # Reset diagnostics
        if self.cfg.track_losses:
            self.loss_history = []
        
        return self._last_obs.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one environment step.
        
        Paper-compliant action space:
        - Action 0: do nothing
        - Action 1: inject noise IF loss exceeds threshold
        
        Returns
        -------
        obs, reward, done, info
        """
        if self._done:
            return self._last_obs.copy(), 0.0, True, {}

        # Get current valuation
        v_now = float(self.events[self.idx].v)
        
        # Predict forward valuation with current window
        vpred = float(self.lstm_predict(self.window))
        
        # Compute expected load under current valuation
        expL = float(expected_load(v_now, self._sample_future_vs(vpred), self.camm.c))
        
        # PAPER-COMPLIANT ACTION SPACE
        # Calculate current loss to decide whether to inject noise
        current_loss = abs(v_now - vpred) + expL
        
        if action == 1 and current_loss > self.cfg.beta_c:
            # Inject Gaussian noise clipped to (-1, 1) as per paper
            noise = self.rng.normal(0.0, self.cfg.sigma_noise, size=self.window.shape)
            noise = np.clip(noise, -1.0, 1.0)
            self.window = self.window + noise
            
            # Re-predict with noisy window
            vpred = float(self.lstm_predict(self.window))
            expL = float(expected_load(v_now, self._sample_future_vs(vpred), self.camm.c))

        # Advance to next event
        self.idx += 1
        if self.idx >= len(self.events):
            self._done = True
            # Return terminal state
            obs = self._make_state(v_now=v_now, vpred=vpred, exp_load_val=expL)
            return obs, 0.0, True, {"reason": "end_of_episode"}

        v_next = float(self.events[self.idx].v)

        # REWARD CALCULATION (Paper Eq. 3)
        # ℓ_t = |v_{t+1} − v′_p| + E[load(v_t)]
        pred_slippage = abs(v_next - vpred)
        loss = pred_slippage + expL
        
        # r_t = +1 if ℓ < β_c, -1 if ℓ > β_c, 0 if ℓ = β_c
        if loss < self.cfg.beta_c:
            reward = 1.0
        elif loss > self.cfg.beta_c:
            reward = -1.0
        else:
            reward = 0.0

        # Track loss components for diagnostics
        if self.cfg.track_losses:
            self.loss_history.append({
                'pred_slippage': pred_slippage,
                'exp_load': expL,
                'total_loss': loss,
                'v_now': v_now,
                'v_next': v_next,
                'vpred': vpred,
                'reward': reward,
            })

        # Simulate pseudo-arbitrage shift to v_next
        dx, dy = self.camm.pseudo_arbitrage_to(v_next, current_v=v_now)

        # Compose next observation
        obs = self._make_state(v_now=v_next, vpred=vpred, exp_load_val=expL)
        self._last_obs = obs

        done = self.idx >= len(self.events) - 1
        self._done = done
        
        info = {
            'v_now': v_now,
            'v_next': v_next,
            'vpred': vpred,
            'pred_slippage': pred_slippage,
            'exp_load': expL,
            'total_loss': loss,
            'drift': (dx, dy),
            'action': action,
            'noise_injected': (action == 1 and current_loss > self.cfg.beta_c),
        }
        
        return obs.copy(), float(reward), bool(done), info

    # --------------- diagnostics -------------------------------------------------
    def get_loss_stats(self) -> Dict[str, float]:
        """Return statistics on loss components from episode history."""
        if not self.loss_history:
            return {}
        
        pred_slips = np.array([x['pred_slippage'] for x in self.loss_history])
        exp_loads = np.array([x['exp_load'] for x in self.loss_history])
        total_losses = np.array([x['total_loss'] for x in self.loss_history])
        rewards = np.array([x['reward'] for x in self.loss_history])
        
        return {
            'pred_slip_mean': float(np.mean(pred_slips)),
            'pred_slip_median': float(np.median(pred_slips)),
            'exp_load_mean': float(np.mean(exp_loads)),
            'exp_load_median': float(np.median(exp_loads)),
            'total_loss_mean': float(np.mean(total_losses)),
            'total_loss_median': float(np.median(total_losses)),
            'below_threshold_ratio': float(np.mean(total_losses < self.cfg.beta_c)),
            'reward_positive_ratio': float(np.mean(rewards > 0)),
            'reward_mean': float(np.mean(rewards)),
        }

    # convenience
    @property
    def camm(self) -> ConfigurableAMM:
        return self._camm

    @camm.setter
    def camm(self, val: ConfigurableAMM) -> None:
        self._camm = val


# ---------------------------- demo ----------------------------------------------
if __name__ == "__main__":
    # Smoke test with fixed environment
    from amm.camm import ConfigurableAMM
    from data.event_stream import make_events, normalize_price_to_valuation
    import numpy as np

    # Synthetic price path
    rng = np.random.default_rng(0)
    price = np.cumprod(1.0 + 0.002 * rng.standard_normal(2000))
    v = normalize_price_to_valuation(price)
    events = make_events(v, beta_v=0.01)

    camm = ConfigurableAMM(x=100.0, y=100.0)

    # Dummy predictor
    def dummy_pred(win: np.ndarray) -> float:
        x = win[:, -1] if win.shape[1] > 0 else win[:, 0]
        return float(1 / (1 + np.exp(-x.mean())))

    window_init = np.random.randn(50, 4) * 0.1

    # Test with fixed config
    cfg = RLEnvConfig(beta_c=0.001, sigma_noise=0.02, track_losses=True)
    env = RLEnv(cfg, events, camm, dummy_pred, window_init)
    
    obs = env.reset()
    total_r = 0.0
    actions = []
    
    print("Testing fixed RL environment...")
    for i in range(min(100, len(events) - 1)):
        action = rng.integers(0, 2)
        actions.append(action)
        obs, r, done, info = env.step(action)
        total_r += r
        
        if i < 5:
            print(f"Step {i}: action={action}, reward={r:.1f}, "
                  f"loss={info['total_loss']:.6f}, noise={info['noise_injected']}")
        
        if done:
            break
    
    print(f"\nCompleted {i+1} steps")
    print(f"Total reward: {total_r:.2f}")
    print(f"Mean reward: {total_r/(i+1):.4f}")
    print(f"Actions: 0={actions.count(0)}, 1={actions.count(1)}")
    
    # Print loss statistics
    stats = env.get_loss_stats()
    print("\nLoss Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v:.6f}")