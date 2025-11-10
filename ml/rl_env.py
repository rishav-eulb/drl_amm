"""
Event‑driven RL environment for predictive AMM control (FIXED VERSION)
==============================================================================
FIXES APPLIED:
1. Action space: agent freely chooses to inject ε (no gating on loss)
2. Window updates: properly rolls forward with new market data
3. Feature extraction: maintains feature history for LSTM
4. Epsilon history: properly tracked and used
5. **NEW FIX**: Epsilon augmentation now REPLACES last column instead of adding

State s_t (7 dimensions):
  • v_t            : current valuation in (0,1)
  • vpred_t        : LSTM prediction of forward valuation v′_p
  • exp_load_t     : Monte‑Carlo expected load E[load | v_t]
  • inv_x, inv_y   : current reserves (normalized)
  • liq_util_t     : liquidity utilization (fraction of L active)
  • ε_current      : current Gaussian input parameter

Actions a_t ∈ {0,1}:
  • 0 = do nothing (ε = 0)
  • 1 = inject Gaussian parameter ε ~ N(μ_ε, σ_ε) into LSTM window

Reward (paper Eq. 3):
  ℓ_t = |v_{t+1} − v′_p| + E[load(v_t)]
  r_t = +1  if  ℓ_t < β_c
        -1  if  ℓ_t > β_c
         0  otherwise
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
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
    """Configuration for RL environment (paper-compliant)."""
    beta_c: float = 0.001          # reward threshold (paper Eq. 3)
    sigma_noise: float = 0.02      # std for MC sampling of future valuations
    
    # ε distribution parameters (paper page 12)
    mu_epsilon: float = 0.0        # mean of ε ~ N(μ_ε, σ_ε)
    sigma_epsilon: float = 0.1     # std of ε ~ N(μ_ε, σ_ε)
    
    samples_per_step: int = 16     # MC samples for expected_load
    lstm_win: int = 50             # length of the LSTM window
    normalize_reserves: bool = True
    seed: int = 0
    track_losses: bool = True
    
    # V3 parameters (for comparison baseline)
    use_concentrated_liquidity: bool = False
    sigma_liq: float = 0.05
    num_positions: int = 5


# ---------------------------- env -----------------------------------------------

class RLEnv:
    """Event-driven RL environment for predictive AMM (FIXED VERSION)."""
    
    def __init__(
        self,
        cfg: RLEnvConfig,
        events: List[Event],
        camm: ConfigurableAMM,
        lstm_predict: Callable[[np.ndarray], float],
        window_init: np.ndarray,
        *,
        full_feature_array: Optional[np.ndarray] = None,
        price_to_val: Optional[Callable[[float], float]] = None,
    ) -> None:
        """Create an event‑driven environment.
        
        Parameters
        ----------
        cfg : RLEnvConfig
            Environment configuration
        events : List[Event]
            Price events from data.event_stream.make_events
        camm : ConfigurableAMM
            Initial AMM state (will be cloned)
        lstm_predict : callable
            Function: window[T, D] -> v'_p (predicted valuation)
        window_init : np.ndarray
            Initial feature window [lstm_win, D] (INCLUDES epsilon column as last column)
        full_feature_array : np.ndarray, optional
            Full feature array [T, D] aligned with events for window updates
        """
        self.cfg = cfg
        self.base_events = events
        self.camm0 = camm
        self.lstm_predict = lstm_predict
        self.price_to_val = price_to_val
        self.rng = np.random.default_rng(cfg.seed)

        # Runtime state
        self.window0 = np.array(window_init, dtype=float)
        assert self.window0.ndim == 2 and self.window0.shape[0] == cfg.lstm_win
        
        # Store full feature array for window updates
        self.full_features = full_feature_array
        
        # Diagnostics
        self.loss_history = [] if cfg.track_losses else None
        self.epsilon_injections = [] if cfg.track_losses else None
        
        self.reset()

    # --------------- Internal helpers --------------------------------------------
    
    def _clone_camm(self) -> ConfigurableAMM:
        """Deep clone of AMM state."""
        c = ConfigurableAMM(
            x=float(self.camm0.x), 
            y=float(self.camm0.y), 
            name=self.camm0.name + "_env"
        )
        
        # Copy drift
        c.drift.dx = self.camm0.drift.dx
        c.drift.dy = self.camm0.drift.dy
        
        # Copy incentive parameters
        if self.camm0.incentives.mu_v is not None:
            c.incentives.mu_v = float(self.camm0.incentives.mu_v)
        if self.camm0.incentives.sigma_v is not None:
            c.incentives.sigma_v = float(self.camm0.incentives.sigma_v)
        
        # Copy V3 mode and positions
        c.use_concentrated = self.camm0.use_concentrated
        if self.camm0.positions:
            from amm.camm import ConcentratedPosition
            c.positions = [
                ConcentratedPosition(
                    v_lower=pos.v_lower,
                    v_upper=pos.v_upper,
                    L=pos.L,
                    active=pos.active
                )
                for pos in self.camm0.positions
            ]
        
        return c

    def _make_state(
        self, 
        v_now: float, 
        vpred: float, 
        exp_load_val: float,
        epsilon_current: float
    ) -> np.ndarray:
        """Construct 7D state observation vector.
        
        Returns
        -------
        state : np.ndarray[7]
            [v_now, vpred, exp_load, x_norm, y_norm, liq_util, ε_current]
        """
        # Normalize reserves
        if self.cfg.normalize_reserves:
            total = self.camm.x + self.camm.y + 1e-12
            x_norm = self.camm.x / total
            y_norm = self.camm.y / total
        else:
            x_norm = self.camm.x / (self.camm0.x + 1e-12)
            y_norm = self.camm.y / (self.camm0.y + 1e-12)
        
        # Get liquidity utilization
        if self.cfg.use_concentrated_liquidity:
            liq_util = self.camm.get_liquidity_utilization(v_now)
        else:
            liq_util = 1.0  # V2 mode: always 100%
        
        return np.array(
            [v_now, vpred, exp_load_val, x_norm, y_norm, liq_util, epsilon_current], 
            dtype=np.float32
        )

    def _sample_future_vs(self, vpred: float) -> np.ndarray:
        """Sample future valuations around prediction for MC expected load."""
        s = self.cfg.samples_per_step
        if s <= 0:
            return np.array([])
        
        # Gaussian noise around prediction
        eps = self.rng.normal(0.0, self.cfg.sigma_noise, size=s)
        v_samp = clip01(vpred + eps)
        return v_samp
    
    def _augment_window_with_epsilon(self, window: np.ndarray, epsilon: float) -> np.ndarray:
        """Update ε column in LSTM window (last column).
        
        Paper (Algorithm 3, line 5):
        "Read the market price v'_obs, external signals τ_t, and Gaussian input parameter ε"
        
        The window already has epsilon as the last column (set to 0 during training).
        This method replaces it with the injected value.
        
        **FIX**: Previously this added a new column, causing dimension mismatch.
        Now it replaces the existing epsilon column (last column).
        """
        # Copy window and replace last column (epsilon) with new value
        augmented = window.copy()
        augmented[:, -1] = epsilon
        return augmented
    
    def _extract_features_at_event(self, event_idx: int) -> np.ndarray:
        """Extract feature vector at a given event index.
        
        If full_features array is available, use it. Otherwise, create
        a simple feature vector from the event data.
        """
        if self.full_features is not None:
            # Use the event's time index to get features
            t = self.events[event_idx].t
            if t < len(self.full_features):
                return self.full_features[t].copy()
        
        # Fallback: create minimal features from event data
        event = self.events[event_idx]
        # Simple feature vector: [valuation, delta_v, abs_delta_v, ...]
        return np.array([
            event.v,
            event.dv,
            abs(event.dv),
            0.0,  # placeholder for other features
        ], dtype=float)

    # --------------- Gym-like API ------------------------------------------------
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state.
        
        Returns
        -------
        obs : np.ndarray[7]
            Initial observation
        """
        self.idx = 0
        self.events = list(self.base_events)
        self.camm = self._clone_camm()
        self.window = self.window0.copy()
        self.current_epsilon = 0.0
        
        # Get initial event
        v0 = float(self.events[self.idx].v)
        
        # Get LSTM prediction with ε=0
        augmented_window = self._augment_window_with_epsilon(self.window, 0.0)
        vpred = float(self.lstm_predict(augmented_window))
        
        # Compute expected load
        expL = float(expected_load(v0, self._sample_future_vs(vpred), self.camm.c))
        
        # Construct initial state
        self._last_obs = self._make_state(v0, vpred, expL, 0.0)
        self._done = False
        
        # Reset diagnostics
        if self.cfg.track_losses:
            self.loss_history = []
            self.epsilon_injections = []
        
        return self._last_obs.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one environment step (FIXED VERSION).
        
        FIXES:
        1. Agent freely chooses action (no gating on loss)
        2. Window properly updates with new features
        3. Epsilon injection logic simplified
        4. **NEW**: Epsilon augmentation replaces last column instead of adding
        
        Parameters
        ----------
        action : int
            Action to take {0, 1}
            
        Returns
        -------
        obs : np.ndarray[7]
            Next observation
        reward : float
            Reward {-1, 0, +1}
        done : bool
            Episode termination flag
        info : dict
            Diagnostic information
        """
        if self._done:
            return self._last_obs.copy(), 0.0, True, {"reason": "already_done"}

        # Current state
        v_now = float(self.events[self.idx].v)
        
        # ========================================================================
        # ACTION EXECUTION (FIXED: agent freely chooses)
        # ========================================================================
        epsilon_injected = 0.0
        
        if action == 1:
            # ACTION 1: Inject Gaussian parameter ε
            # Paper: "Insert input parameter ε_{t+k}"
            epsilon_injected = float(self.rng.normal(
                self.cfg.mu_epsilon, 
                self.cfg.sigma_epsilon
            ))
            self.current_epsilon = epsilon_injected
            
            # Track injection for analysis
            if self.epsilon_injections is not None:
                self.epsilon_injections.append({
                    'step': self.idx,
                    'epsilon': epsilon_injected,
                    'v_now': v_now,
                })
        else:
            # ACTION 0: Do nothing, reset epsilon
            self.current_epsilon = 0.0
        
        # Get LSTM prediction with current ε
        augmented_window = self._augment_window_with_epsilon(self.window, self.current_epsilon)
        vpred = float(self.lstm_predict(augmented_window))
        
        # Compute expected load at current valuation
        expL = float(expected_load(v_now, self._sample_future_vs(vpred), self.camm.c))

        # ========================================================================
        # ADVANCE TO NEXT EVENT
        # ========================================================================
        self.idx += 1
        
        # Check for episode termination
        if self.idx >= len(self.events):
            self._done = True
            obs = self._make_state(v_now=v_now, vpred=vpred, exp_load_val=expL, 
                                   epsilon_current=self.current_epsilon)
            return obs, 0.0, True, {"reason": "end_of_episode"}

        # Get next valuation (actual future state)
        v_next = float(self.events[self.idx].v)

        # ========================================================================
        # REWARD CALCULATION (Paper Eq. 3)
        # ========================================================================
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

        # ========================================================================
        # TRACK DIAGNOSTICS
        # ========================================================================
        if self.loss_history is not None:
            self.loss_history.append({
                'pred_slippage': pred_slippage,
                'exp_load': expL,
                'total_loss': loss,
                'v_now': v_now,
                'v_next': v_next,
                'vpred': vpred,
                'reward': reward,
                'epsilon': epsilon_injected,
                'action': action,
            })

        # ========================================================================
        # UPDATE WINDOW (FIXED: actually updates!)
        # ========================================================================
        # Roll window forward and add new features at the end
        new_features = self._extract_features_at_event(self.idx)
        
        # Ensure feature dimensions match
        if new_features.shape[0] != self.window.shape[1]:
            # Pad or truncate to match
            if new_features.shape[0] < self.window.shape[1]:
                pad_size = self.window.shape[1] - new_features.shape[0]
                new_features = np.pad(new_features, (0, pad_size), mode='constant')
            else:
                new_features = new_features[:self.window.shape[1]]
        
        # Roll and update
        self.window = np.roll(self.window, -1, axis=0)
        self.window[-1] = new_features

        # ========================================================================
        # PSEUDO-ARBITRAGE SHIFT
        # ========================================================================
        # Simulate market moving to v_next (reactive, not predictive)
        dx, dy = self.camm.pseudo_arbitrage_to(v_next, current_v=v_now)
        
        # ========================================================================
        # COMPOSE NEXT OBSERVATION
        # ========================================================================
        obs = self._make_state(
            v_now=v_next, 
            vpred=vpred, 
            exp_load_val=expL,
            epsilon_current=self.current_epsilon
        )
        self._last_obs = obs

        # Check if done
        done = self.idx >= len(self.events) - 1
        self._done = done
        
        # Build info dict
        info = {
            'v_now': v_now,
            'v_next': v_next,
            'vpred': vpred,
            'pred_slippage': pred_slippage,
            'exp_load': expL,
            'total_loss': loss,
            'drift': (dx, dy),
            'action': action,
            'epsilon': epsilon_injected,
            'liquidity_utilization': (
                self.camm.get_liquidity_utilization(v_next) 
                if self.cfg.use_concentrated_liquidity 
                else 1.0
            ),
        }
        
        return obs.copy(), float(reward), bool(done), info

    # --------------- Diagnostics -------------------------------------------------
    
    def get_loss_stats(self) -> Dict[str, float]:
        """Return statistics on loss components from episode history."""
        if not self.loss_history:
            return {}
        
        pred_slips = np.array([x['pred_slippage'] for x in self.loss_history])
        exp_loads = np.array([x['exp_load'] for x in self.loss_history])
        total_losses = np.array([x['total_loss'] for x in self.loss_history])
        rewards = np.array([x['reward'] for x in self.loss_history])
        epsilons = np.array([x.get('epsilon', 0.0) for x in self.loss_history])
        
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
            'epsilon_injection_count': int(np.sum(epsilons != 0)),
            'epsilon_mean_when_injected': float(np.mean(epsilons[epsilons != 0])) if np.any(epsilons != 0) else 0.0,
        }

    @property
    def camm(self) -> ConfigurableAMM:
        return self._camm

    @camm.setter
    def camm(self, val: ConfigurableAMM) -> None:
        self._camm = val


# ---------------------------- Self-test ------------------------------------------
if __name__ == "__main__":
    print("="*70)
    print("RL ENVIRONMENT TEST (FIXED VERSION)")
    print("="*70)
    
    from amm.camm import ConfigurableAMM
    from data.event_stream import make_events, normalize_price_to_valuation
    import numpy as np

    # Generate synthetic data
    rng = np.random.default_rng(0)
    price = np.cumprod(1.0 + 0.002 * rng.standard_normal(2000))
    v = normalize_price_to_valuation(price)
    events = make_events(v, beta_v=0.01)

    print(f"\nGenerated {len(events)} events from {len(price)} prices")

    # Initialize AMM
    camm = ConfigurableAMM(x=100.0, y=100.0)

    # Dummy LSTM predictor
    def dummy_pred(win: np.ndarray) -> float:
        # Win has shape [T, D] where last column is ε
        x = win[:, 0] if win.shape[1] > 1 else win[:, 0]
        epsilon_effect = win[:, -1].mean() if win.shape[1] > 1 else 0.0
        base_pred = 1 / (1 + np.exp(-x.mean()))
        # ε shifts prediction slightly
        return float(np.clip(base_pred + 0.1 * epsilon_effect, 0.01, 0.99))

    # Initial window (D=4 features + 1 epsilon = 5 total)
    window_init = np.random.randn(50, 5) * 0.1

    # Create environment
    cfg = RLEnvConfig(
        beta_c=0.001,
        mu_epsilon=0.0,
        sigma_epsilon=0.1,
        track_losses=True
    )
    env = RLEnv(cfg, events, camm, dummy_pred, window_init)
    
    # Test reset
    obs = env.reset()
    print(f"\nReset:")
    print(f"  State shape: {obs.shape} (should be 7D)")
    print(f"  State values: {obs}")
    print(f"  State labels: [v, vpred, expL, x_norm, y_norm, liq_util, ε]")
    
    # Test episode
    print(f"\nRunning test episode (100 steps)...")
    total_r = 0.0
    epsilon_injections = 0
    actions = []
    
    for i in range(min(100, len(events) - 1)):
        action = rng.integers(0, 2)
        actions.append(action)
        obs, r, done, info = env.step(action)
        total_r += r
        
        if info['epsilon'] != 0.0:
            epsilon_injections += 1
        
        if i < 3:
            print(f"  Step {i}: action={action}, reward={r:+.1f}, "
                  f"ε={info['epsilon']:+.4f}, "
                  f"loss={info['total_loss']:.6f}")
        
        if done:
            break
    
    print(f"\nEpisode Summary:")
    print(f"  Steps: {i+1}")
    print(f"  Total reward: {total_r:.2f}")
    print(f"  Mean reward: {total_r/(i+1):.4f}")
    print(f"  ε injections: {epsilon_injections} ({100*epsilon_injections/(i+1):.1f}%)")
    print(f"  Actions: 0={actions.count(0)}, 1={actions.count(1)}")
    
    # Get statistics
    stats = env.get_loss_stats()
    print(f"\nLoss Statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    
    print("\n" + "="*70)
    print("✓ FIXED environment test completed!")
    print("✓ Action 1 now freely chosen by agent (no gating)")
    print("✓ Window properly updates with new features")
    print("✓ State is 7D including current ε value")
    print("✓ Epsilon dimension fix: replaces last column instead of adding")
    print("="*70)