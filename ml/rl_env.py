"""
Event‑driven RL environment for predictive AMM control (COMPLETE V3 VERSION)
===========================================================================
CHANGES FROM ORIGINAL:
- Action 1 now REPOSITIONS LIQUIDITY (implements paper's Figure 4)
- State space expanded to 6D (added liquidity utilization)
- Integrated Gaussian incentive distribution
- V3 concentrated liquidity support
- Proper tracking of repositioning events

State s_t (6 dimensions):
  • v_t            : current valuation in (0,1)
  • vpred_t        : LSTM prediction of forward valuation v′_p
  • exp_load_t     : Monte‑Carlo expected load E[load | v_t]
  • inv_x, inv_y   : current reserves (normalized)
  • liq_util_t     : liquidity utilization (fraction of L active)

Actions a_t ∈ {0,1}:
  • 0 = do nothing (keep current liquidity distribution)
  • 1 = reposition liquidity to v′_p (Gaussian distribution, paper Figure 4)

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
    """Configuration for RL environment."""
    beta_c: float = 0.001          # reward threshold (paper Eq. 3)
    sigma_noise: float = 0.02      # std for MC sampling of future valuations
    sigma_liq: float = 0.05        # std for Gaussian liquidity distribution (NEW)
    num_positions: int = 5         # number of concentrated positions (NEW)
    samples_per_step: int = 16     # MC samples for expected_load
    lstm_win: int = 50             # length of the LSTM window
    normalize_reserves: bool = True
    seed: int = 0
    track_losses: bool = True
    use_concentrated_liquidity: bool = True  # NEW: Enable V3 mode


# ---------------------------- env -----------------------------------------------

class RLEnv:
    """Event-driven RL environment for predictive AMM.
    
    This implements the paper's training environment where:
    1. Agent observes (v_t, v'_p, E[load], reserves, liq_util)
    2. Agent chooses to reposition liquidity or not
    3. Price moves to v_{t+1}
    4. Agent receives reward based on prediction quality
    """
    
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
            Environment configuration
        events : List[Event]
            Price events from data.event_stream.make_events
        camm : ConfigurableAMM
            Initial AMM state (will be cloned)
        lstm_predict : callable
            Function: window[T, D] -> v'_p (predicted valuation)
        window_init : np.ndarray
            Initial feature window [lstm_win, D]
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

        # Diagnostics
        self.loss_history = [] if cfg.track_losses else None
        self.reposition_history = [] if cfg.track_losses else None
        
        self.reset()

    # --------------- Internal helpers --------------------------------------------
    
    def _clone_camm(self) -> ConfigurableAMM:
        """Deep clone of AMM state including V3 positions."""
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
        exp_load_val: float
    ) -> np.ndarray:
        """Construct 6D state observation vector.
        
        Returns
        -------
        state : np.ndarray[6]
            [v_now, vpred, exp_load, x_norm, y_norm, liq_util]
        """
        # Normalize reserves
        if self.cfg.normalize_reserves:
            total = self.camm.x + self.camm.y + 1e-12
            x_norm = self.camm.x / total
            y_norm = self.camm.y / total
        else:
            x_norm = self.camm.x / (self.camm0.x + 1e-12)
            y_norm = self.camm.y / (self.camm0.y + 1e-12)
        
        # Get liquidity utilization (V3 metric)
        if self.cfg.use_concentrated_liquidity:
            liq_util = self.camm.get_liquidity_utilization(v_now)
        else:
            liq_util = 1.0  # V2 mode: always 100%
        
        return np.array(
            [v_now, vpred, exp_load_val, x_norm, y_norm, liq_util], 
            dtype=np.float32
        )

    def _sample_future_vs(self, vpred: float) -> np.ndarray:
        """Sample future valuations around prediction for MC expected load.
        
        Parameters
        ----------
        vpred : float
            Predicted valuation v'_p
            
        Returns
        -------
        samples : np.ndarray
            Array of sampled future valuations
        """
        s = self.cfg.samples_per_step
        if s <= 0:
            return np.array([])
        
        # Gaussian noise around prediction
        eps = self.rng.normal(0.0, self.cfg.sigma_noise, size=s)
        v_samp = clip01(vpred + eps)
        return v_samp

    # --------------- Gym-like API ------------------------------------------------
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state.
        
        Returns
        -------
        obs : np.ndarray[6]
            Initial observation
        """
        self.idx = 0
        self.events = list(self.base_events)
        self.camm = self._clone_camm()
        self.window = self.window0.copy()
        
        # Get initial event
        v0 = float(self.events[self.idx].v)
        
        # Get LSTM prediction
        vpred = float(self.lstm_predict(self.window))
        
        # Compute expected load
        expL = float(expected_load(v0, self._sample_future_vs(vpred), self.camm.c))
        
        # Initialize concentrated liquidity at current valuation
        if self.cfg.use_concentrated_liquidity:
            self.camm.reposition_to_gaussian(
                mu_v=v0,
                sigma_v=self.cfg.sigma_liq,
                num_positions=self.cfg.num_positions
            )
        
        # Construct initial state
        self._last_obs = self._make_state(v0, vpred, expL)
        self._done = False
        
        # Reset diagnostics
        if self.cfg.track_losses:
            self.loss_history = []
            self.reposition_history = []
        
        return self._last_obs.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one environment step.
        
        Action Space (Paper-compliant):
        - Action 0: do nothing (keep current liquidity distribution)
        - Action 1: reposition liquidity to predicted valuation v'_p
        
        Parameters
        ----------
        action : int
            Action to take {0, 1}
            
        Returns
        -------
        obs : np.ndarray[6]
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
        
        # Get LSTM prediction of next valuation
        vpred = float(self.lstm_predict(self.window))
        
        # Compute expected load at current valuation
        expL = float(expected_load(v_now, self._sample_future_vs(vpred), self.camm.c))
        
        # Compute current loss (used to decide if action needed)
        current_loss = abs(v_now - vpred) + expL
        
        # ========================================================================
        # ACTION EXECUTION (Paper's Core Mechanism)
        # ========================================================================
        repositioned = False
        
        if action == 1 and current_loss > self.cfg.beta_c:
            # ACTION 1: Reposition liquidity to predicted valuation
            # This implements Figure 4 from the paper
            if self.cfg.use_concentrated_liquidity:
                self.camm.reposition_to_gaussian(
                    mu_v=vpred,  # ← Center liquidity at PREDICTED valuation
                    sigma_v=self.cfg.sigma_liq,
                    num_positions=self.cfg.num_positions
                )
                repositioned = True
                
                # Track repositioning for analysis
                if self.reposition_history is not None:
                    self.reposition_history.append({
                        'step': self.idx,
                        'v_now': v_now,
                        'vpred': vpred,
                        'sigma': self.cfg.sigma_liq,
                    })

        # ========================================================================
        # ADVANCE TO NEXT EVENT
        # ========================================================================
        self.idx += 1
        
        # Check for episode termination
        if self.idx >= len(self.events):
            self._done = True
            obs = self._make_state(v_now=v_now, vpred=vpred, exp_load_val=expL)
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
                'repositioned': repositioned,
                'action': action,
            })

        # ========================================================================
        # PSEUDO-ARBITRAGE SHIFT
        # ========================================================================
        # Simulate market moving to v_next (reactive, not predictive)
        dx, dy = self.camm.pseudo_arbitrage_to(v_next, current_v=v_now)

        # ========================================================================
        # COMPOSE NEXT OBSERVATION
        # ========================================================================
        obs = self._make_state(v_now=v_next, vpred=vpred, exp_load_val=expL)
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
            'repositioned': repositioned,
            'liquidity_utilization': (
                self.camm.get_liquidity_utilization(v_next) 
                if self.cfg.use_concentrated_liquidity 
                else 1.0
            ),
        }
        
        return obs.copy(), float(reward), bool(done), info

    # --------------- Diagnostics -------------------------------------------------
    
    def get_loss_stats(self) -> Dict[str, float]:
        """Return statistics on loss components from episode history.
        
        Returns
        -------
        stats : dict
            Dictionary with mean/median/ratio statistics
        """
        if not self.loss_history:
            return {}
        
        pred_slips = np.array([x['pred_slippage'] for x in self.loss_history])
        exp_loads = np.array([x['exp_load'] for x in self.loss_history])
        total_losses = np.array([x['total_loss'] for x in self.loss_history])
        rewards = np.array([x['reward'] for x in self.loss_history])
        repositions = np.array([x.get('repositioned', False) for x in self.loss_history])
        
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
            'reposition_count': int(np.sum(repositions)),
            'reposition_ratio': float(np.mean(repositions)),
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
    print("RL ENVIRONMENT TEST (V3 MODE)")
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
        x = win[:, -1] if win.shape[1] > 0 else win[:, 0]
        return float(1 / (1 + np.exp(-x.mean())))

    # Initial window
    window_init = np.random.randn(50, 4) * 0.1

    # Create environment with V3 mode
    cfg = RLEnvConfig(
        beta_c=0.001, 
        sigma_liq=0.05, 
        num_positions=5,
        use_concentrated_liquidity=True,
        track_losses=True
    )
    env = RLEnv(cfg, events, camm, dummy_pred, window_init)
    
    # Test reset
    obs = env.reset()
    print(f"\nReset:")
    print(f"  State shape: {obs.shape}")
    print(f"  State values: {obs}")
    print(f"  Positions created: {len(env.camm.positions)}")
    
    # Test episode
    print(f"\nRunning test episode (100 steps)...")
    total_r = 0.0
    repositions = 0
    actions = []
    
    for i in range(min(100, len(events) - 1)):
        action = rng.integers(0, 2)
        actions.append(action)
        obs, r, done, info = env.step(action)
        total_r += r
        
        if info['repositioned']:
            repositions += 1
        
        if i < 3:
            print(f"  Step {i}: action={action}, reward={r:+.1f}, "
                  f"repositioned={info['repositioned']}, "
                  f"liq_util={info['liquidity_utilization']:.2%}, "
                  f"loss={info['total_loss']:.6f}")
        
        if done:
            break
    
    print(f"\nEpisode Summary:")
    print(f"  Steps: {i+1}")
    print(f"  Total reward: {total_r:.2f}")
    print(f"  Mean reward: {total_r/(i+1):.4f}")
    print(f"  Repositions: {repositions} ({100*repositions/(i+1):.1f}%)")
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
    print("Environment test completed successfully!")
    print("="*70)