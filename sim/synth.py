"""
Synthetic market generator for simulations
-----------------------------------------
This module creates realistic(ish) price/volume streams to test the
predictive AMM stack without relying on exchange data. It supports:

• Geometric Brownian Motion (GBM)
• Regime‑switching volatility
• Optional jump‑diffusion shocks
• Volume models tied to volatility and trend
• Derived valuation series v = price / (1 + price)
• Simple liquidity‑taker trade stream for utilization/depth tests

Public API
----------
make_synth_series(T, *, seed=0, dt=1/1440, s0=100.0,
                  mu_annual=0.0, sigma_annual=0.7,
                  regimes=None, jump_prob=0.0, jump_mu=0.0, jump_sigma=0.04,
                  vol_of_vol=0.0) -> dict

make_trade_stream(price, *, seed=0, trades_per_hour=(40, 120),
                  mean_notional=5_000.0,
                  skew_with_trend=True) -> list[Trade]

Notes
-----
• dt default ~ 1 minute (1/1440 of a day) so "T" is number of minutes.
• All randomness is via numpy Generator for reproducibility.
• The valuation mapping matches event_stream.normalize_price_to_valuation.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional
import numpy as np

from data.event_stream import normalize_price_to_valuation

# ---------------------------- helpers -------------------------------------------

@dataclass
class Trade:
    t: int           # index in the price array
    side: int        # +1 = buy X (sell Y), -1 = sell X
    notional: float  # trade notional in quote terms

# ---------------------------- core series ---------------------------------------

def _annual_to_step(mu_annual: float, sigma_annual: float, dt_days: float) -> Tuple[float, float]:
    """Convert annualized drift/vol to per‑step (in days) parameters for GBM."""
    mu_step = mu_annual * dt_days
    sigma_step = sigma_annual * np.sqrt(dt_days)
    return mu_step, sigma_step


def make_synth_series(
    T: int,
    *,
    seed: int = 0,
    dt: float = 1.0 / 1440.0,  # 1 minute in days
    s0: float = 100.0,
    mu_annual: float = 0.0,
    sigma_annual: float = 0.7,
    regimes: Optional[List[Tuple[int, float]]] = None,
    jump_prob: float = 0.0,
    jump_mu: float = 0.0,
    jump_sigma: float = 0.04,
    vol_of_vol: float = 0.0,
) -> dict:
    """Generate a synthetic price/volume/valuation dataset.

    Parameters
    ----------
    T : number of steps (minutes if dt=1/1440)
    regimes : optional list of (steps, sigma_annual) segments for regime switching.
              If None, a single regime with sigma_annual is used.
    jump_prob : probability per step of a jump (Poisson); jump size ~ N(jump_mu, jump_sigma).
    vol_of_vol : if >0, adds slow stochastic drift to sigma via AR(1) on log‑sigma.

    Returns a dict with keys: 'price', 'v', 'volume', 'trend', 'sigma_step' (arrays).
    """
    g = np.random.default_rng(seed)

    # Build per‑step sigma timeline (annualized → step) with optional regimes
    if regimes is None:
        regimes = [(T, sigma_annual)]
    sigmas = []
    for steps, sigA in regimes:
        sigmas.extend([sigA] * steps)
    sigmas = np.array(sigmas[:T], dtype=float)
    if len(sigmas) < T:
        sigmas = np.pad(sigmas, (0, T - len(sigmas)), mode='edge')

    # Stochastic volatility (log‑sigma AR(1))
    if vol_of_vol > 0.0:
        log_sig = np.log(np.maximum(1e-8, sigmas))
        rho = 0.995  # very persistent
        noise = vol_of_vol * g.standard_normal(T)
        for t in range(1, T):
            log_sig[t] = rho * log_sig[t - 1] + (1 - rho) * log_sig[t] + noise[t]
        sigmas = np.exp(log_sig)

    mu_step, _ = _annual_to_step(mu_annual, 0.0, dt)  # drift only

    # simulate GBM with per‑step sigma
    price = np.empty(T, dtype=float)
    price[0] = s0
    for t in range(1, T):
        _, sigma_step = _annual_to_step(0.0, sigmas[t], dt)
        z = g.standard_normal()
        # Jump component (log‑normal jump multiplier)
        J = 0.0
        if jump_prob > 0.0 and g.random() < jump_prob:
            J = g.normal(jump_mu, jump_sigma)
        # GBM update in log space
        dlogS = (mu_step - 0.5 * sigma_step**2) + sigma_step * z + J
        price[t] = max(1e-8, price[t - 1] * np.exp(dlogS))

    # volumes tied to |returns| (more vol → more volume) with noise
    ret = np.r_[0.0, np.diff(np.log(price))]
    vol_signal = np.abs(ret)
    base_vol = 1.0
    volume = base_vol * (1.0 + 50.0 * vol_signal)
    volume *= np.exp(0.2 * g.standard_normal(T))  # log‑normal noise
    volume = np.maximum(1e-6, volume)

    # simple trend indicator (EMA of returns)
    lam = 0.02
    trend = np.empty(T)
    m = 0.0
    for t in range(T):
        m = (1 - lam) * m + lam * ret[t]
        trend[t] = m

    v = normalize_price_to_valuation(price)

    # store per‑step sigma in step units for analysis
    sigma_step_arr = np.array([_annual_to_step(0.0, sA, dt)[1] for sA in sigmas])

    return {
        'price': price,
        'v': v,
        'volume': volume,
        'trend': trend,
        'sigma_step': sigma_step_arr,
    }

# ---------------------------- trades --------------------------------------------

def make_trade_stream(
    price: np.ndarray,
    *,
    seed: int = 0,
    trades_per_hour: Tuple[int, int] = (40, 120),
    mean_notional: float = 5_000.0,
    skew_with_trend: bool = True,
) -> List[Trade]:
    """Generate a list of taker trades against the pool.

    • Trades per hour is uniformly sampled in the given range and mapped to per‑minute.
    • Notional is log‑normal around `mean_notional`.
    • If `skew_with_trend`, buys are more likely when recent return > 0.
    """
    g = np.random.default_rng(seed)
    T = len(price)

    # approximate per‑minute rate
    lam_min = g.integers(trades_per_hour[0], trades_per_hour[1] + 1) / 60.0

    # quick return signal for skew
    ret = np.r_[0.0, np.diff(np.log(price))]

    trades: List[Trade] = []
    for t in range(T):
        # Poisson number of trades this minute (0,1,2...)
        k = g.poisson(lam_min)
        if k == 0:
            continue
        # side probability
        p_buy = 0.5
        if skew_with_trend:
            p_buy = 0.5 + 0.4 * np.tanh(8.0 * ret[max(0, t - 5):t + 1].mean())
            p_buy = float(np.clip(p_buy, 0.05, 0.95))
        for _ in range(k):
            side = 1 if g.random() < p_buy else -1
            # log‑normal size
            ln_mu = np.log(max(1e-6, mean_notional)) - 0.5 * 0.6**2
            notional = float(np.exp(g.normal(ln_mu, 0.6)))
            trades.append(Trade(t=t, side=side, notional=notional))
    return trades

# ---------------------------- demo ----------------------------------------------
if __name__ == '__main__':
    ds = make_synth_series(
        10_000,
        seed=42,
        regimes=[(4000, 0.4), (3000, 0.9), (3000, 0.6)],
        jump_prob=0.0005,
        jump_mu=0.0,
        jump_sigma=0.06,
        vol_of_vol=0.05,
    )
    print('series:', {k: (v.shape if hasattr(v, 'shape') else type(v)) for k, v in ds.items()})
    trades = make_trade_stream(ds['price'], seed=7)
    print('num trades:', len(trades), 'first 3:', trades[:3])

