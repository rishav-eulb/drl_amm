"""
Evaluation utilities for AMM simulations
---------------------------------------
This module implements the paper's headline metrics and a few extras so
we can compare a baseline CPMM vs. the predictive cAMM.

Metrics
-------
1) Liquidity utilization
   • Definition (practical): total traded notional over average pool TVL (in quote units).
   • TVL_t (quote) = y_t + price_t * x_t.
   • utilization = sum(notional_traded) / mean(TVL_t).

2) Liquidity concentration (distribution quality)
   • Given per‑tick liquidity weights w (>=0), we compute:
     - HHI = sum((w/∑w)^2)  (higher = more concentrated)
     - Entropy H = −∑ p_i log p_i  (lower = more concentrated)
     - Effective ticks Neff = 1/HHI (number of equally‑weighted ticks with same concentration)

3) Depth at target price impact
   • For a UniV2‑like pool we can query pool.depth_for_price_impact().
   • For generic pools we provide a monotone numeric helper that
     finds the input size that causes the impact.

4) Divergence, slippage, and load
   • Using valuation series v_t in (0,1) and CPMM invariant c, we compute
     stepwise loss_div(v_t, v_{t+1}), loss_slip_X(v_t, v_{t+1}), and load.

5) Slippage stats from trade impacts
   • Given per‑trade fractional price changes, report mean/median/p95.

All functions are pure and NumPy‑friendly for easy notebook use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple, Optional
import numpy as np

from amm.math import (
    clip01,
    divergence_loss,
    slippage_loss_X,
    load_auto,
)
from data.event_stream import normalize_price_to_valuation

# ------------------------- 1) Utilization ---------------------------------------

def liquidity_utilization(
    volume_quote: np.ndarray,
    x_hist: np.ndarray,
    y_hist: np.ndarray,
    price_series: np.ndarray,
) -> float:
    """Total traded notional divided by average TVL in quote units.

    • volume_quote: per‑trade notionals; if you have multiple trades per step,
      pass the flat array (we sum it). If you have per‑step totals, pass that array.
    • TVL_t (quote) = y_t + price_t * x_t.
    """
    vol = float(np.sum(np.asarray(volume_quote, dtype=float)))
    x = np.asarray(x_hist, dtype=float)
    y = np.asarray(y_hist, dtype=float)
    p = np.asarray(price_series, dtype=float)
    tvl = y + p * x
    tvl_mean = float(np.mean(tvl))
    if tvl_mean <= 0:
        return 0.0
    return vol / tvl_mean

# ------------------------- 2) Concentration -------------------------------------

def concentration_metrics(weights: np.ndarray) -> Dict[str, float]:
    """Compute HHI, entropy, and effective support of a non‑negative weight vector."""
    w = np.maximum(0.0, np.asarray(weights, dtype=float))
    s = float(w.sum())
    if s <= 0:
        return {"hhi": 0.0, "entropy": 0.0, "neff": 0.0}
    p = w / s
    hhi = float(np.sum(p * p))
    # entropy with stability
    p_safe = np.clip(p, 1e-12, 1.0)
    ent = float(-np.sum(p_safe * np.log(p_safe)))
    neff = 1.0 / hhi
    return {"hhi": hhi, "entropy": ent, "neff": neff}

# ------------------------- 3) Depth ---------------------------------------------

def depth_numeric(
    x: float,
    y: float,
    fee_bps: float,
    impact: float,
    direction: str = "x_to_y",
    *,
    iters: int = 40,
) -> float:
    """Generic depth finder for a CPMM with fee on input using binary search.

    This mirrors UniV2LikePool.depth_for_price_impact but works standalone.
    """
    assert 0 < impact < 1
    fee = fee_bps / 10_000.0
    c = x * y
    p0 = y / x
    target = p0 * (1.0 - impact if direction == "x_to_y" else 1.0 + impact)

    lo, hi = 0.0, 0.01 * (x if direction == "x_to_y" else y)
    for _ in range(32):
        if direction == "x_to_y":
            dx = hi * (1.0 - fee)
            x_new = x + dx
            y_new = c / x_new
            p1 = y_new / x_new
            cond = p1 <= target
        else:
            dy = hi * (1.0 - fee)
            y_new = y + dy
            x_new = c / y_new
            p1 = y_new / x_new
            cond = p1 >= target
        if cond:
            break
        hi *= 2.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if direction == "x_to_y":
            dx = mid * (1.0 - fee)
            x_new = x + dx
            y_new = c / x_new
            p1 = y_new / x_new
            cond = p1 <= target
        else:
            dy = mid * (1.0 - fee)
            y_new = y + dy
            x_new = c / y_new
            p1 = y_new / x_new
            cond = p1 >= target
        if cond:
            hi = mid
        else:
            lo = mid
    return hi

# ------------------------- 4) Divergence / slippage / load ----------------------

def path_losses(
    price_series: np.ndarray,
    c: float,
) -> Dict[str, float]:
    """Compute average per‑step losses over a price path for a CPMM invariant c.

    Steps:
      • Map price→valuation via v = price/(1+price).
      • For consecutive steps (t,t+1) compute divergence, slippage_X, load_auto.
      • Return averages and totals.
    """
    price = np.asarray(price_series, dtype=float)
    v = normalize_price_to_valuation(price)
    divs = []
    slips = []
    loads = []
    for t in range(len(v) - 1):
        vt, vp = clip01(v[t]), clip01(v[t + 1])
        divs.append(divergence_loss(vt, vp, c))
        slips.append(slippage_loss_X(vt, vp, c))
        loads.append(load_auto(vt, vp, c))
    divs = np.array(divs)
    slips = np.array(slips)
    loads = np.array(loads)
    return {
        "div_mean": float(divs.mean()) if len(divs) else 0.0,
        "slip_mean": float(slips.mean()) if len(slips) else 0.0,
        "load_mean": float(loads.mean()) if len(loads) else 0.0,
        "div_sum": float(divs.sum()),
        "slip_sum": float(slips.sum()),
        "load_sum": float(loads.sum()),
    }

# ------------------------- 5) Slippage from impacts -----------------------------

def impact_stats(impacts: np.ndarray) -> Dict[str, float]:
    x = np.asarray(impacts, dtype=float)
    if x.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
    }

# ------------------------- Bundle helpers --------------------------------------

def summarize_baseline_run(res: Dict[str, np.ndarray], price_series: np.ndarray) -> Dict[str, float]:
    util = liquidity_utilization(
        res.get("volume_quote", np.array([])), res["x"], res["y"], price_series
    )
    conc = None  # baseline has no per‑tick allocations; provided by caller if needed
    losses = path_losses(price_series, c=float(res["x"][0] * res["y"][0]))
    imp = impact_stats(res.get("impact", np.array([])))
    summary = {
        "utilization": util,
        "div_mean": losses["div_mean"],
        "slip_mean": losses["slip_mean"],
        "load_mean": losses["load_mean"],
        "impact_mean": imp["mean"],
        "impact_p95": imp["p95"],
    }
    return summary

# ------------------------- demo -------------------------------------------------
if __name__ == "__main__":
    from sim.synth import make_synth_series, make_trade_stream
    from sim.baseline import UniV2LikePool, run_baseline

    ds = make_synth_series(8000, seed=3)
    trades = make_trade_stream(ds['price'], seed=4)
    pool = UniV2LikePool(x=120_000.0, y=120_000.0, fee_bps=30)
    res = run_baseline(pool, ds['price'], trades)
    summary = summarize_baseline_run(res, ds['price'])
    print("baseline summary:", summary)
