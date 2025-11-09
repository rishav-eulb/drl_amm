"""
Baseline simulator — Uniswap‑style CPMM (no predictive shifts)
-------------------------------------------------------------
This module provides a simple, reproducible baseline to compare against
our predictive cAMM. It simulates trades against a constant‑product pool
with a fixed fee and **no** pseudo‑arbitrage or predictive incentives.

Key features
------------
• UniV2‑like pool with fee on input
• Exact CPMM swap math
• Streaming execution of a taker trade list
• Logs per‑trade price impact, fees, and inventory
• Utility to estimate depth (= trade size for target price impact)

Outputs are compatible with a forthcoming `sim/eval.py`.

Usage
-----
from sim.baseline import UniV2LikePool, run_baseline

pool = UniV2LikePool(x0=100_000.0, y0=100_000.0, fee_bps=30)
result = run_baseline(pool, price_series, trades)

"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import numpy as np

# ----------------------------- Pool --------------------------------------------

@dataclass
class UniV2LikePool:
    x: float
    y: float
    fee_bps: int = 30  # 0.30%
    name: str = "baseline_univ2"

    # --- invariants/helpers ---
    @property
    def c(self) -> float:
        return self.x * self.y

    @property
    def price_y_per_x(self) -> float:
        return self.y / self.x

    @staticmethod
    def _apply_fee(dx: float, fee_bps: int) -> float:
        return dx * (1.0 - fee_bps / 10_000.0)

    # --- swaps ---
    def swap_x_for_y(self, dx: float) -> float:
        if dx <= 0:
            return 0.0
        dx_eff = self._apply_fee(dx, self.fee_bps)
        c = self.c
        x_new = self.x + dx_eff
        y_new = c / x_new
        dy = self.y - y_new
        self.x, self.y = x_new, y_new
        return dy

    def swap_y_for_x(self, dy: float) -> float:
        if dy <= 0:
            return 0.0
        dy_eff = self._apply_fee(dy, self.fee_bps)
        c = self.c
        y_new = self.y + dy_eff
        x_new = c / y_new
        dx_out = self.x - x_new
        self.x, self.y = x_new, y_new
        return dx_out

    # --- depth estimation ---
    def depth_for_price_impact(self, impact: float, direction: str = "x_to_y") -> float:
        """Return input size (dx or dy) that causes <= `impact` fractional price move.

        direction: 'x_to_y' for buying Y with X; 'y_to_x' for buying X with Y.
        Impact is measured vs current mid p=y/x. After an X->Y swap of dx,
        new mid is p' = y'/x'. We find minimal dx such that |p'/p - 1| >= impact.
        Closed form is messy with fee; we use a monotone numeric search.
        """
        assert 0 < impact < 1
        p0 = self.price_y_per_x
        target = p0 * (1.0 - impact if direction == "x_to_y" else 1.0 + impact)

        # binary search on input size (start with a scale of 1% of reserves)
        lo, hi = 0.0, 0.01 * (self.x if direction == "x_to_y" else self.y)
        # expand hi until impact reached
        for _ in range(32):
            # simulate without mutating state
            if direction == "x_to_y":
                dx = hi * (1.0 - self.fee_bps / 10_000.0)
                x_new = self.x + dx
                y_new = self.c / x_new
                p1 = y_new / x_new
                cond = p1 <= target
            else:
                dy = hi * (1.0 - self.fee_bps / 10_000.0)
                y_new = self.y + dy
                x_new = self.c / y_new
                p1 = y_new / x_new
                cond = p1 >= target
            if cond:
                break
            hi *= 2.0
        # binary refine
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if direction == "x_to_y":
                dx = mid * (1.0 - self.fee_bps / 10_000.0)
                x_new = self.x + dx
                y_new = self.c / x_new
                p1 = y_new / x_new
                cond = p1 <= target
            else:
                dy = mid * (1.0 - self.fee_bps / 10_000.0)
                y_new = self.y + dy
                x_new = self.c / y_new
                p1 = y_new / x_new
                cond = p1 >= target
            if cond:
                hi = mid
            else:
                lo = mid
        return hi

# ----------------------------- Simulator ---------------------------------------

def run_baseline(
    pool: UniV2LikePool,
    price_series: np.ndarray,
    trades: List,
) -> Dict[str, np.ndarray | float | int]:
    """Run a trade stream against the baseline pool.

    Parameters
    ----------
    pool : UniV2LikePool
    price_series : used only for analytics/alignment; pool is price‑agnostic
    trades : list of objects with fields (t, side, notional)

    Returns a dict with arrays suitable for downstream eval:
      - 'mid': midprice after each minute (y/x)
      - 'x', 'y': reserves timelines
      - 'fees': cumulative fees collected
      - 'impact': per‑trade fractional price impact
      - 'volume_quote': per‑trade notional processed
    """
    T = len(price_series)
    x_hist = np.empty(T)
    y_hist = np.empty(T)
    mid_hist = np.empty(T)
    fees_cum = np.zeros(T)

    t_ptr = 0
    fee_rate = pool.fee_bps / 10_000.0

    # group trades by time index
    by_t: Dict[int, List] = {}
    for tr in trades:
        by_t.setdefault(tr.t, []).append(tr)

    volume_quote: List[float] = []
    impact: List[float] = []

    for t in range(T):
        # execute all trades at time t
        if t in by_t:
            for tr in by_t[t]:
                p_before = pool.price_y_per_x
                if tr.side == 1:
                    # trader buys Y with X: pay X notional / price
                    # approximate dx via notional / price_before
                    dx_in = tr.notional / max(1e-12, p_before)
                    dy_out = pool.swap_x_for_y(dx_in)
                    fees = dx_in * fee_rate
                else:
                    # trader buys X with Y
                    dy_in = tr.notional
                    dx_out = pool.swap_y_for_x(dy_in)
                    fees = dy_in * fee_rate
                p_after = pool.price_y_per_x
                volume_quote.append(float(tr.notional))
                impact.append(abs(p_after / max(1e-12, p_before) - 1.0))
                # add fees to cumulative (at index t)
                fees_cum[t] += fees

        # snapshot
        x_hist[t] = pool.x
        y_hist[t] = pool.y
        mid_hist[t] = pool.price_y_per_x
        if t > 0:
            fees_cum[t] += fees_cum[t - 1]

    return {
        'x': x_hist,
        'y': y_hist,
        'mid': mid_hist,
        'fees_cum': fees_cum,
        'impact': np.array(impact, dtype=float),
        'volume_quote': np.array(volume_quote, dtype=float),
        'fee_bps': pool.fee_bps,
        'name': pool.name,
    }

# ----------------------------- demo --------------------------------------------
if __name__ == "__main__":
    # quick smoke test
    import numpy as np
    from sim.synth import make_synth_series, make_trade_stream

    ds = make_synth_series(5000, seed=1)
    trades = make_trade_stream(ds['price'], seed=2)
    pool = UniV2LikePool(x=100_000.0, y=100_000.0, fee_bps=30)
    res = run_baseline(pool, ds['price'], trades)
    print({k: (v.shape if hasattr(v, 'shape') else v) for k, v in res.items() if k in ('x','y','mid','fees_cum')})
    print("trades:", len(res['impact']), "avg impact:", float(res['impact'].mean()))
