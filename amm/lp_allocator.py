"""
LP Allocator — Gaussian, valuation space
----------------------------------------
This module turns a *predicted valuation* (µ) and dispersion (σ)
into a *concentrated liquidity plan* across a discrete grid of
valuation ticks v∈(0,1).

The intent mirrors the paper: create a "Gaussian incentive bump" so
that LP depth is waiting near the *predicted* valuation rather than
the current spot. You can feed the allocations to a simulator or to a
frontend that suggests LP ranges.

Core ideas
----------
1) Build weights w_i ∝ N(v_i | µ, σ) over valuation ticks {v_i}.
2) Normalize to 1 and multiply by total liquidity L_total to get per‑tick L_i.
3) Optionally collapse adjacent ticks into *ranges* to reduce fragmentation.

Public API
----------
• gaussian_weights(ticks, mu, sigma, *, floor=0.0) -> np.ndarray
• allocate_liquidity(total_L, ticks, mu, sigma, *, min_per_tick=0.0) -> np.ndarray
• make_ranges_from_weights(weights, ticks, *, target_num_ranges=8, min_width=1,
                           prune_below=1e-8) -> List[Range]
• ticks_from_bounds(v_min, v_max, n) -> np.ndarray

Notation
--------
• ticks: strictly increasing 1‑D array in (0,1); can be price‑mapped later.
• weights: non‑negative vector matching ticks length.
• A Range is (i0, i1, L) meaning inclusive index segment [i0, i1] with liquidity L.

This file is model‑agnostic: pass in any µ,σ from your predictor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

EPS = 1e-12

# ------------------------ basics -------------------------------------------------

def ticks_from_bounds(v_min: float, v_max: float, n: int) -> np.ndarray:
    assert 0.0 < v_min < v_max < 1.0, "bounds must be within (0,1)"
    assert n >= 2, "need at least 2 ticks"
    return np.linspace(v_min, v_max, n, dtype=float)


def gaussian_weights(ticks: np.ndarray, mu: float, sigma: float, *, floor: float = 0.0) -> np.ndarray:
    ticks = np.asarray(ticks, dtype=float)
    assert ticks.ndim == 1 and len(ticks) >= 2
    assert sigma > 0.0, "sigma must be > 0"
    z = (ticks - mu) / sigma
    w = np.exp(-0.5 * z * z)
    if floor > 0:
        w = np.maximum(w, floor)
    return w


def allocate_liquidity(
    total_L: float,
    ticks: np.ndarray,
    mu: float,
    sigma: float,
    *,
    min_per_tick: float = 0.0,
) -> np.ndarray:
    """Allocate total_L across ticks using Gaussian weights.

    min_per_tick provides a uniform base layer (useful for tails) before the
    Gaussian share is added. Final result is an array L_i summing to total_L.
    """
    assert total_L >= 0
    base = np.full(len(ticks), min_per_tick, dtype=float)
    L_base = base.sum()
    rem = max(0.0, total_L - L_base)

    w = gaussian_weights(ticks, mu, sigma)
    s = w.sum()
    if s < EPS:
        # Degenerate: spread equally (or keep only base if total < base)
        out = base.copy()
        if rem > 0:
            out += rem / len(ticks)
        return out

    out = base + rem * (w / s)
    # ensure exact mass conservation (floating rounding)
    scale = total_L / max(EPS, out.sum())
    return out * scale

# ------------------------ ranges -------------------------------------------------

@dataclass
class Range:
    i0: int
    i1: int  # inclusive
    L: float


def make_ranges_from_weights(
    weights: np.ndarray,
    ticks: np.ndarray,
    *,
    target_num_ranges: int = 8,
    min_width: int = 1,
    prune_below: float = 1e-8,
) -> List[Range]:
    """Collapse per‑tick weights into contiguous ranges.

    Greedy algorithm:
      1) Start from the highest‑weight tick, expand left/right while the marginal
         average weight increases, respecting min_width.
      2) Mark the covered segment as taken, then repeat from the next highest
         uncovered tick until we reach target_num_ranges or run out of mass.

    Returns a list of Range(i0,i1,L) where L is proportional to the mass captured
    in that segment (sum of weights within [i0,i1]). The caller can scale L by any
    total liquidity they plan to deploy.
    """
    w = np.asarray(weights, dtype=float).copy()
    ticks = np.asarray(ticks, dtype=float)
    assert w.ndim == 1 and ticks.ndim == 1 and len(w) == len(ticks)

    # mask of uncovered ticks
    used = np.zeros_like(w, dtype=bool)
    ranges: List[Range] = []

    def best_seed() -> int | None:
        # pick highest‑weight unused tick above prune threshold
        cand = np.where(~used & (w >= prune_below))[0]
        if cand.size == 0:
            return None
        idx = cand[np.argmax(w[cand])]
        return int(idx)

    N = len(w)
    while len(ranges) < target_num_ranges:
        i = best_seed()
        if i is None:
            break
        # expand a segment around i
        l = r = i
        mass = w[i]
        used[i] = True

        # ensure min_width first
        while (r - l + 1) < min_width:
            # try expand to the side with higher adjacent weight
            left_ok = l - 1 >= 0 and not used[l - 1]
            right_ok = r + 1 < N and not used[r + 1]
            if left_ok and right_ok:
                if w[l - 1] >= w[r + 1]:
                    l -= 1; mass += w[l]; used[l] = True
                else:
                    r += 1; mass += w[r]; used[r] = True
            elif left_ok:
                l -= 1; mass += w[l]; used[l] = True
            elif right_ok:
                r += 1; mass += w[r]; used[r] = True
            else:
                break

        # opportunistic expansion while it *improves average*
        improved = True
        while improved:
            improved = False
            avg = mass / (r - l + 1)
            # evaluate left expansion
            if l - 1 >= 0 and not used[l - 1]:
                new_avg = (mass + w[l - 1]) / (r - (l - 1) + 1)
                if new_avg >= avg:
                    l -= 1; mass += w[l]; used[l] = True; improved = True
            # evaluate right expansion
            if r + 1 < N and not used[r + 1]:
                new_avg = (mass + w[r + 1]) / ((r + 1) - l + 1)
                if new_avg >= avg:
                    r += 1; mass += w[r]; used[r] = True; improved = True

        ranges.append(Range(i0=l, i1=r, L=float(mass)))

        # stop if remaining uncovered mass is tiny
        remaining_mass = float(w[~used].sum())
        if remaining_mass < prune_below:
            break

    return ranges

# ------------------------ demo ---------------------------------------------------
if __name__ == "__main__":
    ticks = ticks_from_bounds(0.2, 0.8, 41)  # 41 ticks in valuation space
    mu, sigma = 0.5, 0.05
    total_L = 1_000.0

    L_i = allocate_liquidity(total_L, ticks, mu, sigma, min_per_tick=0.0)
    print("per‑tick allocations sum=", L_i.sum())

    ranges = make_ranges_from_weights(L_i / total_L, ticks, target_num_ranges=6, min_width=2)
    print("ranges (i0,i1,L_mass):", [(r.i0, r.i1, round(r.L,4)) for r in ranges])
