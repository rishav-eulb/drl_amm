"""
Event-driven dataset builder for the predictive AMM project (PAPER-COMPLIANT)
-----------------------------------------------------------------------------
FIXED: Event generation now uses RELATIVE threshold (1 ± β_v) as per Algorithm 1
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import numpy as np

EPS = 1e-9

# ------------------------------ Valuation utils --------------------------------

def clip01(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    return np.minimum(1.0 - eps, np.maximum(eps, x))


def normalize_price_to_valuation(price: np.ndarray) -> np.ndarray:
    """Map price -> valuation v in (0,1).
    For CPMM: v = price / (1 + price) so that v/(1-v) = price.
    """
    price = np.asarray(price, dtype=float)
    return clip01(price / (1.0 + price))

# ------------------------------ Event definition --------------------------------

@dataclass
class Event:
    t: int      # index in the regular series where the event fired
    v: float    # valuation at t (already in (0,1))
    dv: float   # signed delta vs last event valuation

# ------------------------------ Core builders (PAPER-COMPLIANT) -----------------

def make_events(v: np.ndarray, beta_v: float) -> List[Event]:
    """Generate price-based events using RELATIVE threshold (PAPER Algorithm 1).

    Paper (Algorithm 1, lines 3-7):
```
    upper ← v_t(1 + β_v)
    lower ← v_t(1 − β_v)
    if upper ≤ v_{t+k} ≤ lower then
        step ← True
```
    
    We emit an event at index i whenever:
    v[i] >= v[last] * (1 + β_v)  OR  v[i] <= v[last] * (1 - β_v)
    
    This is DIFFERENT from original implementation which used:
    |v[i] - v[last]| >= β_v  (absolute threshold)
    """
    v = clip01(np.asarray(v, dtype=float))
    assert v.ndim == 1, "v must be 1-D"
    assert 0 < beta_v < 1, "beta_v must be in (0,1)"

    events: List[Event] = []
    last_idx = 0
    last_v = v[0]
    events.append(Event(t=0, v=float(last_v), dv=0.0))

    for i in range(1, len(v)):
        # Paper Algorithm 1: relative threshold
        upper = last_v * (1.0 + beta_v)
        lower = last_v * (1.0 - beta_v)
        
        # Event fires if current valuation crosses bounds
        if v[i] >= upper or v[i] <= lower:
            dv = float(v[i] - last_v)
            events.append(Event(t=i, v=float(v[i]), dv=dv))
            last_v = v[i]
            last_idx = i
            
    return events


def build_windows(series: np.ndarray, win: int) -> np.ndarray:
    """
    Build fixed-length windows.
    - If series is 1-D [T] -> returns [N, win]
    - If series is 2-D [T, D] -> returns [N, win, D]
    where N = T - win + 1.
    """
    x = np.asarray(series, dtype=float)
    assert win >= 2, "window must be >= 2"

    if x.ndim == 1:
        N = len(x) - win + 1
        if N <= 0:
            return np.zeros((0, win), dtype=float)
        out = np.lib.stride_tricks.sliding_window_view(x, win_shape := win)
        return out.copy()

    elif x.ndim == 2:
        T, D = x.shape
        N = T - win + 1
        if N <= 0:
            return np.zeros((0, win, D), dtype=float)
        out = np.lib.stride_tricks.sliding_window_view(x, (win, D))
        out = out.reshape(T - win + 1, win, D)
        return out.copy()

    else:
        raise AssertionError("series must be 1-D or 2-D")


def align_features(features: Dict[str, np.ndarray]) -> np.ndarray:
    """Column-stack features after validating equal lengths."""
    keys = list(features.keys())
    if not keys:
        raise ValueError("features cannot be empty")
    L = len(features[keys[0]])
    for k in keys:
        if features[k].ndim != 1:
            raise ValueError(f"feature {k} must be 1-D")
        if len(features[k]) != L:
            raise ValueError("all features must share equal length")
    cols = [np.asarray(features[k], dtype=float) for k in keys]
    return np.stack(cols, axis=1)


def scale_minmax(X: np.ndarray, clip: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    lo = X.min(axis=0, keepdims=True)
    hi = X.max(axis=0, keepdims=True)
    rng = np.maximum(hi - lo, 1e-12)
    Xn = (X - lo) / rng
    Xn = np.clip(Xn, clip[0], clip[1])
    return Xn

# ------------------------------ High-level dataset ------------------------------

def make_event_dataset(v, tau, beta_v=0.01, lstm_win=50, lookahead=1):
    """
    v: 1-D valuation array
    tau: 2-D array or None
    """
    v = np.asarray(v, dtype=float)

    # Feature frame construction
    if tau is not None:
        tau = np.asarray(tau, dtype=float)
        feats = np.concatenate([v[:, None], tau], axis=1)
    else:
        feats = v[:, None]

    # Build windows
    Xwin = build_windows(feats, lstm_win)

    # Build event stream (using RELATIVE threshold - PAPER Algorithm 1)
    events = make_events(v, beta_v)

    return {
        "events": events,
        "Xwin": Xwin,
        "feats": feats,
    }

# ------------------------------ Self-test ---------------------------------------
if __name__ == "__main__":
    print("="*70)
    print("EVENT GENERATION TEST (PAPER-COMPLIANT VERSION)")
    print("="*70)
    
    # Synthetic demo
    T = 300
    rng = np.random.default_rng(0)
    price = np.cumprod(1.0 + 0.001 * rng.standard_normal(T))
    v = normalize_price_to_valuation(price)
    
    # Test both methods
    print("\n1. Testing RELATIVE threshold (Paper Algorithm 1):")
    beta_v_rel = 0.01  # 1% relative change
    events_paper = make_events(v, beta_v_rel)
    print(f"   β_v = {beta_v_rel} (relative)")
    print(f"   Events: {len(events_paper)} ({100*len(events_paper)/T:.2f}%)")
    print(f"   First 3 events: ")
    for e in events_paper[:3]:
        print(f"     t={e.t}, v={e.v:.6f}, Δv={e.dv:+.6f}")
    
    print("\n2. Comparison with absolute threshold:")
    # Old method for comparison
    def make_events_absolute(v, beta_v):
        """Original implementation - ABSOLUTE threshold."""
        v = clip01(v)
        events = []
        last_v = v[0]
        events.append(Event(t=0, v=float(last_v), dv=0.0))
        for i in range(1, len(v)):
            if abs(v[i] - last_v) >= beta_v:
                events.append(Event(t=i, v=float(v[i]), dv=float(v[i] - last_v)))
                last_v = v[i]
        return events
    
    events_abs = make_events_absolute(v, beta_v_rel)
    print(f"   β_v = {beta_v_rel} (absolute)")
    print(f"   Events: {len(events_abs)} ({100*len(events_abs)/T:.2f}%)")
    
    print(f"\n3. Difference:")
    print(f"   Relative method: {len(events_paper)} events")
    print(f"   Absolute method: {len(events_abs)} events")
    print(f"   Ratio: {len(events_paper)/len(events_abs):.2f}x")
    
    print("\n" + "="*70)
    print("✓ Paper-compliant event generation (relative threshold)")
    print("="*70)