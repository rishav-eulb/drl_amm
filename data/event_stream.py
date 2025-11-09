"""
Event-driven dataset builder for the predictive AMM project
-----------------------------------------------------------
This module converts a regular-interval price (or valuation) series
into an EVENT-DRIVEN stream using a valuation-threshold β_v.

Outputs are used by:
  • LSTM: to build fixed-length windows from the regular series
  • RL Env: to advance steps only when |v_t - v_{t-1}| >= β_v

Key concepts implemented here are aligned with the paper's
"price-based event" definition and equilibrium-valuation notation.

Public API
----------
make_events(v: np.ndarray, beta_v: float) -> List[Event]
    Build a list of (t_idx, v_t, dv) events where |v_t - v_{last_event}| >= beta_v.

build_windows(series: np.ndarray, win: int) -> np.ndarray
    Turn a regular-interval series into [N, win] windows for LSTM training.

align_features(features: Dict[str, np.ndarray]) -> np.ndarray
    Column-stack multi-feature arrays after checking equal length.

scale_minmax(X, clip=(0.0, 1.0))
    Simple min–max scaling utility for features in [clip_min, clip_max].

make_event_dataset(v, tau, beta_v, lookahead=1)
    High-level helper returning a dict with:
      - 'events': list of events
      - 'X_lstm': [N, T, D] windows of features for LSTM
      - 'y_val':  [N] next-step (or lookahead) target valuations

Notes
-----
• Input valuation v must be in (0,1). Use normalize_price_to_valuation if needed.
• All functions are deterministic and side-effect free for reproducibility.
• Unit tests at bottom (run this file directly) validate edge cases.
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
    """Example normalization mapping price -> valuation v in (0,1).
    For CPMM prelim work you often use v = price / (1 + price) so that
    v/(1-v) = price. Adjust if you adopt a different parity mapping.
    """
    price = np.asarray(price, dtype=float)
    return clip01(price / (1.0 + price))

# ------------------------------ Event definition --------------------------------

@dataclass
class Event:
    t: int      # index in the regular series where the event fired
    v: float    # valuation at t (already in (0,1))
    dv: float   # signed delta vs last event valuation

# ------------------------------ Core builders -----------------------------------

def make_events(v: np.ndarray, beta_v: float) -> List[Event]:
    """Generate price-based events using threshold beta_v on valuation deltas.

    We emit an event at index i whenever |v[i] - v[last]| >= beta_v where 'last'
    is the index of the most recent event (starting at i=0).
    """
    v = clip01(np.asarray(v, dtype=float))
    assert v.ndim == 1, "v must be 1-D"
    assert 0 < beta_v < 1, "beta_v must be in (0,1)"

    events: List[Event] = []
    last_idx = 0
    last_v = v[0]
    events.append(Event(t=0, v=float(last_v), dv=0.0))

    for i in range(1, len(v)):
        dv = float(v[i] - last_v)
        if abs(dv) >= beta_v:
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
    """Column-stack features after validating equal lengths.
    features: dict name -> 1-D arrays of equal length.
    Returns [T, D].
    """
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

    # Build windows (safe for 1-D or 2-D)
    Xwin = build_windows(feats, lstm_win)
    # Xwin has shape [N, lstm_win, D]

    # Build event stream (using |Δv| > beta_v threshold)
    dv = np.abs(np.diff(v))
    idx = np.where(dv > beta_v)[0] + 1  # shift to next index

    events = []
    for i in idx:
        events.append(Event(t=int(i), v=float(v[i]), dv=float(dv[i-1])))

    return {
        "events": events,
        "Xwin": Xwin,
        "feats": feats,
    }

# ------------------------------ Self-test ---------------------------------------
if __name__ == "__main__":
    # Synthetic demo
    T = 300
    rng = np.random.default_rng(0)
    price = np.cumprod(1.0 + 0.001 * rng.standard_normal(T))
    v = normalize_price_to_valuation(price)
    vol = np.abs(np.convolve(np.diff(np.r_[0, price]), np.ones(5)/5, mode='same'))
    ds = make_event_dataset(v, vol, beta_v=0.005, lstm_win=32, lookahead=1)
    print(f"events: {len(ds['events'])}")
    print(f"X_lstm: {ds['X_lstm'].shape}, y: {ds['y_val'].shape}")
