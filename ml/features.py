"""
Feature engineering utilities (τ) for predictive AMM
----------------------------------------------------
This module builds supervised inputs for the LSTM forecaster (v′_p) and
auxiliary signals for the RL environment. It focuses on *lightweight,*
stationary-ish transforms that are robust and fast to compute.

Design goals
------------
• Pure NumPy, deterministic, side‑effect free
• Works directly from regular‑interval price/volume arrays
• Outputs are already scaled/clipped to stable ranges where possible
• Compatible with data/event_stream.make_event_dataset()

Key APIs
--------
- basic_price_features(price): returns a dict of 1‑D arrays
- volume_features(volume, price): OBV‑like features
- realized_vol(price, win): rolling σ of log returns
- rsi(price, win): 0..1 RSI
- zscore(x, win): rolling z‑score
- feature_frame(price, volume=None, extra=None, config=...): stacks to [T, D]
- make_lstm_supervised(X, y, win, lookahead): -> Xseq [N, win, D], y [N]
- split_train_val_test(T, ratios=(0.7,0.15,0.15)) -> (idx tuples)
- StandardScaler: fit/transform/inverse_transform with μ/σ persistable

Conventions
-----------
• "price" is a strictly positive level series.
• Log‑returns r_t = log(P_t/P_{t-1}).
• Unless stated otherwise, outputs are clipped or normalized to [0,1] or ~[−3,3].

Note
----
If you want to feed *valuation* directly, call
`data.event_stream.normalize_price_to_valuation(price)` upstream and then
pass it in via `extra={"v": v}`.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
import numpy as np

# ------------------------- low‑level helpers ------------------------------------

def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(1e-12, x))


def _nan_to_num(x: np.ndarray, fill: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)


def _clip01(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return np.minimum(1.0 - eps, np.maximum(eps, x))

# ------------------------- price/volume primitives ------------------------------

def logret(price: np.ndarray) -> np.ndarray:
    p = np.asarray(price, dtype=float)
    r = np.r_[0.0, np.diff(_safe_log(p))]
    return r


def basic_price_features(price: np.ndarray, *, ema_spans: Tuple[int, int, int] = (5, 20, 60)) -> Dict[str, np.ndarray]:
    """Return a dict with stationary-ish transforms of price.

    Outputs:
      r: log returns
      ema_s, ema_m, ema_l: EMAs (normalized as ratio to price - 1)
      mom_1, mom_5, mom_20: momentum (P/P_{t-k} - 1)
    """
    p = np.asarray(price, dtype=float)
    T = len(p)
    r = logret(p)

    def ema(x: np.ndarray, span: int) -> np.ndarray:
        alpha = 2.0 / (span + 1.0)
        out = np.empty_like(x)
        m = x[0]
        out[0] = m
        for t in range(1, len(x)):
            m = (1 - alpha) * m + alpha * x[t]
            out[t] = m
        return out

    ema_s = ema(p, ema_spans[0]) / np.maximum(1e-12, p) - 1.0
    ema_m = ema(p, ema_spans[1]) / np.maximum(1e-12, p) - 1.0
    ema_l = ema(p, ema_spans[2]) / np.maximum(1e-12, p) - 1.0

    def momentum(k: int) -> np.ndarray:
        base = np.r_[np.full(k, np.nan), p[:-k]]
        out = p / np.maximum(1e-12, base) - 1.0
        out[:k] = 0.0
        return out

    mom_1 = momentum(1)
    mom_5 = momentum(5)
    mom_20 = momentum(20)

    return {
        "r": r,
        "ema_s": _nan_to_num(ema_s),
        "ema_m": _nan_to_num(ema_m),
        "ema_l": _nan_to_num(ema_l),
        "mom_1": _nan_to_num(mom_1),
        "mom_5": _nan_to_num(mom_5),
        "mom_20": _nan_to_num(mom_20),
    }


def realized_vol(price: np.ndarray, win: int = 30) -> np.ndarray:
    r = logret(price)
    # rolling std of returns
    out = np.zeros_like(r)
    buf = np.zeros(win)
    s2 = 0.0
    for t in range(len(r)):
        idx = t % win
        old = buf[idx]
        buf[idx] = r[t]
        s2 += r[t] * r[t] - old * old
        if t + 1 >= win:
            out[t] = np.sqrt(max(1e-18, s2 / win))
        else:
            out[t] = 0.0
    return out


def rsi(price: np.ndarray, win: int = 14) -> np.ndarray:
    p = np.asarray(price, dtype=float)
    delta = np.r_[0.0, np.diff(p)]
    up = np.maximum(0.0, delta)
    dn = np.maximum(0.0, -delta)
    # Wilder's smoothing
    def smooth(x):
        out = np.empty_like(x)
        out[0] = x[0]
        alpha = 1.0 / win
        for t in range(1, len(x)):
            out[t] = (1 - alpha) * out[t - 1] + alpha * x[t]
        return out
    avg_up = smooth(up)
    avg_dn = smooth(dn) + 1e-12
    rs = avg_up / avg_dn
    rsi = 1.0 - (1.0 / (1.0 + rs))  # map to 0..1
    return _clip01(rsi)


def volume_features(volume: np.ndarray, price: np.ndarray, *, win: int = 20) -> Dict[str, np.ndarray]:
    v = np.asarray(volume, dtype=float)
    p = np.asarray(price, dtype=float)
    r = logret(p)
    sign = np.sign(r)
    obv = np.cumsum(sign * v)
    # z-score of obv and volume
    obv_z = zscore(obv, win)
    vol_z = zscore(v, win)
    return {"obv_z": obv_z, "vol_z": vol_z}


def zscore(x: np.ndarray, win: int = 50) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    # rolling mean/std (one‑pass Welford style)
    mean = 0.0
    m2 = 0.0
    buf = np.zeros(win)
    for t in range(len(x)):
        idx = t % win
        old = buf[idx]
        buf[idx] = x[t]
        # remove old
        if t >= win:
            # recompute mean/m2 over window (simple & stable for clarity)
            window = buf if t >= win else buf[:t + 1]
            mu = float(np.mean(window))
            sd = float(np.std(window))
        else:
            window = buf[:t + 1]
            mu = float(np.mean(window))
            sd = float(np.std(window))
        out[t] = 0.0 if sd < 1e-12 else (x[t] - mu) / sd
    return np.tanh(out / 3.0)  # squash heavy tails

# ------------------------- feature frame & scaler -------------------------------

def feature_frame(
    price: np.ndarray,
    volume: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, np.ndarray]] = None,
    *,
    rv_win: int = 30,
    rsi_win: int = 14,
    z_win: int = 50,
) -> Dict[str, np.ndarray]:
    """Assemble a dictionary of aligned 1‑D features for time t.

    Fields include:
      - r, ema_*, mom_* from price
      - rv (realized vol), rsi
      - (optional) obv_z, vol_z if volume provided
      - any arrays passed via `extra`
    """
    feats: Dict[str, np.ndarray] = {}
    feats.update(basic_price_features(price))
    feats["rv"] = realized_vol(price, rv_win)
    feats["rsi"] = rsi(price, rsi_win)
    if volume is not None:
        feats.update(volume_features(volume, price, win=z_win))
    if extra:
        for k, v in extra.items():
            feats[k] = _nan_to_num(np.asarray(v, dtype=float))
    # ensure equal length & stack
    L = len(price)
    for k, v in list(feats.items()):
        if len(v) != L:
            raise ValueError(f"feature {k} has length {len(v)} != {L}")
    return feats


def stack_features(feats: Dict[str, np.ndarray]) -> Tuple[np.ndarray, list]:
    keys = sorted(feats.keys())
    X = np.stack([feats[k] for k in keys], axis=1)
    return X.astype(float), keys


@dataclass
class StandardScaler:
    mean_: np.ndarray
    std_: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "StandardScaler":
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        return cls(mean_=mu, std_=sd)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def inverse_transform(self, Xn: np.ndarray) -> np.ndarray:
        return Xn * self.std_ + self.mean_

# ------------------------- supervised windows ----------------------------------

def make_lstm_supervised(
    X: np.ndarray,
    y: np.ndarray,
    *,
    win: int = 50,
    lookahead: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Turn frame [T, D] into sequences [N, win, D] and targets [N].

    Targets are y shifted by `lookahead` relative to the end of each window.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    T, D = X.shape
    N = T - win - lookahead + 1
    if N <= 0:
        return np.zeros((0, win, D)), np.zeros((0,))
    # sliding window view (copy for safety)
    Xseq = np.lib.stride_tricks.sliding_window_view(X, (win, D))
    Xseq = Xseq.reshape(T - win + 1, win, D)[:N].copy()
    y_out = y[win - 1 + lookahead : win - 1 + lookahead + N].copy()
    return Xseq, y_out


def split_train_val_test(T: int, ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> Tuple[slice, slice, slice]:
    a, b, c = ratios
    if abs(a + b + c - 1.0) > 1e-9:
        raise ValueError("ratios must sum to 1.0")
    i1 = int(T * a)
    i2 = int(T * (a + b))
    return slice(0, i1), slice(i1, i2), slice(i2, T)

# ------------------------- demo -------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    from data.event_stream import normalize_price_to_valuation

    T = 10_000
    rng = np.random.default_rng(0)
    price = np.cumprod(1.0 + 0.001 * rng.standard_normal(T)) * 100.0
    volume = np.exp(rng.normal(0.0, 1.0, T))
    v = normalize_price_to_valuation(price)

    feats = feature_frame(price, volume=volume, extra={"v": v})
    X, keys = stack_features(feats)
    scaler = StandardScaler.fit(X)
    Xn = scaler.transform(X)

    # supervised for forecasting valuation one step ahead
    Xseq, y = make_lstm_supervised(Xn, v, win=50, lookahead=1)
    tr, va, te = split_train_val_test(len(Xseq))
    print("Xseq:", Xseq.shape, "y:", y.shape, "splits:", tr, va, te)
    print("feature keys:", keys)
