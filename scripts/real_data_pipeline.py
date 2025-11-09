"""
End‑to‑end training & evaluation on REAL market data (robust + auto‑beta)
========================================================================
This script loads large 1‑minute OHLCV (millions of rows), trains an LSTM to
predict the next **valuation** v′, builds an event stream, constructs the
pseudo‑arbitrage cAMM RL environment, trains a DD‑DQN agent, and benchmarks a
UniV2‑style baseline.

What's new vs. prior version
----------------------------
• **Auto event threshold**: set `--beta_v -1` to compute β_v from a quantile of
  |Δv| via `--beta_q` (e.g., 0.8 ⇒ β_v = Q80(|Δv|)). If eventization yields too
  few events, the script automatically relaxes β_v (q ↓) until it gets a
  reasonable count.
• Tight shape checks (always build [T,D] → [N,win,D]).
• RL window seeds from the **last** normalized feature slice.

CSV schema (header case‑insensitive)
------------------------------------
  timestamp, open, high, low, close, volume

Example
-------
python scripts/real_data_pipeline.py \
  --csv predictive_amm/data/raw/eth_1m.csv \
  --beta_v -1 --beta_q 0.80 \
  --lstm_win 50 --lookahead 1 \
  --epochs 20 \
  --rl_steps 80000
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
# Local imports from this repo
from data.event_stream import (
    normalize_price_to_valuation,
    make_event_dataset,
)
from ml.features import (
    feature_frame,
    stack_features,
    StandardScaler,
    make_lstm_supervised,
    split_train_val_test,
)
from ml.lstm import TrainConfig as LSTMTrainConfig, fit as lstm_fit, evaluate as lstm_evaluate, predict as lstm_predict
from ml.rl_env import RLEnv, RLEnvConfig
from ml.dqn import DDQNConfig, DDQNAgent, train as dqn_train
from amm.camm import ConfigurableAMM
from sim.synth import make_trade_stream
from sim.baseline import UniV2LikePool, run_baseline
from sim.eval import summarize_baseline_run, path_losses

# ------------------------------ IO helpers --------------------------------------

def load_ohlcv_csv(path: str, price_proxy: str = "close") -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(path)
    # normalize columns
    df.columns = [c.lower() for c in df.columns]
    req = ["timestamp", "open", "high", "low", "close", "volume"]
    for k in req:
        if k not in df.columns:
            raise ValueError(f"CSV must include column '{k}' (case‑insensitive)")
    # ensure dtypes
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    if price_proxy == "close":
        price = df["close"].to_numpy(dtype=np.float64)
    elif price_proxy == "open":
        price = df["open"].to_numpy(dtype=np.float64)
    elif price_proxy == "hlc3":
        price = ((df["high"] + df["low"] + df["close"]) / 3.0).to_numpy(dtype=np.float64)
    elif price_proxy == "ohlc4":
        price = ((df["open"] + df["high"] + df["low"] + df["close"]) / 4.0).to_numpy(dtype=np.float64)
    else:
        raise ValueError("Unknown price proxy")

    volume = df["volume"].to_numpy(dtype=np.float64)
    return price, volume, df

# ------------------------------ beta helpers ------------------------------------

def choose_beta_v(val: np.ndarray, beta_v: float, beta_q: float, *, min_events: int = 10) -> float:
    """Return a β_v threshold. If beta_v >= 0, use it.
    If beta_v < 0, compute β_v = quantile(|Δv|, beta_q) and relax if events are too few.
    """
    v = np.asarray(val, dtype=float)
    dv = np.abs(np.diff(v))
    dv = dv[np.isfinite(dv)]
    if dv.size == 0:
        return 1e-9
    if beta_v >= 0:
        return float(beta_v)
    # quantile‑based initial guess
    q = float(np.clip(beta_q, 0.0, 0.999))
    guess = float(np.quantile(dv, q))
    # Never zero (guard for flat series)
    guess = max(guess, 1e-9)
    return guess

# ------------------------------ main --------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to 1m OHLCV CSV")
    ap.add_argument("--beta_v", type=float, default=-1.0, help="Event threshold on |Δv|; set <0 to auto")
    ap.add_argument("--beta_q", type=float, default=0.80, help="If --beta_v<0, use this quantile of |Δv| as threshold")
    ap.add_argument("--lstm_win", type=int, default=50)
    ap.add_argument("--lookahead", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--rl_steps", type=int, default=80_000)
    ap.add_argument("--fee_bps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--price_proxy", choices=["close", "open", "hlc3", "ohlc4"], default="close")
    args = ap.parse_args()

    np.random.seed(args.seed)

    # 1) Load real OHLCV
    price, volume, df = load_ohlcv_csv(args.csv, price_proxy=args.price_proxy)
    print(f"loaded {len(price)} rows from {args.csv}")

    # 2) Build valuation and 2‑D feature frame [T,D]
    val = normalize_price_to_valuation(price)  # 1‑D in (0,1)
    feats = feature_frame(price, volume=volume, extra={"valuation": val})  # dict of 1‑D arrays
    X, keys = stack_features(feats)  # [T,D]
    if X.ndim != 2:
        raise RuntimeError(f"feature_frame/stack_features must return 2‑D, got shape {X.shape}")
    scaler = StandardScaler.fit(X)
    Xn = scaler.transform(X)  # [T,D]

    # 3) Supervised sequences for LSTM: Xseq [N,win,D], y [N]
    Xseq, y = make_lstm_supervised(Xn, val, win=args.lstm_win, lookahead=args.lookahead)
    tr, va, te = split_train_val_test(len(Xseq))
    print("LSTM dataset:", Xseq.shape, "splits:", tr, va, te)

    # 4) Train LSTM
    cfg = LSTMTrainConfig(epochs=args.epochs, batch_size=256, lr=1e-3, hidden=128, seed=args.seed)
    model, hist = lstm_fit(Xseq[tr], y[tr], Xseq[va], y[va], cfg)
    print("LSTM eval (test):", lstm_evaluate(model, Xseq[te], y[te]))

    # 5) Eventize the valuation path for RL with auto‑beta if requested
    initial_beta = choose_beta_v(val, args.beta_v, args.beta_q)
    beta_try = [initial_beta]
    # backup relaxations (lower thresholds ⇒ more events)
    for q in (0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05):
        beta_try.append(float(np.quantile(np.abs(np.diff(val)), q)))
    beta_try.append(1e-9)

    events = []
    chosen_beta = None
    for b in beta_try:
        b = max(float(b), 1e-9)
        ds = make_event_dataset(val, tau=np.zeros((len(val), 1)), beta_v=b, lstm_win=args.lstm_win, lookahead=args.lookahead)
        events = ds["events"]
        if len(events) >= 2:
            chosen_beta = b
            break
    if chosen_beta is None:
        raise RuntimeError("Eventization failed to produce >=2 events even with very small β_v. Check data.")
    print(f"Eventization: β_v={chosen_beta:.6g}, events={len(events)}")

    # 6) Predictor wrapper for RL env, using the **last** normalized feature window
    if Xn.shape[0] < args.lstm_win:
        raise RuntimeError("Not enough rows to build the initial LSTM window for RL")
    window_init = Xn[-args.lstm_win :, :]  # [win,D]

    def lstm_pred_fn(win_np: np.ndarray) -> float:
        if win_np.ndim != 2:
            raise ValueError(f"lstm_pred_fn expects [T,D] window, got {win_np.shape}")
        w = win_np[None, :, :].astype(np.float32)      # [1,T,D]
        out = lstm_predict(model, w)                   # [1]
        return float(np.clip(out[0], 1e-6, 1.0 - 1e-6))

    # 7) Create cAMM and RL environment
    p0 = float(price[0])
    y0 = 100_000.0
    x0 = y0 / max(1e-12, p0)
    camm = ConfigurableAMM(x=x0, y=y0, name="cAMM_real")

    env = RLEnv(
        RLEnvConfig(beta_c=0.02, sigma_noise=0.02, samples_per_step=16, lstm_win=args.lstm_win, seed=args.seed),
        events,
        camm,
        lstm_pred_fn,
        window_init=window_init,
    )

    # 8) Train DD‑DQN agent
    agent = DDQNAgent(state_dim=5, n_actions=2, cfg=DDQNConfig(buffer_size=100_000, min_buffer=2_000, eps_decay_steps=25_000, seed=args.seed))
    hist = dqn_train(env, agent, steps=args.rl_steps, warm_start_random=5_000)
    print(f"RL training done. steps={len(hist['reward'])}, mean reward={np.mean(hist['reward']):.4f}")

    # 9) Baseline simulation on the same path
    trades = make_trade_stream(price, seed=args.seed + 7)
    base_pool = UniV2LikePool(x=x0, y=y0, fee_bps=args.fee_bps)
    base_res = run_baseline(base_pool, price, trades)
    base_sum = summarize_baseline_run(base_res, price)

    # 10) Path‑only CPMM losses for information
    losses = path_losses(price, c=float(x0 * y0))

    print("=== SUMMARY ===")
    print("Baseline:", base_sum)
    print("Path losses (CPMM invariant, price path only):", losses)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("ERROR:", e)
        sys.exit(1)
