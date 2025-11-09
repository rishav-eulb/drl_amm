"""
End‑to‑end training & evaluation on REAL market data
===================================================
This script wires up the codebase you built to:
  1) Ingest a 1‑minute CSV of price/volume
  2) Build features & LSTM windows, train LSTM to predict next valuation v′
  3) Eventize the valuation path (β_v) and create the RL environment
  4) Train a DD‑DQN agent to control cAMM (exploration via feature‑noise)
  5) Run a UniV2‑style baseline vs predictive cAMM and print eval metrics

Expected CSV format (header case‑insensitive):
  timestamp, price, volume

Usage
-----
python scripts/real_data_pipeline.py \
  --csv data/raw/your_asset_1m.csv \
  --beta_v 0.005 \
  --lstm_win 50 --lookahead 1 \
  --epochs 25 \
  --rl_steps 50000 \
  --seed 42

Notes
-----
• Keep CUDA visible if available; else it will run on CPU.
• You can safely adjust β_v to control the number of RL steps (more events = more steps).
• For quick tests, reduce --rl_steps.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

# local imports
from data.event_stream import (
    normalize_price_to_valuation,
    make_event_dataset,
)
from ml.features import feature_frame, stack_features, StandardScaler, make_lstm_supervised, split_train_val_test
from ml.lstm import TrainConfig as LSTMTrainConfig, fit as lstm_fit, evaluate as lstm_evaluate, predict as lstm_predict
from ml.rl_env import RLEnv, RLEnvConfig
from ml.dqn import DDQNConfig, DDQNAgent, train as dqn_train
from amm.camm import ConfigurableAMM
from sim.synth import make_trade_stream
from sim.baseline import UniV2LikePool, run_baseline
from sim.eval import summarize_baseline_run, path_losses

import pandas as pd

# ------------------------------ helpers -----------------------------------------

def load_csv_1m(path: str):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    price_col = cols.get("price") or cols.get("close") or list(df.columns)[1]
    volume_col = cols.get("volume") or list(df.columns)[2]
    p = df[price_col].astype(float).to_numpy()
    v = df[volume_col].astype(float).to_numpy()
    return p, v, df

# ------------------------------ main --------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to 1m CSV with price,volume")
    ap.add_argument("--beta_v", type=float, default=0.005, help="Event threshold on |Δvaluation|")
    ap.add_argument("--lstm_win", type=int, default=50)
    ap.add_argument("--lookahead", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rl_steps", type=int, default=50_000)
    ap.add_argument("--fee_bps", type=int, default=30)
    args = ap.parse_args()

    np.random.seed(args.seed)

    # 1) Load real data
    price, volume, df = load_csv_1m(args.csv)
    print(f"loaded {len(price)} rows from {args.csv}")

    # 2) Build valuation and features
    val = normalize_price_to_valuation(price)
    feats = feature_frame(price, volume=volume, extra={"valuation": val})
    X, keys = stack_features(feats)
    scaler = StandardScaler.fit(X)
    Xn = scaler.transform(X)

    # 3) Supervised sequences for LSTM (predict next valuation)
    Xseq, y = make_lstm_supervised(Xn, val, win=args.lstm_win, lookahead=args.lookahead)
    tr, va, te = split_train_val_test(len(Xseq))
    print("LSTM dataset:", Xseq.shape, "splits:", tr, va, te)

    # 4) Train LSTM
    cfg = LSTMTrainConfig(epochs=args.epochs, batch_size=256, lr=1e-3, hidden=128, seed=args.seed)
    model, hist = lstm_fit(Xseq[tr], y[tr], Xseq[va], y[va], cfg)
    print("LSTM eval (test):", lstm_evaluate(model, Xseq[te], y[te]))

    # 5) Eventize full valuation path for RL
    ds = make_event_dataset(val, tau=np.zeros_like(val), beta_v=args.beta_v, lstm_win=args.lstm_win, lookahead=args.lookahead)
    events = ds["events"]

    # 6) Predictor wrapper for RL env (use last window; here we reuse our normalized features)
    #    Build an initial window using the *feature frame* (Xn) trailing slice
    assert Xn.shape[0] >= args.lstm_win
    window_init = Xn[: args.lstm_win, :]

    def lstm_pred_fn(win_np: np.ndarray) -> float:
        # win_np shape [T,D]; model expects [B,T,D]
        w = win_np[None, :, :].astype(np.float32)
        out = lstm_predict(model, w)
        return float(np.clip(out[0], 1e-6, 1 - 1e-6))

    # 7) Create cAMM and RL environment
    #    Initialize pool roughly balanced at the CSV's start price
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
    agent = DDQNAgent(state_dim=5, n_actions=2, cfg=DDQNConfig(buffer_size=100_000, min_buffer=2000, eps_decay_steps=25_000, seed=args.seed))
    hist = dqn_train(env, agent, steps=args.rl_steps, warm_start_random=5_000)
    print(f"RL training done. steps={len(hist['reward'])}, mean reward={np.mean(hist['reward']):.4f}")

    # 9) Baseline simulation (for comparison)
    #    Make a taker trade stream from the real price path (statistically tied to returns)
    trades = make_trade_stream(price, seed=args.seed + 7)
    base_pool = UniV2LikePool(x=x0, y=y0, fee_bps=args.fee_bps)
    base_res = run_baseline(base_pool, price, trades)
    base_sum = summarize_baseline_run(base_res, price)

    # 10) Predictive path losses (for info)
    losses = path_losses(price, c=float(x0 * y0))

    print("\n=== SUMMARY ===")
    print("Baseline:", base_sum)
    print("Path losses (CPMM invariant, price path only):", losses)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
