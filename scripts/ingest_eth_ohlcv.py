"""
Ingest 4M‑row ETH 1‑minute OHLCV, convert to Parquet + numpy memmaps, and
build features lazily for LSTM/RL without blowing RAM.

Input CSV schema (header case‑insensitive):
  timestamp,open,high,low,close,volume

Artifacts produced under data/:
  • data/raw/eth_1m.parquet           — columnar store (fast reload)
  • data/cache/price.npy              — float64 close price array
  • data/cache/volume.npy             — float64 volume array
  • data/cache/features.npy           — float32 [T, D] feature frame (optional)
  • data/cache/features.shape.txt     — T D sizes for memory‑mapped loading

If you set --no-materialize, features.npy is skipped; use ml/datasets.py to
compute features on the fly per batch.

Usage
-----
python scripts/ingest_eth_ohlcv.py \
  --csv /path/to/eth_1m.csv \
  --price close \
  --materialize-features \
  --out-prefix data
"""
from __future__ import annotations    # <-- MUST BE FIRST IMPORT

import sys
import argparse
from pathlib import Path

# Add parent directory to path so we can import ml.*
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from ml.features import feature_frame, stack_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-prefix", default="data")
    ap.add_argument("--price", default="close", choices=["close", "hlc3", "ohlc4", "open"])  # price proxy
    ap.add_argument("--materialize-features", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_prefix)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)

    # 1) Load CSV (chunk‑safe)
    # For 4M rows this fits in RAM on 32GB machines; otherwise chunk and append.
    print("Reading CSV…")
    usecols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.read_csv(args.csv, usecols=usecols)

    # Normalize column case
    df.columns = [c.lower() for c in df.columns]

    # ensure dtypes
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    # choose price proxy
    if args.price == "close":
        price = df["close"].to_numpy(dtype=np.float64)
    elif args.price == "open":
        price = df["open"].to_numpy(dtype=np.float64)
    elif args.price == "hlc3":
        price = ((df["high"] + df["low"] + df["close"]) / 3.0).to_numpy(dtype=np.float64)
    else:  # ohlc4
        price = ((df["open"] + df["high"] + df["low"] + df["close"]) / 4.0).to_numpy(dtype=np.float64)

    volume = df["volume"].to_numpy(dtype=np.float64)

    # 2) Save Parquet and raw arrays
    pq_path = out_dir / "raw/eth_1m.parquet"
    print("Writing Parquet:", pq_path)
    df.to_parquet(pq_path, index=False)

    np.save(out_dir / "cache/price.npy", price)
    np.save(out_dir / "cache/volume.npy", volume)

    # 3) Optionally precompute feature frame and store as float32
    if args.materialize_features:
        print("Computing features (this may take several minutes)…")
        feats = feature_frame(price, volume=volume, extra=None)
        X, keys = stack_features(feats)
        X32 = X.astype(np.float32)
        np.save(out_dir / "cache/features.npy", X32)
        with open(out_dir / "cache/features.shape.txt", "w") as f:
            f.write(f"{X32.shape[0]} {X32.shape[1]}\n")
        with open(out_dir / "cache/features.columns.txt", "w") as f:
            f.write("\n".join(keys))
        print("Saved features with shape:", X32.shape)
    else:
        print("Skipping feature materialization (use on‑the‑fly dataset).")

    print("Done.")

if __name__ == "__main__":
    main()
