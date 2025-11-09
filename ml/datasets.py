"""
Memory‑efficient datasets for 4M×1m OHLCV
----------------------------------------
Provides lazy sequence dataset and streaming windows to avoid materializing
[N, T, D] tensors in RAM.

Key classes
-----------
• RollingWindowDataset — wraps a [T, D] feature frame and target vector y[T]
  and yields windows [win, D] with lookahead target, on the fly.

• MemmapFeatureFrame — loads features from .npy (or Parquet) as memory‑mapped
  arrays; can also compute features from OHLCV arrays per batch if no materialized
  features exist.

Usage
-----
from ml.datasets import MemmapFeatureFrame, RollingWindowDataset
from torch.utils.data import DataLoader

mm = MemmapFeatureFrame(base_dir="data/cache")
X = mm.features()      # memmap view [T, D]
y = mm.valuation()     # valuation in (0,1)

ds = RollingWindowDataset(X, y, win=50, lookahead=1)
dl = DataLoader(ds, batch_size=256, shuffle=True)

Pass `dl` to ml.lstm.fit by using its X_train, y_train expectations or adapt the
training loop to consume the loader directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from data.event_stream import normalize_price_to_valuation
from ml.features import feature_frame, stack_features

@dataclass
class MemmapFeatureFrame:
    base_dir: str = "data/cache"

    def _load_npy(self, name: str, mmap: bool = True) -> np.ndarray:
        path = Path(self.base_dir) / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        mode = "r" if mmap else None
        return np.load(path, mmap_mode=mode)

    def price(self) -> np.ndarray:
        return self._load_npy("price")

    def volume(self) -> np.ndarray:
        return self._load_npy("volume")

    def valuation(self) -> np.ndarray:
        return normalize_price_to_valuation(np.asarray(self.price()))

    def features(self) -> np.ndarray:
        path = Path(self.base_dir) / "features.npy"
        if path.exists():
            return np.load(path, mmap_mode="r")
        # compute on the fly (returns regular ndarray; for huge T consider chunking)
        p = np.asarray(self.price())
        v = np.asarray(self.volume())
        feats = feature_frame(p, volume=v)
        X, _ = stack_features(feats)
        return X.astype(np.float32)

class RollingWindowDataset:
    def __init__(self, X: np.ndarray, y: np.ndarray, *, win: int = 50, lookahead: int = 1):
        X = np.asarray(X)
        y = np.asarray(y)
        self.X = X
        self.y = y
        self.win = int(win)
        self.look = int(lookahead)
        self.N = X.shape[0] - win - lookahead + 1
        if self.N <= 0:
            raise ValueError("window too long for given series")

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        i = int(i)
        sl_X = slice(i, i + self.win)
        sl_y = i + self.win - 1 + self.look
        return self.X[sl_X], self.y[sl_y]

    # optional: torch Dataset wrapper when torch is available
    def as_torch(self):
        import torch
        from torch.utils.data import Dataset
        class _DS(Dataset):
            def __init__(self, parent):
                self.p = parent
            def __len__(self):
                return self.p.N
            def __getitem__(self, idx):
                Xw, yv = self.p[idx]
                return torch.as_tensor(Xw, dtype=torch.float32), torch.as_tensor(yv, dtype=torch.float32)
        return _DS(self)

# convenience split that returns index arrays (no copies)

def split_indices(T: int, ratios=(0.7, 0.15, 0.15)):
    a, b, c = ratios
    i1 = int(T * a)
    i2 = int(T * (a + b))
    return np.arange(0, i1), np.arange(i1, i2), np.arange(i2, T)
