"""
LSTM forecaster for equilibrium valuation v′_p
----------------------------------------------
This module provides a lightweight PyTorch LSTM regressor to predict the
*forward valuation* (v′) from feature windows. It includes:

• LSTMPredictor nn.Module
• Dataset and DataLoader helpers
• Training loop with early stopping
• Evaluation utilities (MSE/MAE/R2)
• Save/Load helpers and a simple inference wrapper

Inputs
------
Xseq: numpy array [N, T, D] — N windows, length T, feature dim D
 y  : numpy array [N]       — target valuation in (0,1)

Notes
-----
• The model outputs a scalar passed through sigmoid to keep (0,1).
• You can swap in tanh + rescale if you change target scaling.
• Keep batch sizes small if running on CPU-only.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------- model -----------------------------------------------

class LSTMPredictor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 100, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=num_layers,
                            dropout=(dropout if num_layers > 1 else 0.0), batch_first=True)
        self.fc = nn.Linear(hidden, 1)
        self.act = nn.Sigmoid()  # ensure output in (0,1)

    def forward(self, x):  # x: [B, T, D]
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        y = self.fc(h)
        return self.act(y).squeeze(-1)  # [B]

# -------------------------- dataset ---------------------------------------------

class SeqDataset(Dataset):
    def __init__(self, Xseq: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(Xseq, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        assert self.X.ndim == 3 and self.y.ndim == 1
        assert len(self.X) == len(self.y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

# -------------------------- config ----------------------------------------------

@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    hidden: int = 100
    num_layers: int = 1
    dropout: float = 0.0
    patience: int = 6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grad_clip: Optional[float] = 1.0
    seed: int = 0

# -------------------------- training utils --------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _make_model(in_dim: int, cfg: TrainConfig) -> LSTMPredictor:
    model = LSTMPredictor(in_dim, hidden=cfg.hidden, num_layers=cfg.num_layers, dropout=cfg.dropout)
    return model


def fit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    cfg: TrainConfig = TrainConfig(),
) -> Tuple[LSTMPredictor, Dict[str, list]]:
    set_seed(cfg.seed)
    train_ds = SeqDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    has_val = X_val is not None and y_val is not None and len(X_val) > 0
    if has_val:
        val_ds = SeqDataset(X_val, y_val)
        val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = _make_model(in_dim=X_train.shape[-1], cfg=cfg).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    # Debug info
    print(f"\n{'='*60}")
    print(f"LSTM Training Started")
    print(f"{'='*60}")
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val) if has_val else 0}")
    print(f"Input shape: {X_train.shape}, Feature dim: {X_train.shape[-1]}")
    print(f"Config: hidden={cfg.hidden}, epochs={cfg.epochs}, batch_size={cfg.batch_size}")
    print(f"Device: {cfg.device}, LR: {cfg.lr}")
    print(f"{'='*60}\n")

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    patience_left = cfg.patience

    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in train_dl:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        train_loss = total / max(1, n)
        history["train_loss"].append(train_loss)

        if has_val:
            model.eval()
            tot_v = 0.0
            m = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb = xb.to(cfg.device)
                    yb = yb.to(cfg.device)
                    pred = model(xb)
                    l = loss_fn(pred, yb)
                    tot_v += float(l.item()) * len(xb)
                    m += len(xb)
            val_loss = tot_v / max(1, m)
            history["val_loss"].append(val_loss)
            improved = val_loss < best_val - 1e-8
            
            # Print progress
            status = "✓ IMPROVED" if improved else f"  (patience: {patience_left})"
            print(f"Epoch {epoch+1:3d}/{cfg.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | {status}")
            
            if improved:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_left = cfg.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
                    break
        else:
            # no val: just keep updating best
            print(f"Epoch {epoch+1:3d}/{cfg.epochs} | Train Loss: {train_loss:.6f}")
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"LSTM Training Complete")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val:.6f}" if has_val else f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Total epochs trained: {len(history['train_loss'])}")
    print(f"{'='*60}\n")
    
    return model, history

# -------------------------- evaluation ------------------------------------------

def evaluate(model: LSTMPredictor, X: np.ndarray, y: np.ndarray, device: Optional[str] = None) -> Dict[str, float]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n[LSTM Evaluate] Evaluating on {len(X)} samples...")
    
    ds = SeqDataset(X, y)
    dl = DataLoader(ds, batch_size=512, shuffle=False)
    loss_fn = nn.MSELoss()
    model = model.to(device)
    model.eval()
    yp = []
    with torch.no_grad():
        tot = 0.0
        n = 0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            tot += float(loss_fn(pred, yb).item()) * len(xb)
            n += len(xb)
            yp.append(pred.cpu().numpy())
    mse = tot / max(1, n)
    yhat = np.concatenate(yp) if yp else np.zeros_like(y)
    mae = float(np.mean(np.abs(yhat - y)))
    # R^2
    ss_res = float(np.sum((yhat - y) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    
    result = {"mse": mse, "mae": mae, "r2": r2}
    print(f"[LSTM Evaluate] MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}")
    
    return result

# -------------------------- inference -------------------------------------------

def predict(model: LSTMPredictor, X: np.ndarray, device: Optional[str] = None) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[LSTM Predict] Predicting on {len(X)} samples...")
    
    model = model.to(device)
    model.eval()
    X_t = torch.as_tensor(X, dtype=torch.float32)
    dl = DataLoader(torch.utils.data.TensorDataset(X_t), batch_size=512, shuffle=False)
    preds = []
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            pred = model(xb)
            preds.append(pred.cpu().numpy())
    
    result = np.concatenate(preds) if preds else np.zeros((0,))
    
    if len(result) > 0:
        print(f"[LSTM Predict] Complete. Predictions range: [{result.min():.4f}, {result.max():.4f}], mean: {result.mean():.4f}")
    
    return result

# -------------------------- save/load -------------------------------------------

def save_model(model: LSTMPredictor, path: str, meta: Optional[Dict] = None) -> None:
    obj = {"state_dict": model.state_dict(), "meta": (meta or {})}
    torch.save(obj, path)


def load_model(path: str, in_dim: int, hidden: int = 100, num_layers: int = 1, dropout: float = 0.0) -> Tuple[LSTMPredictor, Dict]:
    obj = torch.load(path, map_location="cpu")
    model = LSTMPredictor(in_dim=in_dim, hidden=hidden, num_layers=num_layers, dropout=dropout)
    model.load_state_dict(obj["state_dict"])  # type: ignore
    return model, obj.get("meta", {})

# -------------------------- demo -------------------------------------------------
if __name__ == "__main__":
    # quick smoke test with random data
    N, T, D = 2000, 32, 8
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, T, D)).astype(np.float32)
    y = (1 / (1 + np.exp(-X.mean(axis=(1, 2)))))  # some sigmoid target

    tr = slice(0, 1400)
    va = slice(1400, 1700)
    te = slice(1700, N)

    cfg = TrainConfig(epochs=20, batch_size=128, lr=1e-3, hidden=64)
    model, hist = fit(X[tr], y[tr], X[va], y[va], cfg)
    print("train last loss:", hist["train_loss"][-1], "val last:", (hist["val_loss"][-1] if hist["val_loss"] else None))
    print("eval:", evaluate(model, X[te], y[te]))
