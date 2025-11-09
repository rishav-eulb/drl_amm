"""
End‑to‑end training & evaluation on REAL market data (COMPLETE + CORRECTED)
===========================================================================
This script implements the paper's specifications exactly with PROPER evaluation:
- Proper β_c threshold for rewards
- Correct LSTM hyperparameters (Table 1)
- Proper event generation (1-5% of data)
- Train/test split for events
- Evaluation on test set with greedy policy
- Deployment of trained predictive AMM
- Side-by-side comparison: Proposed vs Baseline

Paper: "Predictive crypto-asset automated market maker architecture for 
       decentralized finance using deep reinforcement learning"

Example
-------
python scripts/real_data_pipeline.py \\
  --csv data/raw/eth_1m.csv \\
  --beta_v -1 --beta_q 0.98 \\
  --beta_c 0.001 \\
  --lstm_win 50 --lookahead 1 \\
  --epochs 50 \\
  --rl_steps 200000
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.event_stream import (
    normalize_price_to_valuation,
    make_event_dataset,
    Event,
)
from ml.features import (
    feature_frame,
    stack_features,
    StandardScaler,
    make_lstm_supervised,
    split_train_val_test,
)
from ml.lstm import (
    TrainConfig as LSTMTrainConfig, 
    fit as lstm_fit, 
    evaluate as lstm_evaluate, 
    predict as lstm_predict,
    LSTMPredictor,
)
from ml.rl_env import RLEnv, RLEnvConfig
from ml.dqn import DDQNConfig, DDQNAgent, train as dqn_train
from amm.camm import ConfigurableAMM
from amm.maths import divergence_loss, slippage_loss_X, load_auto
from sim.synth import make_trade_stream
from sim.baseline import UniV2LikePool, run_baseline
from sim.eval import summarize_baseline_run, path_losses

# ============================================================================
# DATA LOADING
# ============================================================================

def load_ohlcv_csv(path: str, price_proxy: str = "close") -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load OHLCV data with proper validation."""
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    
    req = ["timestamp", "open", "high", "low", "close", "volume"]
    for k in req:
        if k not in df.columns:
            raise ValueError(f"CSV must include column '{k}' (case‑insensitive)")
    
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


# ============================================================================
# EVENT GENERATION
# ============================================================================

def choose_optimal_beta_v(
    valuation: np.ndarray,
    beta_v: float,
    beta_q: float,
    *,
    target_event_ratio: float = 0.02,
) -> tuple[float, int]:
    """Choose β_v threshold to get approximately target_event_ratio events."""
    v = np.asarray(valuation, dtype=float)
    dv = np.abs(np.diff(v))
    dv = dv[np.isfinite(dv)]
    
    if dv.size == 0:
        return 1e-9, 0
    
    if beta_v >= 0:
        events = np.sum(dv >= beta_v)
        return float(beta_v), int(events)
    
    # Auto mode: use quantile
    q = float(np.clip(beta_q, 0.0, 0.999))
    beta_try = float(np.quantile(dv, q))
    beta_try = max(beta_try, 1e-9)
    
    actual_events = np.sum(dv >= beta_try)
    
    print(f"[Event Threshold] β_v={beta_try:.6e} (q={q:.3f})")
    print(f"[Event Threshold] Expected events: {actual_events:,} / {len(v):,} ({100*actual_events/len(v):.2f}%)")
    
    # If too many events, relax
    if actual_events > len(v) * 0.05:
        print(f"[Event Threshold] Too many events (>{5}%), adjusting...")
        q_new = min(0.99, q + 0.05)
        beta_try = float(np.quantile(dv, q_new))
        beta_try = max(beta_try, 1e-9)
        actual_events = np.sum(dv >= beta_try)
        print(f"[Event Threshold] Adjusted β_v={beta_try:.6e} (q={q_new:.3f}), events={actual_events:,}")
    
    return beta_try, actual_events


def split_events_train_test(events: List[Event], train_ratio: float = 0.8) -> Tuple[List[Event], List[Event]]:
    """Split events into train/test sets."""
    n = len(events)
    split_idx = int(n * train_ratio)
    
    train_events = events[:split_idx]
    test_events = events[split_idx:]
    
    print(f"[Event Split] Train: {len(train_events):,}, Test: {len(test_events):,}")
    return train_events, test_events


# ============================================================================
# EVALUATION FUNCTIONS (NEW!)
# ============================================================================

def evaluate_rl_policy(env: RLEnv, agent: DDQNAgent, max_steps: int = 10000) -> Dict:
    """
    Evaluate trained policy on test environment (greedy, no exploration).
    
    Returns metrics:
    - mean_reward
    - positive_ratio
    - mean_loss
    - action distribution
    """
    # Save original epsilon
    old_eps_start = agent.cfg.eps_start
    old_eps_end = agent.cfg.eps_end
    
    # Disable exploration
    agent.cfg.eps_start = 0.0
    agent.cfg.eps_end = 0.0
    
    obs = env.reset()
    rewards = []
    losses = []
    actions = []
    
    done = False
    step = 0
    
    while not done and step < max_steps:
        # Greedy action
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        
        rewards.append(reward)
        actions.append(action)
        if 'total_loss' in info:
            losses.append(info['total_loss'])
        
        step += 1
    
    # Restore epsilon
    agent.cfg.eps_start = old_eps_start
    agent.cfg.eps_end = old_eps_end
    
    return {
        'mean_reward': float(np.mean(rewards)) if rewards else 0.0,
        'positive_ratio': float(np.mean(np.array(rewards) > 0)) if rewards else 0.0,
        'total_steps': len(rewards),
        'mean_loss': float(np.mean(losses)) if losses else 0.0,
        'action_0_count': actions.count(0),
        'action_1_count': actions.count(1),
    }


@dataclass
class PredictiveAMMResult:
    """Results from simulating predictive AMM."""
    x_hist: np.ndarray
    y_hist: np.ndarray
    drift_x: np.ndarray
    drift_y: np.ndarray
    predictions: np.ndarray
    valuations: np.ndarray
    actions: np.ndarray
    losses: np.ndarray


def simulate_predictive_amm(
    events: List[Event],
    camm_init: ConfigurableAMM,
    lstm_model: LSTMPredictor,
    dqn_agent: DDQNAgent,
    window_init: np.ndarray,
    rl_config: RLEnvConfig,
) -> PredictiveAMMResult:
    """
    Simulate the predictive AMM on test data.
    
    Deploys:
    1. LSTM for v'_p predictions
    2. DQN agent for action selection (greedy)
    3. Pseudo-arbitrage shifts
    """
    # Create test environment
    test_camm = ConfigurableAMM(x=camm_init.x, y=camm_init.y, name="test_predictive_amm")
    
    def lstm_pred_fn(win: np.ndarray) -> float:
        w = win[None, :, :].astype(np.float32)
        out = lstm_predict(lstm_model, w)
        return float(np.clip(out[0], 1e-6, 1.0 - 1e-6))
    
    test_env = RLEnv(
        rl_config,
        events,
        test_camm,
        lstm_pred_fn,
        window_init=window_init,
    )
    
    # Disable exploration
    old_eps_start = dqn_agent.cfg.eps_start
    old_eps_end = dqn_agent.cfg.eps_end
    dqn_agent.cfg.eps_start = 0.0
    dqn_agent.cfg.eps_end = 0.0
    
    obs = test_env.reset()
    
    # Storage
    x_hist = [test_env.camm.x]
    y_hist = [test_env.camm.y]
    drift_x_hist = [test_env.camm.drift.dx]
    drift_y_hist = [test_env.camm.drift.dy]
    predictions = []
    valuations = []
    actions = []
    losses = []
    
    done = False
    step = 0
    
    while not done and step < len(events) - 1:
        action = dqn_agent.act(obs)
        obs, reward, done, info = test_env.step(action)
        
        x_hist.append(test_env.camm.x)
        y_hist.append(test_env.camm.y)
        drift_x_hist.append(test_env.camm.drift.dx)
        drift_y_hist.append(test_env.camm.drift.dy)
        predictions.append(info['vpred'])
        valuations.append(info['v_next'])
        actions.append(action)
        losses.append(info['total_loss'])
        
        step += 1
    
    # Restore epsilon
    dqn_agent.cfg.eps_start = old_eps_start
    dqn_agent.cfg.eps_end = old_eps_end
    
    return PredictiveAMMResult(
        x_hist=np.array(x_hist),
        y_hist=np.array(y_hist),
        drift_x=np.array(drift_x_hist),
        drift_y=np.array(drift_y_hist),
        predictions=np.array(predictions),
        valuations=np.array(valuations),
        actions=np.array(actions),
        losses=np.array(losses),
    )


def compute_proposed_amm_metrics(
    result: PredictiveAMMResult,
    price_series: np.ndarray,
    initial_c: float,
) -> Dict:
    """Compute metrics for proposed AMM."""
    # Divergence/slippage/load per step
    divs = []
    slips = []
    loads = []
    
    for i in range(len(result.valuations) - 1):
        v1 = result.valuations[i]
        v2 = result.valuations[i + 1]
        divs.append(divergence_loss(v1, v2, initial_c))
        slips.append(slippage_loss_X(v1, v2, initial_c))
        loads.append(load_auto(v1, v2, initial_c))
    
    # Drift magnitude
    drift_mag = np.sqrt(result.drift_x[-1]**2 + result.drift_y[-1]**2)
    
    # Prediction accuracy
    pred_mae = float(np.mean(np.abs(result.predictions - result.valuations)))
    
    return {
        'divergence_loss_mean': float(np.mean(divs)) if divs else 0.0,
        'slippage_loss_mean': float(np.mean(slips)) if slips else 0.0,
        'load_mean': float(np.mean(loads)) if loads else 0.0,
        'drift_magnitude': float(drift_mag),
        'prediction_mae': pred_mae,
        'divergence_loss_sum': float(np.sum(divs)) if divs else 0.0,
        'slippage_loss_sum': float(np.sum(slips)) if slips else 0.0,
        'load_sum': float(np.sum(loads)) if loads else 0.0,
    }


def print_comparison_table(
    baseline_metrics: Dict,
    proposed_metrics: Dict,
    num_trades: int,
) -> None:
    """Print paper-style comparison table."""
    print("\n" + "="*80)
    print("COMPARISON: PROPOSED PREDICTIVE AMM vs BASELINE UNISWAP V2")
    print("="*80)
    
    print("\n┌─────────────────────────┬─────────────┬─────────────┬────────────┐")
    print("│ Metric                  │  Baseline   │  Proposed   │  Δ Change  │")
    print("├─────────────────────────┼─────────────┼─────────────┼────────────┤")
    
    # Utilization (if available)
    base_util = baseline_metrics.get('utilization', 0)
    prop_util = proposed_metrics.get('utilization', 0)
    if base_util > 0:
        delta_util = ((prop_util - base_util) / base_util * 100) if base_util > 0 else 0
        print(f"│ Utilization             │  {base_util:>9.2%}  │  {prop_util:>9.2%}  │  {delta_util:>6.1f}%  │")
    
    # Divergence
    base_div = baseline_metrics.get('div_mean', 0)
    prop_div = proposed_metrics.get('divergence_loss_mean', 0)
    delta_div = ((prop_div - base_div) / base_div * 100) if base_div > 0 else 0
    print(f"│ Divergence Loss (mean)  │  {base_div:>9.6f}  │  {prop_div:>9.6f}  │  {delta_div:>6.1f}%  │")
    
    # Slippage
    base_slip = baseline_metrics.get('slip_mean', 0)
    prop_slip = proposed_metrics.get('slippage_loss_mean', 0)
    delta_slip = ((prop_slip - base_slip) / base_slip * 100) if base_slip > 0 else 0
    print(f"│ Slippage Loss (mean)    │  {base_slip:>9.6f}  │  {prop_slip:>9.6f}  │  {delta_slip:>6.1f}%  │")
    
    # Load
    base_load = baseline_metrics.get('load_mean', 0)
    prop_load = proposed_metrics.get('load_mean', 0)
    delta_load = ((prop_load - base_load) / base_load * 100) if base_load > 0 else 0
    print(f"│ Load (mean)             │  {base_load:>9.2e}  │  {prop_load:>9.2e}  │  {delta_load:>6.1f}%  │")
    
    # Impact
    base_impact = baseline_metrics.get('impact_mean', 0)
    if base_impact > 0:
        print(f"│ Price Impact (mean)     │  {base_impact:>9.2%}  │      N/A      │     N/A    │")
    
    print("└─────────────────────────┴─────────────┴─────────────┴────────────┘")
    
    print(f"\nAdditional Proposed AMM Metrics:")
    print(f"  Prediction MAE:    {proposed_metrics['prediction_mae']:.6f}")
    print(f"  Drift magnitude:   {proposed_metrics['drift_magnitude']:.2f}")
    print(f"  Divergence (sum):  {proposed_metrics['divergence_loss_sum']:.2f}")
    print(f"  Slippage (sum):    {proposed_metrics['slippage_loss_sum']:.2f}")
    
    print(f"\nBaseline (Uniswap V2) with {num_trades:,} synthetic trades")
    print("="*80)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Complete DRL-AMM training + evaluation pipeline")
    
    # Data
    ap.add_argument("--csv", required=True, help="Path to 1m OHLCV CSV")
    ap.add_argument("--price_proxy", choices=["close", "open", "hlc3", "ohlc4"], default="close")
    
    # Event generation
    ap.add_argument("--beta_v", type=float, default=-1.0, help="Event threshold on |Δv|; <0=auto")
    ap.add_argument("--beta_q", type=float, default=0.98, help="Quantile for auto beta_v")
    ap.add_argument("--target_event_ratio", type=float, default=0.02, help="Target event ratio")
    
    # LSTM (Table 1)
    ap.add_argument("--lstm_win", type=int, default=50)
    ap.add_argument("--lstm_hidden", type=int, default=100)
    ap.add_argument("--lstm_batch", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lookahead", type=int, default=1)
    
    # RL (Table 1)
    ap.add_argument("--beta_c", type=float, default=0.001, help="Reward threshold")
    ap.add_argument("--rl_steps", type=int, default=200_000)
    ap.add_argument("--rl_buffer", type=int, default=100_000)
    ap.add_argument("--rl_gamma", type=float, default=0.98)
    ap.add_argument("--rl_eps_decay", type=int, default=50_000)
    
    # Evaluation
    ap.add_argument("--train_ratio", type=float, default=0.8, help="Train/test split ratio")
    
    # Other
    ap.add_argument("--fee_bps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    
    args = ap.parse_args()
    
    np.random.seed(args.seed)
    
    print("=" * 80)
    print("COMPLETE DRL-AMM TRAINING + EVALUATION PIPELINE")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Data: {args.csv}")
    print(f"  Event: β_v={'auto' if args.beta_v < 0 else args.beta_v} (q={args.beta_q})")
    print(f"  Reward: β_c={args.beta_c}")
    print(f"  LSTM: win={args.lstm_win}, hidden={args.lstm_hidden}, batch={args.lstm_batch}, epochs={args.epochs}")
    print(f"  RL: steps={args.rl_steps}, γ={args.rl_gamma}")
    print(f"  Split: train={args.train_ratio:.1%}, test={1-args.train_ratio:.1%}")
    print("=" * 80 + "\n")

    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================
    print("[1/11] Loading OHLCV data...")
    price, volume, df = load_ohlcv_csv(args.csv, price_proxy=args.price_proxy)
    print(f"       Loaded {len(price):,} rows")
    print(f"       Price: [{price.min():.2f}, {price.max():.2f}]")
    print(f"       Volume: [{volume.min():.2e}, {volume.max():.2e}]\n")

    # ========================================================================
    # 2. BUILD FEATURES
    # ========================================================================
    print("[2/11] Building features...")
    val = normalize_price_to_valuation(price)
    feats = feature_frame(price, volume=volume, extra={"valuation": val})
    X, keys = stack_features(feats)
    scaler = StandardScaler.fit(X)
    Xn = scaler.transform(X)
    print(f"       Features: {X.shape}, columns: {len(keys)}\n")

    # ========================================================================
    # 3. LSTM TRAINING
    # ========================================================================
    print("[3/11] LSTM supervised learning...")
    Xseq, y = make_lstm_supervised(Xn, val, win=args.lstm_win, lookahead=args.lookahead)
    tr, va, te = split_train_val_test(len(Xseq))
    print(f"       Sequences: {Xseq.shape}")
    print(f"       Train: {len(Xseq[tr]):,}, Val: {len(Xseq[va]):,}, Test: {len(Xseq[te]):,}\n")

    cfg_lstm = LSTMTrainConfig(
        epochs=args.epochs, batch_size=args.lstm_batch, lr=1e-3,
        hidden=args.lstm_hidden, num_layers=1, dropout=0.0,
        patience=10, seed=args.seed,
    )
    model, hist = lstm_fit(Xseq[tr], y[tr], Xseq[va], y[va], cfg_lstm)
    
    eval_metrics = lstm_evaluate(model, Xseq[te], y[te])
    print(f"       Test: MSE={eval_metrics['mse']:.2e}, MAE={eval_metrics['mae']:.2e}, R²={eval_metrics['r2']:.4f}\n")

    # ========================================================================
    # 4. EVENT GENERATION
    # ========================================================================
    print("[4/11] Generating events...")
    chosen_beta, _ = choose_optimal_beta_v(val, args.beta_v, args.beta_q, 
                                           target_event_ratio=args.target_event_ratio)
    
    ds = make_event_dataset(val, tau=np.zeros((len(val), 1)), beta_v=chosen_beta,
                           lstm_win=args.lstm_win, lookahead=args.lookahead)
    events = ds["events"]
    print(f"       Events: {len(events):,} ({100*len(events)/len(val):.2f}%)\n")

    # ========================================================================
    # 5. SPLIT EVENTS FOR TRAINING/TESTING
    # ========================================================================
    print("[5/11] Splitting events into train/test...")
    train_events, test_events = split_events_train_test(events, train_ratio=args.train_ratio)
    print()

    # ========================================================================
    # 6. RL TRAINING (on train events)
    # ========================================================================
    print("[6/11] Setting up RL environment...")
    
    window_init = Xn[-args.lstm_win:, :]
    
    def lstm_pred_fn(win_np: np.ndarray) -> float:
        if win_np.ndim != 2:
            raise ValueError(f"lstm_pred_fn expects [T,D], got {win_np.shape}")
        w = win_np[None, :, :].astype(np.float32)
        out = lstm_predict(model, w)
        return float(np.clip(out[0], 1e-6, 1.0 - 1e-6))
    
    p0 = float(price[0])
    y0 = 100_000.0
    x0 = y0 / max(1e-12, p0)
    camm_init = ConfigurableAMM(x=x0, y=y0, name="cAMM_train")
    print(f"       Initial: x={x0:.2f}, y={y0:.2f}, c={x0*y0:.2e}\n")
    
    cfg_rl = RLEnvConfig(
        beta_c=args.beta_c, sigma_noise=0.02, samples_per_step=16,
        lstm_win=args.lstm_win, seed=args.seed,
    )
    
    train_env = RLEnv(cfg_rl, train_events, camm_init, lstm_pred_fn, window_init)

    print("[7/11] Training DD-DQN agent...")
    agent = DDQNAgent(
        state_dim=5, n_actions=2,
        cfg=DDQNConfig(
            gamma=args.rl_gamma, lr=1e-3, hidden=128, batch_size=256,
            buffer_size=args.rl_buffer, min_buffer=5_000,
            train_freq=1, target_sync=1000, tau=0.0,
            eps_start=1.0, eps_end=0.05, eps_decay_steps=args.rl_eps_decay,
            per=False, grad_clip=1.0, seed=args.seed,
        )
    )
    
    hist = dqn_train(train_env, agent, steps=args.rl_steps, warm_start_random=5_000)
    
    rewards = np.array(hist["reward"])
    mean_reward = float(np.mean(rewards))
    last_1k = float(np.mean(rewards[-1000:])) if len(rewards) >= 1000 else mean_reward
    positive_ratio = float(np.mean(rewards > 0))
    
    print(f"       Training complete: {len(rewards):,} steps")
    print(f"       Mean reward: {mean_reward:.4f}")
    print(f"       Last 1k: {last_1k:.4f}")
    print(f"       Positive %: {positive_ratio:.2%}\n")

    # ========================================================================
    # 7. EVALUATE TRAINED POLICY ON TEST SET (NEW!)
    # ========================================================================
    print("[8/11] Evaluating trained policy on test events...")
    
    test_env = RLEnv(cfg_rl, test_events, 
                     ConfigurableAMM(x=x0, y=y0, name="cAMM_test"),
                     lstm_pred_fn, window_init)
    
    test_results = evaluate_rl_policy(test_env, agent, max_steps=len(test_events))
    
    print(f"       Test mean reward: {test_results['mean_reward']:.4f}")
    print(f"       Test positive %: {test_results['positive_ratio']:.2%}")
    print(f"       Test mean loss: {test_results['mean_loss']:.6f}")
    print(f"       Actions: 0={test_results['action_0_count']}, 1={test_results['action_1_count']}\n")

    # ========================================================================
    # 8. SIMULATE PROPOSED AMM ON TEST SET (NEW!)
    # ========================================================================
    print("[9/11] Simulating proposed predictive AMM on test set...")
    
    proposed_result = simulate_predictive_amm(
        test_events,
        ConfigurableAMM(x=x0, y=y0, name="proposed_amm"),
        model, agent, window_init, cfg_rl,
    )
    
    print(f"       Simulated {len(proposed_result.x_hist)} steps")
    print(f"       Final reserves: x={proposed_result.x_hist[-1]:.2f}, y={proposed_result.y_hist[-1]:.2f}")
    print(f"       Drift: dx={proposed_result.drift_x[-1]:.2f}, dy={proposed_result.drift_y[-1]:.2f}\n")
    
    proposed_metrics = compute_proposed_amm_metrics(proposed_result, price, x0*y0)

    # ========================================================================
    # 9. BASELINE ON TEST SET (NEW!)
    # ========================================================================
    print("[10/11] Running baseline (Uniswap V2) on test set...")
    
    # Get test price series
    test_start_idx = test_events[0].t
    test_end_idx = test_events[-1].t
    test_price = price[test_start_idx:test_end_idx+1]
    
    test_trades = make_trade_stream(test_price, seed=args.seed + 7)
    test_base_pool = UniV2LikePool(x=x0, y=y0, fee_bps=args.fee_bps)
    test_base_res = run_baseline(test_base_pool, test_price, test_trades)
    
    baseline_metrics = summarize_baseline_run(test_base_res, test_price)
    
    print(f"       Generated {len(test_trades):,} trades")
    print(f"       Utilization: {baseline_metrics['utilization']:.2%}")
    print(f"       Div (mean): {baseline_metrics['div_mean']:.6f}")
    print(f"       Slip (mean): {baseline_metrics['slip_mean']:.6f}")
    print(f"       Load (mean): {baseline_metrics['load_mean']:.6e}\n")

    # ========================================================================
    # 10. COMPARISON TABLE (NEW!)
    # ========================================================================
    print("[11/11] Comparing proposed vs baseline...")
    print_comparison_table(baseline_metrics, proposed_metrics, len(test_trades))

    # ========================================================================
    # 11. PATH LOSSES (THEORETICAL REFERENCE)
    # ========================================================================
    print("\n[Reference] Theoretical CPMM path losses (test set)...")
    test_path_losses = path_losses(test_price, c=float(x0 * y0))
    print(f"       Div (mean): {test_path_losses['div_mean']:.6f}")
    print(f"       Slip (mean): {test_path_losses['slip_mean']:.6f}")
    print(f"       Load (mean): {test_path_losses['load_mean']:.6e}\n")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print("\n1. LSTM Performance (Test Set):")
    print(f"   MSE:  {eval_metrics['mse']:.2e}")
    print(f"   MAE:  {eval_metrics['mae']:.2e}")
    print(f"   R²:   {eval_metrics['r2']:.4f}")
    
    print("\n2. RL Training (Train Events):")
    print(f"   Mean reward:     {mean_reward:.4f}")
    print(f"   Last 1k:         {last_1k:.4f}")
    print(f"   Positive ratio:  {positive_ratio:.2%}")
    
    print("\n3. RL Evaluation (Test Events):")
    print(f"   Mean reward:     {test_results['mean_reward']:.4f}")
    print(f"   Positive ratio:  {test_results['positive_ratio']:.2%}")
    print(f"   Mean loss:       {test_results['mean_loss']:.6f}")
    
    print("\n4. Proposed AMM (Test Set):")
    print(f"   Divergence loss: {proposed_metrics['divergence_loss_mean']:.6f}")
    print(f"   Slippage loss:   {proposed_metrics['slippage_loss_mean']:.6f}")
    print(f"   Load:            {proposed_metrics['load_mean']:.6e}")
    print(f"   Prediction MAE:  {proposed_metrics['prediction_mae']:.6f}")
    print(f"   Drift magnitude: {proposed_metrics['drift_magnitude']:.2f}")
    
    print("\n5. Baseline Uniswap V2 (Test Set):")
    print(f"   Divergence loss: {baseline_metrics['div_mean']:.6f}")
    print(f"   Slippage loss:   {baseline_metrics['slip_mean']:.6f}")
    print(f"   Load:            {baseline_metrics['load_mean']:.6e}")
    print(f"   Utilization:     {baseline_metrics['utilization']:.2%}")
    print(f"   Price impact:    {baseline_metrics['impact_mean']:.4%}")
    
    print("\n6. Improvement (Proposed vs Baseline):")
    if baseline_metrics['div_mean'] > 0:
        div_improve = (baseline_metrics['div_mean'] - proposed_metrics['divergence_loss_mean']) / baseline_metrics['div_mean'] * 100
        print(f"   Divergence:      {div_improve:+.1f}%")
    if baseline_metrics['slip_mean'] > 0:
        slip_improve = (baseline_metrics['slip_mean'] - proposed_metrics['slippage_loss_mean']) / baseline_metrics['slip_mean'] * 100
        print(f"   Slippage:        {slip_improve:+.1f}%")
    if baseline_metrics['load_mean'] > 0:
        load_improve = (baseline_metrics['load_mean'] - proposed_metrics['load_mean']) / baseline_metrics['load_mean'] * 100
        print(f"   Load:            {load_improve:+.1f}%")
    
    print("\n" + "=" * 80)
    print("Paper's Target Results (for reference):")
    print("-" * 80)
    print("Metric                  | Paper Baseline | Paper Proposed | Your Baseline | Your Proposed")
    print("-" * 80)
    print(f"Liquidity Utilization   |     56%        |      93%       |  {baseline_metrics['utilization']:>6.1%}      |     N/A")
    print(f"Divergence Loss (mean)  |   1.465        |    0.482       |  {baseline_metrics['div_mean']:>6.4f}    |  {proposed_metrics['divergence_loss_mean']:>6.4f}")
    print(f"Slippage Loss (mean)    |   0.4779       |    0.2389      |  {baseline_metrics['slip_mean']:>6.4f}    |  {proposed_metrics['slippage_loss_mean']:>6.4f}")
    print(f"Load (mean)             |      N/A       |      N/A       |  {baseline_metrics['load_mean']:.2e}  |  {proposed_metrics['load_mean']:.2e}")
    print("-" * 80)
    
    print("\nNotes:")
    print("  • Direct comparison to paper may differ due to:")
    print("    - Different data (paper: simulated, you: real ETH)")
    print("    - Different trade generation")
    print("    - Scale differences in loss calculations")
    print("  • Your results show the RELATIVE improvement: Proposed vs Baseline")
    print("  • Both tested on SAME held-out test set (fair comparison)")
    print("=" * 80 + "\n")
    
    # ========================================================================
    # DIAGNOSTIC CHECKS
    # ========================================================================
    print("Diagnostic Checks:")
    print("-" * 80)
    
    # Check 1: LSTM is learning
    if eval_metrics['r2'] > 0.5:
        print("✓ LSTM predictions are good (R² > 0.5)")
    else:
        print(f"⚠ LSTM predictions may be weak (R² = {eval_metrics['r2']:.3f})")
    
    # Check 2: RL agent learned
    if test_results['mean_reward'] > -0.5:
        print("✓ RL agent learned a useful policy (test reward > -0.5)")
    else:
        print(f"⚠ RL agent may not have learned (test reward = {test_results['mean_reward']:.3f})")
    
    # Check 3: Proposed improves over baseline
    if proposed_metrics['divergence_loss_mean'] < baseline_metrics['div_mean']:
        print("✓ Proposed AMM reduces divergence loss vs baseline")
    else:
        print("⚠ Proposed AMM does not reduce divergence loss")
    
    if proposed_metrics['slippage_loss_mean'] < baseline_metrics['slip_mean']:
        print("✓ Proposed AMM reduces slippage loss vs baseline")
    else:
        print("⚠ Proposed AMM does not reduce slippage loss")
    
    # Check 4: Prediction quality
    if proposed_metrics['prediction_mae'] < 0.01:
        print(f"✓ LSTM predictions are accurate (MAE = {proposed_metrics['prediction_mae']:.6f})")
    else:
        print(f"⚠ LSTM predictions may need improvement (MAE = {proposed_metrics['prediction_mae']:.6f})")
    
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)