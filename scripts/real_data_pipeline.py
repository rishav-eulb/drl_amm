"""
Modified Pipeline Sections for Improved RL Training
====================================================
Replace the corresponding sections in scripts/real_data_pipeline.py
with these improved versions.

This file shows the key changes needed to use the scale-invariant
RL environment for better training results.
"""

# ============================================================================
# SECTION TO REPLACE: RL Training (lines ~380-450 in original)
# ============================================================================

def setup_and_train_rl_improved(
    train_events,
    test_events, 
    model,  # LSTM model
    window_init,
    args,
    price  # Full price series for volatility estimation
):
    """Improved RL training with scale-invariant environment."""
    
    # Import the improved environment
    from ml.rl_env_improved import ImprovedRLEnv, RLEnvConfig
    from ml.dqn import DDQNAgent, DDQNConfig, train as dqn_train
    from amm.camm import ConfigurableAMM
    import numpy as np
    from ml.lstm import predict as lstm_predict
    
    print("[6/11] Setting up improved RL environment...")
    
    # ========== REALISTIC POOL INITIALIZATION ==========
    # Use realistic pool size based on initial price
    p0 = float(price[0])
    
    # Option 1: Fixed TVL approach (e.g., $1M total)
    target_tvl = 1_000_000.0  # $1M TVL
    y0 = target_tvl / 2.0  # Split evenly in value
    x0 = y0 / p0
    
    # Option 2: Normalized approach (for testing)
    # x0 = 1.0
    # y0 = p0
    
    camm_init = ConfigurableAMM(x=x0, y=y0, name="cAMM_train")
    print(f"       Pool initialization:")
    print(f"         x={x0:.2f}, y=${y0:,.2f}")
    print(f"         Initial price=${p0:.2f}")
    print(f"         TVL=${x0*p0 + y0:,.2f}")
    print(f"         c={x0*y0:.2e}\n")
    
    # ========== LSTM PREDICTOR FUNCTION ==========
    def lstm_pred_fn(win_np: np.ndarray) -> float:
        """LSTM predictor with epsilon handling."""
        if win_np.ndim != 2:
            raise ValueError(f"lstm_pred_fn expects [T,D], got {win_np.shape}")
        
        w = win_np[None, :, :].astype(np.float32)
        out = lstm_predict(model, w)
        return float(np.clip(out[0], 1e-6, 1.0 - 1e-6))
    
    # ========== IMPROVED RL CONFIG ==========
    cfg_rl = RLEnvConfig(
        # Scale-invariant thresholds
        beta_c_relative=0.0001,        # 0.01% relative loss threshold
        prediction_weight=0.5,          # Balance prediction vs market-making
        
        # Adaptive threshold based on market volatility
        use_adaptive_threshold=True,
        volatility_window=100,
        vol_multiplier=2.0,
        
        # Epsilon parameters (unchanged)
        mu_epsilon=args.mu_epsilon,
        sigma_epsilon=args.sigma_epsilon,
        
        # Other parameters
        samples_per_step=16,
        sigma_noise=0.02,
        lstm_win=args.lstm_win,
        seed=args.seed,
        track_losses=True,
        
        # Feature processing
        normalize_reserves=True,
        use_log_reserves=True,  # Better for neural networks
    )
    
    # ========== CREATE TRAINING ENVIRONMENT ==========
    train_env = ImprovedRLEnv(
        cfg_rl,
        train_events,
        camm_init,
        lstm_pred_fn,
        window_init,
        price_series=price  # Pass full price series for volatility
    )
    
    print("[7/11] Training improved DD-DQN agent...")
    
    # ========== DQN AGENT WITH UPDATED STATE DIM ==========
    agent = DDQNAgent(
        state_dim=8,  # UPDATED: 8D state vector
        n_actions=2,
        cfg=DDQNConfig(
            gamma=args.rl_gamma,
            lr=5e-4,  # Slightly lower learning rate
            hidden=256,  # Larger network for complex patterns
            batch_size=256,
            buffer_size=args.rl_buffer,
            min_buffer=5_000,
            train_freq=1,
            target_sync=1000,
            tau=0.001,  # Soft updates for stability
            eps_start=1.0,
            eps_end=0.01,  # Lower final exploration
            eps_decay_steps=args.rl_eps_decay,
            per=False,  # Could enable prioritized replay
            grad_clip=1.0,
            seed=args.seed,
        )
    )
    
    # ========== TRAINING LOOP ==========
    hist = dqn_train(train_env, agent, steps=args.rl_steps, warm_start_random=5_000)
    
    rewards = np.array(hist["reward"])
    mean_reward = float(np.mean(rewards))
    last_1k = float(np.mean(rewards[-1000:])) if len(rewards) >= 1000 else mean_reward
    positive_ratio = float(np.mean(rewards > 0))
    
    print(f"       Training complete: {len(rewards):,} steps")
    print(f"       Mean reward: {mean_reward:.4f}")
    print(f"       Last 1k: {last_1k:.4f}")
    print(f"       Positive %: {positive_ratio:.2%}")
    
    # Get training statistics
    train_stats = train_env.get_loss_stats()
    print(f"       Mean relative loss: {train_stats.get('total_loss_mean', 0):.6f}")
    print(f"       Below threshold %: {train_stats.get('below_threshold_ratio', 0):.2%}\n")
    
    # ========== EVALUATE ON TEST SET ==========
    print("[8/11] Evaluating on test events...")
    
    test_env = ImprovedRLEnv(
        cfg_rl,
        test_events,
        ConfigurableAMM(x=x0, y=y0, name="cAMM_test"),
        lstm_pred_fn,
        window_init,
        price_series=price
    )
    
    # Test with greedy policy (no exploration)
    old_eps_start = agent.cfg.eps_start
    old_eps_end = agent.cfg.eps_end
    agent.cfg.eps_start = 0.0
    agent.cfg.eps_end = 0.0
    
    obs = test_env.reset()
    test_rewards = []
    test_actions = []
    
    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = test_env.step(action)
        test_rewards.append(reward)
        test_actions.append(action)
    
    # Restore exploration settings
    agent.cfg.eps_start = old_eps_start
    agent.cfg.eps_end = old_eps_end
    
    test_stats = test_env.get_loss_stats()
    
    print(f"       Test mean reward: {np.mean(test_rewards):.4f}")
    print(f"       Test positive %: {np.mean(np.array(test_rewards) > 0):.2%}")
    print(f"       Test relative loss: {test_stats.get('total_loss_mean', 0):.6f}")
    print(f"       Actions: 0={test_actions.count(0)}, 1={test_actions.count(1)}\n")
    
    return agent, camm_init, train_env, test_env


# ============================================================================
# EXAMPLE: How to integrate into main pipeline
# ============================================================================

def example_integration():
    """Example of how to integrate the improved RL into your pipeline."""
    
    # After LSTM training and event generation...
    
    # Replace the original RL training section with:
    agent, camm_init, train_env, test_env = setup_and_train_rl_improved(
        train_events=train_events,
        test_events=test_events,
        model=lstm_model,  # Your trained LSTM
        window_init=window_init,  # Your prepared window
        args=args,  # Your argparse args
        price=price  # Full price series
    )
    
    # The rest of your pipeline continues normally...
    # The agent will now be properly trained with scale-invariant losses


# ============================================================================
# KEY IMPROVEMENTS EXPLAINED
# ============================================================================

"""
1. SCALE-INVARIANT LOSSES
   - All losses are relative (percentage-based)
   - Works with any pool size ($1K to $100M+)
   - No need to manually tune beta_c for different pools

2. ADAPTIVE THRESHOLDS
   - Threshold adjusts based on market volatility
   - Agent learns when market is volatile vs calm
   - Better performance in different market conditions

3. IMPROVED STATE REPRESENTATION
   - Log-scale reserves for better neural network training
   - Includes adaptive threshold ratio in state
   - 8D state vector with all relevant information

4. CONTINUOUS REWARDS
   - Smoother reward signal for better gradient flow
   - Clipped to [-1, 1] for stability
   - Proportional to how far loss is from threshold

5. REALISTIC POOL SIZES
   - Initialize with realistic TVL (e.g., $1M)
   - Consistent between training and deployment
   - No artificial scaling needed

EXPECTED IMPROVEMENTS:
- Positive reward ratio: ~30-60% (vs 0% before)
- Mean reward: positive (vs -1.0 before)
- Relative loss: ~0.0001 (vs 78,380 before)
- Learned policy: intelligent epsilon injection based on market conditions
"""

# ============================================================================
# HYPERPARAMETER TUNING GUIDE
# ============================================================================

"""
If results are still suboptimal, try adjusting:

1. beta_c_relative: 
   - Lower (0.00005) for stricter agent
   - Higher (0.0005) for more lenient agent

2. prediction_weight:
   - Higher (0.7) to focus on prediction accuracy
   - Lower (0.3) to focus on market-making efficiency

3. vol_multiplier:
   - Higher (3.0) for more adaptive thresholds
   - Lower (1.0) for more consistent thresholds

4. DQN hyperparameters:
   - Increase hidden size (512) for more complex patterns
   - Enable PER (prioritized experience replay)
   - Adjust learning rate (1e-4 to 1e-3)
   - Longer training (500k steps)

5. Pool initialization:
   - Match your target deployment size
   - Consider multiple pool sizes during training
"""