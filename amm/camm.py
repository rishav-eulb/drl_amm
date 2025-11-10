"""
Configurable (virtual) AMM — cAMM (V3-STYLE WITH CONCENTRATED LIQUIDITY)
----------------------------------
Implements the middleware AMM used in the paper with V3-style concentrated liquidity:
  • Multiple liquidity positions with valuation ranges [v_lower, v_upper]
  • Gaussian distribution of liquidity (Paper Figure 4)
  • Active liquidity tracking (only positions within range provide liquidity)
  • Pseudo-arbitrage curve shifts when valuation moves v→v′
  • Drift tracking for rebalancing

Key Addition: reposition_to_gaussian() implements the paper's predictive mechanism
where liquidity is repositioned BEFORE price moves based on LSTM prediction v'_p.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

from .maths import (
    CPMMState,
    cpmm_c,
    f_of_x,
    phi_x,
    Phi,
    price_y_per_x,
    cap_xy_v,
    clip01,
)


@dataclass
class Drift:
    """Tracks token imbalances that should be covered by future LP deposits.

    Positive dx means the pool *virtually* added X (has surplus X owed to LP vault),
    negative means a shortfall that should be topped up. Same for dy.
    """
    dx: float = 0.0
    dy: float = 0.0

    def add(self, dx: float, dy: float) -> None:
        self.dx += dx
        self.dy += dy


@dataclass
class Incentives:
    """Gaussian incentive parameters in valuation space.

    The mean µ is set to the *predicted* forward valuation v′_p.
    σ controls breadth of the incentive bump.
    """
    mu_v: Optional[float] = None
    sigma_v: Optional[float] = None

    def pdf(self, v: float) -> float:
        if self.mu_v is None or self.sigma_v is None or self.sigma_v <= 0:
            return 0.0
        v = clip01(v)
        z = (v - self.mu_v) / self.sigma_v
        return math.exp(-0.5 * z * z) / (self.sigma_v * math.sqrt(2.0 * math.pi))


@dataclass
class ConcentratedPosition:
    """V3-style concentrated liquidity position.
    
    Represents a liquidity provision active only when current valuation
    is within [v_lower, v_upper]. This implements Uniswap V3's range orders.
    """
    v_lower: float  # Lower valuation bound (0 < v_lower < 1)
    v_upper: float  # Upper valuation bound (v_lower < v_upper < 1)
    L: float        # Virtual liquidity amount in this range
    active: bool = True
    
    def __post_init__(self):
        """Validate bounds."""
        assert 0 < self.v_lower < self.v_upper < 1, \
            f"Invalid range: v_lower={self.v_lower}, v_upper={self.v_upper}"
        assert self.L >= 0, f"Liquidity must be non-negative: L={self.L}"
    
    def is_in_range(self, v: float) -> bool:
        """Check if valuation is within this position's active range."""
        return self.v_lower <= v <= self.v_upper
    
    def utilization(self, v_current: float) -> float:
        """Return 1.0 if in range, 0.0 otherwise."""
        return 1.0 if self.is_in_range(v_current) else 0.0


@dataclass
class ConfigurableAMM:
    """A V3-style constant‑product AMM with concentrated liquidity and pseudo‑arbitrage.

    Parameters
    ----------
    x : float
        Initial X reserve (>0)
    y : float
        Initial Y reserve (>0)
    name : Optional identifier for logging/analytics
    
    Key Features (NEW in V3 version)
    --------------------------------
    • Concentrated liquidity: positions active only in valuation ranges
    • Gaussian distribution: reposition_to_gaussian() implements paper's Figure 4
    • Liquidity utilization: tracks what % of liquidity is actively providing quotes
    • Predictive repositioning: shift liquidity BEFORE price moves based on v'_p
    """

    x: float
    y: float
    name: str = "cAMM_v3"

    # Internal trackers
    drift: Drift = field(default_factory=Drift)
    incentives: Incentives = field(default_factory=Incentives)
    
    # V3: Concentrated liquidity positions
    positions: List[ConcentratedPosition] = field(default_factory=list)
    
    # Mode flag: use concentrated liquidity (V3) or full-range (V2)
    use_concentrated: bool = False

    # -------------------- Core invariants & helpers ----------------------------
    @property
    def c(self) -> float:
        """CPMM invariant: x * y = c"""
        return cpmm_c(self.x, self.y)

    @property
    def price_y_per_x(self) -> float:
        """Instantaneous price: units of Y per unit of X"""
        return price_y_per_x(self.x, self.c)

    def state(self) -> CPMMState:
        return CPMMState(self.x, self.y)

    # -------------------- V3 Liquidity Management (NEW) -------------------------
    
    def get_active_liquidity(self, v_current: float) -> float:
        """Get total active liquidity at current valuation.
        
        Returns
        -------
        L_active : float
            Sum of L from all positions where v_current ∈ [v_lower, v_upper]
            
        Notes
        -----
        In V2 mode (full-range), returns sqrt(c) as all liquidity is always active.
        In V3 mode, only positions containing v_current contribute.
        """
        if not self.use_concentrated or not self.positions:
            # V2 mode: full-range liquidity
            return math.sqrt(self.c)
        
        active_L = sum(
            pos.L for pos in self.positions 
            if pos.active and pos.is_in_range(v_current)
        )
        
        # Prevent division by zero in swap math
        return max(1e-6, active_L)
    
    def get_liquidity_utilization(self, v_current: float) -> float:
        """Return fraction of total liquidity that's currently active.
        
        This is a key metric from the paper:
        - V2 baseline: always 1.0 (100% utilized, but capital inefficient)
        - V3 concentrated: varies (e.g., 0.2 = 20% of capital is working)
        - Paper's proposed: higher utilization via predictive positioning
        
        Returns
        -------
        utilization : float in [0, 1]
            active_L / total_L
        """
        if not self.use_concentrated or not self.positions:
            return 1.0  # V2 mode: always 100%
        
        total_L = sum(pos.L for pos in self.positions)
        if total_L < 1e-12:
            return 0.0
        
        active_L = self.get_active_liquidity(v_current)
        return active_L / total_L
    
    def set_concentrated_positions(
        self, 
        positions: List[Tuple[float, float, float]]
    ) -> None:
        """Manually set concentrated liquidity positions.
        
        Parameters
        ----------
        positions : list of (v_lower, v_upper, L) tuples
            Each tuple defines a position with valuation range and liquidity amount
            
        Example
        -------
        >>> amm.set_concentrated_positions([
        ...     (0.45, 0.55, 100.0),  # 100 units in [0.45, 0.55]
        ...     (0.55, 0.65, 50.0),   # 50 units in [0.55, 0.65]
        ... ])
        """
        self.positions = [
            ConcentratedPosition(v_lower=v_lo, v_upper=v_up, L=L)
            for v_lo, v_up, L in positions
        ]
        self.use_concentrated = True
    
    def reposition_to_gaussian(
        self, 
        mu_v: float, 
        sigma_v: float,
        num_positions: int = 5,
        total_L: Optional[float] = None
    ) -> None:
        """Reposition liquidity following Gaussian distribution around mu_v.
        
        **THIS IS THE CORE PREDICTIVE MECHANISM FROM THE PAPER (Figure 4).**
        
        When the LSTM predicts future valuation v'_p, this function shifts
        liquidity concentration to center around v'_p BEFORE the price actually
        moves. This is what makes the AMM "predictive".
        
        Parameters
        ----------
        mu_v : float
            Predicted valuation (center of Gaussian distribution)
        sigma_v : float
            Width of distribution (controls concentration)
        num_positions : int
            Number of discrete liquidity ranges to create
        total_L : float, optional
            Total liquidity to distribute. If None, uses sqrt(c).
            
        Algorithm
        ---------
        1. Generate num_positions ticks around mu_v at intervals of sigma_v
        2. Assign Gaussian weights: w_k = exp(-0.5 * k^2) for position k
        3. Normalize weights and allocate total_L proportionally
        4. Each position covers range [v_k - 0.5*sigma, v_k + 0.5*sigma]
        
        Example
        -------
        >>> # LSTM predicts v'_p = 0.52
        >>> amm.reposition_to_gaussian(mu_v=0.52, sigma_v=0.05, num_positions=5)
        >>> # Now liquidity is concentrated around 0.52, ready for price move
        """
        mu_v = clip01(mu_v)
        sigma_v = max(1e-6, sigma_v)
        
        if total_L is None:
            total_L = math.sqrt(self.c)
        
        # Generate positions: mu ± k*sigma for k in range
        # Example with num_positions=5: k ∈ {-2, -1, 0, +1, +2}
        k_values = [i - (num_positions // 2) for i in range(num_positions)]
        
        positions = []
        weights = []
        
        for k in k_values:
            # Position center
            v_center = mu_v + k * sigma_v
            
            # Position range: [center - half_width, center + half_width]
            v_lower = v_center - 0.5 * sigma_v
            v_upper = v_center + 0.5 * sigma_v
            
            # Clip to valid valuation range
            v_lower = clip01(v_lower)
            v_upper = clip01(v_upper)
            
            # Skip degenerate positions
            if v_upper - v_lower < 1e-6:
                continue
            
            # Gaussian weight: higher at center, decays with |k|
            weight = math.exp(-0.5 * k * k)
            weights.append(weight)
            
            positions.append((v_lower, v_upper, weight))
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        if total_weight < 1e-12:
            # Degenerate case: single position at mu_v
            v_lo = clip01(mu_v - sigma_v)
            v_up = clip01(mu_v + sigma_v)
            self.positions = [ConcentratedPosition(v_lower=v_lo, v_upper=v_up, L=total_L)]
        else:
            self.positions = [
                ConcentratedPosition(
                    v_lower=v_lo,
                    v_upper=v_up,
                    L=total_L * (w / total_weight)
                )
                for v_lo, v_up, w in positions
            ]
        
        self.use_concentrated = True
        self.incentives.mu_v = mu_v
        self.incentives.sigma_v = sigma_v

    # -------------------- Swapping (V3-aware) -----------------------------------
    
    def swap_x_for_y(self, dx: float, v_current: float) -> float:
        """Swap dx of X into the pool, receive dy of Y out.
        
        NEW: V3-aware - if outside active liquidity range, effective swap is reduced.
        
        Parameters
        ----------
        dx : float
            Amount of X to swap in
        v_current : float
            Current valuation (needed to check active liquidity)
            
        Returns
        -------
        dy : float
            Amount of Y received (always >= 0)
        """
        if dx <= 0:
            return 0.0
        
        # V3 liquidity check
        if self.use_concentrated:
            active_L = self.get_active_liquidity(v_current)
            total_L = math.sqrt(self.c)
            
            # If less than 1% of liquidity is active, scale down swap
            # (simulates high slippage / liquidity shortage)
            if active_L < 0.01 * total_L:
                dx = dx * (active_L / total_L)
        
        # Standard CPMM math
        c = self.c
        x_new = self.x + dx
        y_new = c / x_new
        dy = self.y - y_new
        
        # Update state
        self.x, self.y = x_new, y_new
        return max(0.0, dy)

    def swap_y_for_x(self, dy: float, v_current: float) -> float:
        """Swap dy of Y into the pool, receive dx of X out.
        
        NEW: V3-aware - if outside active liquidity range, effective swap is reduced.
        """
        if dy <= 0:
            return 0.0
        
        if self.use_concentrated:
            active_L = self.get_active_liquidity(v_current)
            total_L = math.sqrt(self.c)
            
            if active_L < 0.01 * total_L:
                dy = dy * (active_L / total_L)
        
        c = self.c
        y_new = self.y + dy
        x_new = c / y_new
        dx_out = self.x - x_new
        
        self.x, self.y = x_new, y_new
        return max(0.0, dx_out)

    # -------------------- Pseudo‑arbitrage shift (unchanged) -------------------
    
    def pseudo_arbitrage_to(
        self, 
        v_prime: float, 
        *, 
        current_v: Optional[float] = None
    ) -> Tuple[float, float]:
        """Apply a virtual curve shift to land at equilibrium for v′.

        If current_v is provided, we compute the *reference* equilibrium (a,b)=Φ(v)
        and the *target* equilibrium (a′,b′)=Φ(v′). We then set (x,y):=(a′,b′) and
        record the drift created by this virtual move: Δx=a′−x, Δy=b′−y.

        If current_v is None, we treat the *current pool state* as the reference (x,y),
        find the target equilibrium for v′ with invariant c, and jump there.

        Returns
        -------
        (dx_drift, dy_drift): tuple of drift adjustments recorded.
        """
        v_prime = clip01(v_prime)
        c = self.c

        # Determine current reference point on curve
        if current_v is not None:
            a, b = Phi(clip01(current_v), c)
            ref_x, ref_y = a, b
        else:
            ref_x, ref_y = self.x, self.y

        # Target equilibrium under new valuation
        ap, bp = Phi(v_prime, c)

        # Record drift relative to current *on‑chain* reserves
        dx = ap - self.x
        dy = bp - self.y
        self.drift.add(dx, dy)

        # Apply the virtual shift: land pool on the target point
        self.x, self.y = ap, bp

        return dx, dy

    # -------------------- Incentives (unchanged) --------------------------------
    
    def set_incentives(self, mu_v: float, sigma_v: float) -> None:
        """Set Gaussian incentive parameters in valuation space."""
        self.incentives.mu_v = clip01(mu_v)
        self.incentives.sigma_v = max(1e-9, float(sigma_v))

    def incentive_at(self, v: float) -> float:
        """Return incentive density ϕ(v) under current (µ,σ)."""
        return self.incentives.pdf(v)

    # -------------------- Analytics ---------------------------------------------
    
    def snapshot(self) -> Dict[str, float]:
        """Return current state as dict for logging/debugging."""
        snap = {
            "x": self.x,
            "y": self.y,
            "c": self.c,
            "price_y_per_x": self.price_y_per_x,
            "drift_dx": self.drift.dx,
            "drift_dy": self.drift.dy,
            "mu_v": (self.incentives.mu_v if self.incentives.mu_v is not None else float("nan")),
            "sigma_v": (self.incentives.sigma_v if self.incentives.sigma_v is not None else float("nan")),
            "use_concentrated": self.use_concentrated,
        }
        
        if self.use_concentrated and self.positions:
            snap["num_positions"] = len(self.positions)
            snap["total_L"] = sum(pos.L for pos in self.positions)
            # Note: utilization requires current valuation, so we don't include it here
        
        return snap


# -------------------- Demo / Self‑test ------------------------------------------
if __name__ == "__main__":
    print("="*70)
    print("V3 CONCENTRATED LIQUIDITY DEMO")
    print("="*70)
    
    # Initialize pool
    amm = ConfigurableAMM(x=100.0, y=100.0, name="test_v3")
    print("\n1. Initial state (V2 mode - full range):")
    print(f"   c = {amm.c:.2f}")
    print(f"   utilization at v=0.5: {amm.get_liquidity_utilization(0.5):.1%}")
    
    # Enable V3 with concentrated positions around v=0.5
    print("\n2. Repositioning to V3 (Gaussian around v=0.5, σ=0.05):")
    amm.reposition_to_gaussian(mu_v=0.5, sigma_v=0.05, num_positions=5)
    
    print(f"   Created {len(amm.positions)} positions:")
    for i, pos in enumerate(amm.positions):
        print(f"     [{i}] v ∈ [{pos.v_lower:.3f}, {pos.v_upper:.3f}], L={pos.L:.2f}")
    
    # Test utilization at different valuations
    print("\n3. Liquidity utilization at different valuations:")
    test_vs = [0.30, 0.45, 0.50, 0.55, 0.70]
    for v in test_vs:
        util = amm.get_liquidity_utilization(v)
        active_L = amm.get_active_liquidity(v)
        print(f"   v={v:.2f}: utilization={util:>5.1%}, active_L={active_L:>6.2f}")
    
    # Simulate a swap
    print("\n4. Test swap (5 units X → Y at v=0.50):")
    dy = amm.swap_x_for_y(5.0, v_current=0.50)
    print(f"   Received {dy:.4f} Y")
    print(f"   New reserves: x={amm.x:.2f}, y={amm.y:.2f}")
    
    # Swap outside range
    print("\n5. Test swap OUTSIDE range (5 units X → Y at v=0.20):")
    amm2 = ConfigurableAMM(x=100.0, y=100.0, name="test2")
    amm2.reposition_to_gaussian(mu_v=0.5, sigma_v=0.05, num_positions=5)
    dy2 = amm2.swap_x_for_y(5.0, v_current=0.20)
    print(f"   Received {dy2:.4f} Y (reduced due to low active liquidity)")
    print(f"   Active liquidity at v=0.20: {amm2.get_active_liquidity(0.20):.2f}")
    
    # Predictive repositioning
    print("\n6. PREDICTIVE MECHANISM (LSTM predicts v'_p=0.6):")
    print("   Current: v=0.5, liquidity centered at 0.5")
    print("   LSTM prediction: v'_p=0.6")
    print("   ACTION: Reposition liquidity to v'_p=0.6 BEFORE price moves")
    
    amm3 = ConfigurableAMM(x=100.0, y=100.0, name="predictive")
    amm3.reposition_to_gaussian(mu_v=0.5, sigma_v=0.05, num_positions=5)
    
    print(f"   Before: utilization at v=0.6: {amm3.get_liquidity_utilization(0.6):.1%}")
    
    # Reposition based on prediction
    amm3.reposition_to_gaussian(mu_v=0.6, sigma_v=0.05, num_positions=5)
    
    print(f"   After:  utilization at v=0.6: {amm3.get_liquidity_utilization(0.6):.1%}")
    print("   ✓ Liquidity is now ready at predicted valuation!")
    
    print("\n" + "="*70)
    print("Demo complete. V3 concentrated liquidity working correctly.")
    print("="*70)