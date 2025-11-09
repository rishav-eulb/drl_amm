"""
Configurable (virtual) AMM — cAMM
----------------------------------
Implements the middleware AMM used in the paper to:
  • hold CPMM state (x, y, c)
  • execute swaps with exact-out math
  • perform pseudo‑arbitrage curve shifts when valuation moves v→v′
  • track inventory drift created by pseudo‑arbitrage shifts
  • store (µ,σ) incentive parameters tied to predicted valuation v′_p

This cAMM is a *simulation* component. It models the paper's idea that when
market valuation jumps, the protocol can apply a virtual curve shift to land on the
new equilibrium immediately, reducing arbitrageable divergence. Any residual token
imbalance is tracked as drift to be rebalanced by LP deposits later.

NOTE: Pricing is CPMM (constant product) and delegates math to amm.math.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple
import math

from .maths import (
    CPMMState,
    cpmm_c,
    f_of_x,
    phi_x,
    Phi,
    price_y_per_x,
    cap_xy_v,
    expected_load,
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
class ConfigurableAMM:
    """A configurable constant‑product AMM with pseudo‑arbitrage shifts.

    Parameters
    ----------
    x : float
        Initial X reserve (>0)
    y : float
        Initial Y reserve (>0)
    name : Optional identifier for logging/analytics
    """

    x: float
    y: float
    name: str = "cAMM"

    # Internal trackers
    drift: Drift = field(default_factory=Drift)
    incentives: Incentives = field(default_factory=Incentives)

    # -------------------- core invariants & helpers ----------------------------
    @property
    def c(self) -> float:
        return cpmm_c(self.x, self.y)

    @property
    def price_y_per_x(self) -> float:
        return price_y_per_x(self.x, self.c)

    def state(self) -> CPMMState:
        return CPMMState(self.x, self.y)

    # -------------------- swapping ------------------------------------------------
    def swap_x_for_y(self, dx: float) -> float:
        """Swap dx of X into the pool, receive dy of Y out (exact‑out CPMM math).
        Returns dy (>0) the taker receives. Updates reserves.
        """
        if dx <= 0:
            return 0.0
        c = self.c
        x_new = self.x + dx
        y_new = c / x_new
        dy = self.y - y_new
        # update state
        self.x, self.y = x_new, y_new
        return dy

    def swap_y_for_x(self, dy: float) -> float:
        """Swap dy of Y into the pool, receive dx of X out.
        Returns dx (>0) the taker receives. Updates reserves.
        """
        if dy <= 0:
            return 0.0
        c = self.c
        y_new = self.y + dy
        x_new = c / y_new
        dx_out = self.x - x_new
        self.x, self.y = x_new, y_new
        return dx_out

    # -------------------- pseudo‑arbitrage shift ---------------------------------
    def pseudo_arbitrage_to(self, v_prime: float, *, current_v: Optional[float] = None) -> Tuple[float, float]:
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

    # -------------------- incentives --------------------------------------------
    def set_incentives(self, mu_v: float, sigma_v: float) -> None:
        """Set Gaussian incentive parameters in valuation space."""
        self.incentives.mu_v = clip01(mu_v)
        self.incentives.sigma_v = max(1e-9, float(sigma_v))

    def incentive_at(self, v: float) -> float:
        """Return incentive density ϕ(v) under current (µ,σ)."""
        return self.incentives.pdf(v)

    # -------------------- analytics ---------------------------------------------
    def snapshot(self) -> Dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "c": self.c,
            "price_y_per_x": self.price_y_per_x,
            "drift_dx": self.drift.dx,
            "drift_dy": self.drift.dy,
            "mu_v": (self.incentives.mu_v if self.incentives.mu_v is not None else float("nan")),
            "sigma_v": (self.incentives.sigma_v if self.incentives.sigma_v is not None else float("nan")),
        }


# -------------------- quick demo / self‑test -------------------------------------
if __name__ == "__main__":
    # initialize a pool
    amm = ConfigurableAMM(x=100.0, y=100.0)
    print("init", amm.snapshot())

    # execute a trade X->Y
    got_y = amm.swap_x_for_y(5.0)
    print("swap_x_for_y dy=", got_y)
    print("post‑trade", amm.snapshot())

    # pseudo‑arb shift to new valuation v′
    vprime = 0.6
    dx, dy = amm.pseudo_arbitrage_to(vprime)
    print("pseudo‑arb shift to v′=", vprime, " drift=(", dx, dy, ")")
    print("after shift", amm.snapshot())

    # set incentives around predicted valuation v′_p
    amm.set_incentives(mu_v=0.58, sigma_v=0.02)
    print("ϕ(0.58)=", amm.incentive_at(0.58), " ϕ(0.55)=", amm.incentive_at(0.55))
