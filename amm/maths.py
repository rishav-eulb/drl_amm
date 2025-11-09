"""
AMM math utilities (CPMM + Engel–Herlihy equilibrium tools)
-----------------------------------------------------------------
This module implements the core mathematics you need to reproduce
Section 3 / Preliminaries from the paper you’re studying:

  • CPMM pool: f(x) = c / x
  • Equilibrium mapping  φ(v)  from valuation v∈(0,1) to the x-coordinate
    on the CPMM curve where the instantaneous exchange rate matches v
  • Capitalization (cap) at arbitrary states and at equilibrium
  • Divergence loss  loss_div(v, v′)
  • Slippage loss    loss_slip_X(v, v′)
  • Composite load   load_X(v, v′) = loss_div * loss_slip_X
  • Monte‑Carlo expected load   E_p[load]

Notes
-----
• We work with the CPMM bonding curve f(x) = c/x.
• Exchange rate (units of Y per unit of X) is  −f'(x) = c/x^2.
• Equilibrium slope condition (from the paper):
      df/dx = − v/(1−v)   ⇒   c/x^2 = v/(1−v)
  which yields the closed form:
      φ(v) = sqrt( c * (1−v) / v )
• Vector at equilibrium  Φ(v) = ( x(v), y(v) )  with  y(v) = c/x(v).
• Capitalization at (x, y) under valuation v is  cap(x,y;v) = v*x + (1−v)*y.

Implementation hints
--------------------
• All valuations are numerically clipped to (eps, 1−eps) for stability.
• The divergence/slippage/load formulas follow the definitions presented
  in Engel & Herlihy (and restated in the uploaded paper). For CPMM, these
  reduce to simple closed forms built from Φ(v) and Φ(v′).

Author: ChatGPT
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List
import math

_EPS = 1e-9

# ---------- Basic CPMM helpers -------------------------------------------------

def clip01(v: float, eps: float = _EPS) -> float:
    """Clip a scalar into the open interval (0,1)."""
    return min(1.0 - eps, max(eps, v))


def cpmm_c(x: float, y: float) -> float:
    """Constant product invariant c = x * y."""
    if x <= 0 or y <= 0:
        raise ValueError("CPMM requires x>0 and y>0")
    return x * y


def f_of_x(x: float, c: float) -> float:
    """CPMM bonding curve f(x) = c/x (y as a function of x)."""
    if x <= 0:
        raise ValueError("x must be > 0 for CPMM")
    return c / x


def price_y_per_x(x: float, c: float) -> float:
    """Instantaneous exchange rate (units of Y per unit of X): -f'(x) = c/x^2."""
    if x <= 0:
        raise ValueError("x must be > 0")
    return c / (x * x)


# ---------- Equilibrium mapping Φ, capitalization ------------------------------

def phi_x(v: float, c: float) -> float:
    """Equilibrium x given valuation v for CPMM: φ(v) = sqrt(c * (1−v) / v)."""
    v = clip01(v)
    return math.sqrt(c * (1.0 - v) / v)


def Phi(v: float, c: float) -> Tuple[float, float]:
    """Equilibrium point Φ(v) = (x(v), y(v)) on the CPMM curve for valuation v."""
    x = phi_x(v, c)
    y = f_of_x(x, c)
    return x, y


def cap_xy_v(x: float, y: float, v: float) -> float:
    """Capitalization at arbitrary state (x,y) under valuation v: v*x + (1−v)*y."""
    v = clip01(v)
    return v * x + (1.0 - v) * y


def cap_at_equilibrium(v: float, c: float) -> float:
    """Capitalization at equilibrium Φ(v)."""
    x, y = Phi(v, c)
    return cap_xy_v(x, y, v)
d

# ---------- Divergence loss ----------------------------------------------------

def divergence_loss(v: float, v_prime: float, c: float) -> float:
    """Divergence loss as defined in the paper:

    loss_div(v, v′) = v′ · Φ(v) − v′ · Φ(v′)
                    = v′*x(v) + (1−v′)*y(v) − [ v′*x(v′) + (1−v′)*y(v′) ]

    Intuition: start in equilibrium for v, the market moves to valuation v′. An
    arbitrageur can extract this amount by moving the pool to the new equilibrium.
    """
    v = clip01(v)
    v_prime = clip01(v_prime)
    xv, yv = Phi(v, c)
    xv2, yv2 = Phi(v_prime, c)
    term_current = v_prime * xv + (1.0 - v_prime) * yv
    term_equil   = v_prime * xv2 + (1.0 - v_prime) * yv2
    loss = term_current - term_equil
    return max(0.0, loss)


# ---------- Slippage and Load (with-respect-to-X) ------------------------------

def slippage_loss_X(v: float, v_prime: float, c: float) -> float:
    """Slippage loss with respect to X, using the closed form stated in the paper:

    lossslip_X(v, v′) = ((1−v′)/(1−v)) * ( v · Φ(v′) − v · Φ(v) )
                      = ((1−v′)/(1−v)) * [ v*x(v′) + (1−v)*y(v′) − (v*x(v) + (1−v)*y(v)) ]

    This measures how the *nonlinearity* of the curve penalizes an X→Y trade over
    the interval implied by valuations v→v′, relative to a linear-rate execution.
    """
    v = clip01(v)
    v_prime = clip01(v_prime)
    if abs(1.0 - v) < _EPS:
        # Degenerate valuation (almost all weight on X); define slippage as 0.
        return 0.0
    xv, yv = Phi(v, c)
    xv2, yv2 = Phi(v_prime, c)
    dot_v_at_vp = v * xv2 + (1.0 - v) * yv2
    dot_v_at_v  = v * xv  + (1.0 - v) * yv
    return max(0.0, ((1.0 - v_prime) / (1.0 - v)) * (dot_v_at_vp - dot_v_at_v))


def load_X(v: float, v_prime: float, c: float) -> float:
    """Composite load (with-respect-to-X): load_X = loss_div * lossslip_X.

    In the paper, the expected load mixes load_X and load_Y depending on whether
    future valuation moves up or down relative to v. For simplicity, many sims use
    load_X for upward moves (v′>v) and load_Y for downward moves (v′<v).
    """
    return divergence_loss(v, v_prime, c) * slippage_loss_X(v, v_prime, c)


# ---------- Symmetry helper for Y-direction (optional) -------------------------

def _swap_XY_load(v: float, v_prime: float, c: float) -> float:
    """By symmetry, Y-direction quantities can be obtained by swapping X and Y,
    which corresponds to mapping valuations via  v ↦ (1−v).

    We reuse the X-formulas under the transform to approximate load_Y.
    """
    v_s, vp_s = 1.0 - clip01(v), 1.0 - clip01(v_prime)
    return load_X(v_s, vp_s, c)


def load_auto(v: float, v_prime: float, c: float) -> float:
    """Choose X- or Y- direction load automatically by comparing v and v′.

    If v′ ≥ v (valuation of X increased), traders tend to sell Y for X (load_X).
    If v′ <  v, traders tend to sell X for Y (use the symmetric Y‑load).
    """
    return load_X(v, v_prime, c) if v_prime >= v else _swap_XY_load(v, v_prime, c)


# ---------- Expected load via Monte Carlo --------------------------------------

def expected_load(
    v: float,
    samples_vprime: Iterable[float],
    c: float,
    *,
    auto_direction: bool = True,
) -> float:
    """Monte‑Carlo estimate of E_p[load] from a set of future valuation samples.

    Parameters
    ----------
    v : float
        Current valuation in (0,1).
    samples_vprime : Iterable[float]
        Sampled future valuations v′ drawn from some distribution p(v′).
    c : float
        CPMM invariant.
    auto_direction : bool
        If True, pick X vs Y load automatically per sample. If False, always
        use load_X (useful for one‑sided what‑ifs).
    """
    vals: List[float] = []
    for vp in samples_vprime:
        vp = clip01(vp)
        if auto_direction:
            vals.append(load_auto(v, vp, c))
        else:
            vals.append(load_X(v, vp, c))
    return 0.0 if not vals else sum(vals) / len(vals)


# ---------- Convenience: pack/unpack state -------------------------------------

@dataclass
class CPMMState:
    x: float
    y: float

    @property
    def c(self) -> float:
        return cpmm_c(self.x, self.y)

    @property
    def price_y_per_x(self) -> float:
        return price_y_per_x(self.x, self.c)

    def cap(self, v: float) -> float:
        return cap_xy_v(self.x, self.y, v)

    @classmethod
    def from_equilibrium(cls, v: float, c: float) -> "CPMMState":
        x, y = Phi(v, c)
        return cls(x=x, y=y)


# ---------- Minimal self‑test (can be run manually) ----------------------------
if __name__ == "__main__":
    c = 10_000.0
    v  = 0.4
    vp = 0.6
    x = phi_x(v, c)
    y = f_of_x(x, c)
    assert abs(x * y - c) < 1e-6
    cap_v  = cap_at_equilibrium(v, c)
    cap_vp = cap_at_equilibrium(vp, c)
    ld = divergence_loss(v, vp, c)
    ls = slippage_loss_X(v, vp, c)
    l  = load_X(v, vp, c)
    print("Equilibrium x(v)=", x, " y(v)=", y)
    print("cap(v)=", cap_v, " cap(v′)=", cap_vp)
    print("loss_div=", ld, " loss_slip_X=", ls, " load_X=", l)
    est = expected_load(v, [vp]*100, c)
    print("E[load] (degenerate 100 samples at v′)=", est)
