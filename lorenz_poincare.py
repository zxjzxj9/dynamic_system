"""
Lorenz System — Poincaré Sections
==================================

Poincaré sections record where the trajectory pierces specific planes,
revealing the fractal microstructure hidden within the strange attractor.

Three sections through the Lorenz attractor (σ=10, ρ=28, β=8/3):
    1. z = 27 (= ρ−1), upward crossings  → plot (x, y)
    2. x = 0, rightward crossings         → plot (y, z)
    3. y = 0, crossings with ẏ > 0        → plot (x, z)

Crossing detection: sign-change scan + linear interpolation for accuracy.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ── Parameters ──

sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0


def lorenz(t, state):
    """Lorenz system right-hand side."""
    x, y, z = state
    return [
        sigma * (y - x),
        rho * x - x * z - y,
        x * y - beta * z,
    ]


# ── Integration (long run for many crossings) ──

print("Integrating Lorenz system (t ∈ [0, 1000])...")
y0 = [1.0, 1.0, 1.0]
t_span = (0.0, 1000.0)
sol = solve_ivp(lorenz, t_span, y0, method="RK45", max_step=0.01,
                rtol=1e-9, atol=1e-12)

# Discard transient (first 20 time units)
mask = sol.t > 20.0
t = sol.t[mask]
x, y, z = sol.y[0, mask], sol.y[1, mask], sol.y[2, mask]

print(f"  {len(t)} points after transient removal")


# ── Crossing detection ──

def find_crossings(values, states, direction="positive"):
    """Find plane crossings by sign changes with linear interpolation.

    Parameters
    ----------
    values : array — signed distance to the section plane for each point
    states : array of shape (N, D) — state vectors at each point
    direction : "positive" for crossings where value goes - → +

    Returns
    -------
    crossings : array of shape (M, D) — interpolated states at crossings
    """
    crossings = []
    for i in range(len(values) - 1):
        if direction == "positive" and values[i] < 0 and values[i + 1] >= 0:
            pass
        elif direction == "negative" and values[i] > 0 and values[i + 1] <= 0:
            pass
        else:
            continue
        # Linear interpolation fraction
        alpha = -values[i] / (values[i + 1] - values[i])
        crossing = states[i] + alpha * (states[i + 1] - states[i])
        crossings.append(crossing)
    return np.array(crossings) if crossings else np.empty((0, states.shape[1]))


states = np.column_stack([x, y, z])

# Section 1: z = 27 (= ρ−1), upward crossings (ż > 0)
print("Finding z = 27 crossings (upward)...")
crossings_z27 = find_crossings(z - 27.0, states, direction="positive")
print(f"  {len(crossings_z27)} crossings")

# Section 2: x = 0, rightward crossings (ẋ > 0)
print("Finding x = 0 crossings (rightward)...")
crossings_x0 = find_crossings(x, states, direction="positive")
print(f"  {len(crossings_x0)} crossings")

# Section 3: y = 0, crossings with ẏ > 0
print("Finding y = 0 crossings (ẏ > 0)...")
crossings_y0 = find_crossings(y, states, direction="positive")
print(f"  {len(crossings_y0)} crossings")


# ── Figure: 1×3 panels ──

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sections = [
    (axes[0], crossings_z27, 0, 1, "x", "y",
     r"$z = 27\;(\rho - 1)$, $\dot{z} > 0$"),
    (axes[1], crossings_x0, 1, 2, "y", "z",
     r"$x = 0$, $\dot{x} > 0$"),
    (axes[2], crossings_y0, 0, 2, "x", "z",
     r"$y = 0$, $\dot{y} > 0$"),
]

for ax, crossings, hi, vi, hlabel, vlabel, title in sections:
    if len(crossings) == 0:
        ax.text(0.5, 0.5, "No crossings", transform=ax.transAxes,
                ha="center", va="center")
        continue
    # Color by crossing index to show temporal structure
    colors = np.arange(len(crossings))
    ax.scatter(crossings[:, hi], crossings[:, vi],
               c=colors, cmap="inferno", s=1.5, alpha=0.7, edgecolors="none")
    ax.set_xlabel(hlabel, fontsize=12)
    ax.set_ylabel(vlabel, fontsize=12)
    ax.set_title(f"Poincaré section: {title}", fontsize=11)
    ax.grid(True, alpha=0.2)

fig.suptitle(
    "Poincaré Sections of the Lorenz Attractor\n"
    r"$\sigma=10,\;\rho=28,\;\beta=8/3$"
    r"$\qquad t \in [20,\, 1000]$",
    fontsize=14, y=1.02)

plt.tight_layout()
plt.savefig("lorenz_poincare.png", dpi=150, bbox_inches="tight")
print("\nSaved lorenz_poincare.png")

plt.show()

# ── Summary ──

print("\n=== Poincaré Sections Summary ===")
print(f"  Integration: t ∈ [0, 1000], transient discarded t < 20")
print(f"  Total solution points: {len(t)}")
print(f"\n  Section crossings:")
print(f"    z = 27 (upward):    {len(crossings_z27)}")
print(f"    x = 0  (rightward): {len(crossings_x0)}")
print(f"    y = 0  (ẏ > 0):     {len(crossings_y0)}")
print(f"\n  The fractal banding visible in each section reflects the")
print(f"  strange attractor's non-integer dimension (≈ 2.06).")
print(f"  Each 'band' contains infinitely many sub-bands — a Cantor-like")
print(f"  structure that is the geometric signature of deterministic chaos.")
