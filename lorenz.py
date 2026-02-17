"""
Lorenz System — Strange Attractor
==================================

    dx/dt = σ(y − x)
    dy/dt = ρx − xz − y
    dz/dt = xy − βz

Classic chaotic parameters: σ = 10, ρ = 28, β = 8/3

The Lorenz system is one of the first examples of deterministic chaos.
Its trajectory never repeats yet remains bounded on the strange attractor —
a fractal set with non-integer dimension ≈ 2.06.

Equilibria (for ρ > 1):
    Origin (0, 0, 0)         — unstable (saddle with one unstable direction)
    C± = (±√(β(ρ−1)), ±√(β(ρ−1)), ρ−1)  — unstable spirals
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


# ── Integration ──

print("Integrating Lorenz system...")
y0 = [1.0, 1.0, 1.0]
t_span = (0.0, 100.0)
sol = solve_ivp(lorenz, t_span, y0, method="RK45", max_step=0.01,
                rtol=1e-9, atol=1e-12, dense_output=True)

# Discard transient (first 5 time units)
mask = sol.t > 5.0
t = sol.t[mask]
x, y, z = sol.y[0, mask], sol.y[1, mask], sol.y[2, mask]

print(f"  {len(t)} points after transient removal")

# ── Equilibria ──

C = np.sqrt(beta * (rho - 1))  # ≈ 8.485
eq_origin = (0.0, 0.0, 0.0)
eq_plus = (C, C, rho - 1)
eq_minus = (-C, -C, rho - 1)

# ── Color by time for visual depth ──

colors = t - t[0]
colors /= colors[-1]  # normalize to [0, 1]

# ── Figure: multi-panel projections ──

fig = plt.figure(figsize=(16, 12))

# Layout: XZ projection spans top row (2 columns), XY and YZ below
ax_xz = fig.add_subplot(2, 2, (1, 2))  # top row, spans both columns
ax_xy = fig.add_subplot(2, 2, 3)
ax_yz = fig.add_subplot(2, 2, 4)

panels = [
    (ax_xz, x, z, "x", "z", "X–Z Projection  (the butterfly)"),
    (ax_xy, x, y, "x", "y", "X–Y Projection"),
    (ax_yz, y, z, "y", "z", "Y–Z Projection"),
]

equilibria = [
    (eq_origin, "Origin (unstable saddle)"),
    (eq_plus, "C⁺ (unstable spiral)"),
    (eq_minus, "C⁻ (unstable spiral)"),
]

# Axis index mapping: x=0, y=1, z=2
axis_map = {"x": 0, "y": 1, "z": 2}

for ax, h_data, v_data, h_label, v_label, title in panels:
    # Trajectory colored by time
    for i in range(len(h_data) - 1):
        ax.plot(h_data[i:i+2], v_data[i:i+2],
                color=plt.cm.inferno(colors[i]), lw=0.3, alpha=0.8)

    # Equilibria
    h_idx, v_idx = axis_map[h_label], axis_map[v_label]
    for eq, label in equilibria:
        ax.plot(eq[h_idx], eq[v_idx], "o", color="lime", markersize=6,
                markeredgecolor="black", markeredgewidth=0.8, zorder=5)
        ax.annotate(label, xy=(eq[h_idx], eq[v_idx]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=7.5, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7))

    ax.set_xlabel(h_label, fontsize=11)
    ax.set_ylabel(v_label, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_facecolor("#1a1a2e")
    ax.grid(True, alpha=0.15, color="white")
    ax.tick_params(colors="0.6")
    for spine in ax.spines.values():
        spine.set_color("0.4")

# Colorbar for time
sm = plt.cm.ScalarMappable(cmap="inferno",
                           norm=plt.Normalize(vmin=t[0], vmax=t[-1]))
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax_xz, ax_xy, ax_yz], location="bottom",
                    fraction=0.04, pad=0.08, aspect=60)
cbar.set_label("Time (t)", fontsize=11)

fig.patch.set_facecolor("#0f0f23")
fig.suptitle(
    "Lorenz Strange Attractor\n"
    r"$\dot{x}=\sigma(y-x),\quad \dot{y}=\rho x - xz - y,\quad \dot{z}=xy - \beta z$"
    f"\n(σ={sigma:.0f}, ρ={rho:.0f}, β=8/3)",
    fontsize=14, color="white", y=0.98)

plt.savefig("lorenz_attractor.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved lorenz_attractor.png")

plt.show()

# ── Summary ──

print("\n=== Lorenz System Summary ===")
print(f"  Parameters: σ = {sigma}, ρ = {rho}, β = 8/3 ≈ {beta:.4f}")
print(f"\n  Equilibria:")
print(f"    Origin   (0, 0, 0)            — unstable saddle")
print(f"    C⁺       ({C:.3f}, {C:.3f}, {rho-1:.1f})  — unstable spiral")
print(f"    C⁻       ({-C:.3f}, {-C:.3f}, {rho-1:.1f}) — unstable spiral")
print(f"\n  All three equilibria are unstable, so the trajectory is")
print(f"  forever attracted to the strange attractor but never settles.")
