"""
Van der Pol Oscillator Phase Portrait
======================================

    dx/dt = y
    dy/dt = μ(1 - x²)y - x

Parameters used:  μ = 1.5
Equilibrium:
    (0, 0) — unstable spiral (repeller) surrounded by a stable limit cycle
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.integrate import solve_ivp

from phase_portrait import find_equilibria, classify_equilibrium


# ── Parameters ──
mu = 1.5

x, y = sp.symbols("x y")
f_expr = y                              # dx/dt
g_expr = mu * (1 - x**2) * y - x       # dy/dt

f_func = sp.lambdify((x, y), f_expr, modules="numpy")
g_func = sp.lambdify((x, y), g_expr, modules="numpy")

# ── Equilibria ──
equilibria = find_equilibria(f_expr, g_expr, x, y)
classifications = {}
for eq in equilibria:
    cls, eigs = classify_equilibrium(f_expr, g_expr, x, y, eq)
    classifications[eq] = (cls, eigs)

# ── Plot ──
xlim, ylim = (-5, 5), (-5, 5)
grid_n = 300
qgrid = 30

xs = np.linspace(xlim[0], xlim[1], grid_n)
ys = np.linspace(ylim[0], ylim[1], grid_n)
X, Y = np.meshgrid(xs, ys)
U = f_func(X, Y)
V = g_func(X, Y)
speed = np.sqrt(U**2 + V**2)
speed = np.where(speed == 0, 1e-10, speed)

Xq, Yq = np.meshgrid(np.linspace(xlim[0], xlim[1], qgrid),
                      np.linspace(ylim[0], ylim[1], qgrid))
Uq, Vq = f_func(Xq, Yq), g_func(Xq, Yq)
sq = np.sqrt(Uq**2 + Vq**2)
sq = np.where(sq == 0, 1e-10, sq)

fig, ax = plt.subplots(figsize=(9, 8))

# Nullclines
ax.contour(X, Y, U, levels=[0], colors="limegreen", linewidths=2, linestyles="--")
ax.contour(X, Y, V, levels=[0], colors="magenta", linewidths=2, linestyles="--")
ax.plot([], [], color="limegreen", ls="--", lw=2,
        label=r"x-nullcline ($\dot{x}=0$): $y=0$")
ax.plot([], [], color="magenta", ls="--", lw=2,
        label=r"y-nullcline ($\dot{y}=0$): $y = \frac{x}{\mu(1-x^2)}$")

# Streamplot
norm = Normalize(vmin=sq.min(), vmax=sq.max())
strm = ax.streamplot(Xq, Yq, Uq, Vq, color=sq, cmap="coolwarm",
                     norm=norm, density=1.6, linewidth=0.8,
                     arrowsize=1.2, arrowstyle="->")
fig.colorbar(strm.lines, ax=ax, label="speed")

# Quiver background
ax.quiver(Xq, Yq, Uq / sq, Vq / sq, alpha=0.12, scale=35, width=0.003,
          color="gray")

# ── Limit cycle & sample trajectories ──
def rhs(t, s):
    return [f_func(s[0], s[1]), g_func(s[0], s[1])]

# Trajectory starting far outside → spirals inward to limit cycle
sol_out = solve_ivp(rhs, [0, 40], [4.0, 0.0], max_step=0.02)
ax.plot(sol_out.y[0], sol_out.y[1], color="dodgerblue", lw=1.5,
        alpha=0.7, zorder=3, label="trajectory (from outside)")

# Trajectory starting near origin → spirals outward to limit cycle
sol_in = solve_ivp(rhs, [0, 40], [0.1, 0.1], max_step=0.02)
ax.plot(sol_in.y[0], sol_in.y[1], color="orange", lw=1.5,
        alpha=0.7, zorder=3, label="trajectory (from inside)")

# Extract the limit cycle from the long-time behaviour of the outer trajectory
# (use the last ~1 period worth of data)
T_approx = 2 * np.pi  # rough period estimate
t_mask = sol_out.t > sol_out.t[-1] - 2 * T_approx
ax.plot(sol_out.y[0][t_mask], sol_out.y[1][t_mask],
        color="gold", lw=3, alpha=0.9, zorder=4, label="limit cycle")

# ── Mark equilibria ──
marker_style = {
    "unstable node":   dict(marker="^", color="red"),
    "unstable spiral": dict(marker="^", color="orange"),
    "stable node":     dict(marker="o", color="blue"),
    "stable spiral":   dict(marker="o", color="cyan"),
    "saddle":          dict(marker="s", color="green"),
    "center":          dict(marker="D", color="purple"),
}
for eq, (cls, eigs) in classifications.items():
    style = marker_style.get(cls, dict(marker="x", color="black"))
    ax.plot(eq[0], eq[1], marker=style["marker"], color=style["color"],
            markersize=12, markeredgecolor="black", markeredgewidth=1.5,
            label=f"{cls} ({eq[0]:.1f}, {eq[1]:.1f})", zorder=5)
    eig_str = ", ".join(f"{e:.3g}" for e in eigs)
    ax.annotate(f"{cls}\nλ = {eig_str}",
                xy=eq, xytext=(14, 14), textcoords="offset points",
                fontsize=8, bbox=dict(boxstyle="round,pad=0.3",
                                      fc="white", alpha=0.85),
                arrowprops=dict(arrowstyle="->", color="black"))

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel("x  (displacement)")
ax.set_ylabel("y  (velocity)")
ax.set_aspect("equal")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_title(
    "Van der Pol Oscillator Phase Portrait\n"
    r"$\dot{x} = y$,  $\dot{y} = \mu(1 - x^2)y - x$"
    f"   (μ = {mu})",
    fontsize=11)

plt.tight_layout()
plt.savefig("van_der_pol.png", dpi=150)
plt.show()

# Print summary
print("\n=== Van der Pol Equilibrium Points ===")
for eq, (cls, eigs) in classifications.items():
    eig_str = ", ".join(f"{e:.4g}" for e in eigs)
    print(f"  ({eq[0]:.4f}, {eq[1]:.4f})  →  {cls}  (eigenvalues: {eig_str})")


# ── Limit cycles for different μ values ──
mu_values = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
cmap = plt.cm.viridis
colors = [cmap(i / (len(mu_values) - 1)) for i in range(len(mu_values))]

fig2, ax2 = plt.subplots(figsize=(9, 9))

for mu_val, color in zip(mu_values, colors):
    def rhs_mu(t, s, _mu=mu_val):
        return [s[1], _mu * (1 - s[0]**2) * s[1] - s[0]]

    # Integrate long enough to settle onto the limit cycle
    T_run = 100 if mu_val >= 3.0 else 60
    sol = solve_ivp(rhs_mu, [0, T_run], [3.0, 0.0], max_step=0.01,
                    rtol=1e-9, atol=1e-10)
    # Extract last ~2 periods for a clean cycle
    T_period = 2 * np.pi * (1 + mu_val**2 / 16)  # rough period estimate
    t_mask = sol.t > sol.t[-1] - 2 * T_period
    ax2.plot(sol.y[0][t_mask], sol.y[1][t_mask],
             color=color, lw=2, label=f"μ = {mu_val}")

ax2.plot(0, 0, marker="^", color="orange", markersize=10,
         markeredgecolor="black", markeredgewidth=1.2, zorder=5,
         label="unstable spiral (0, 0)")

ax2.set_xlim(-5, 5)
ax2.set_ylim(-5, 5)
ax2.set_xlabel("x  (displacement)")
ax2.set_ylabel("y  (velocity)")
ax2.set_aspect("equal")
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_title(
    "Van der Pol Limit Cycles for Different μ\n"
    r"$\dot{x} = y$,  $\dot{y} = \mu(1 - x^2)y - x$",
    fontsize=12)

plt.tight_layout()
plt.savefig("van_der_pol_limit_cycles.png", dpi=150)
plt.show()

print("\n=== Limit cycles plotted for μ =", mu_values, "===")
