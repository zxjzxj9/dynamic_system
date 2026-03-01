"""
Lotka–Volterra Predator-Prey Phase Portrait
============================================

    dx/dt = αx  - βxy   =  x(α - βy)     (prey)
    dy/dt = δxy - γy     =  y(δx - γ)     (predator)

Parameters used:  α = 1, β = 1.5, δ = 1.25, γ = 1
Equilibria:
    (0, 0)              — saddle
    (γ/δ, α/β) = (0.8, 0.667) — center  (conserved quantity → closed orbits)
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.integrate import solve_ivp

from phase_portrait import (find_equilibria, classify_equilibrium)


# ── Parameters ──
alpha, beta, delta, gamma_ = 1.0, 1.5, 1.25, 1.0

x, y = sp.symbols("x y")
f_expr = alpha * x - beta * x * y      # prey  dx/dt
g_expr = delta * x * y - gamma_ * y    # predator  dy/dt

f_func = sp.lambdify((x, y), f_expr, modules="numpy")
g_func = sp.lambdify((x, y), g_expr, modules="numpy")

# ── Equilibria ──
equilibria = find_equilibria(f_expr, g_expr, x, y)
classifications = {}
for eq in equilibria:
    cls, eigs = classify_equilibrium(f_expr, g_expr, x, y, eq)
    classifications[eq] = (cls, eigs)

# ── Conserved quantity  H(x,y) = δx - γ ln(x) + βy - α ln(y) ──
def H(xv, yv):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (delta * xv - gamma_ * np.log(xv)
                + beta * yv - alpha * np.log(yv))

# ── Plot ──
xlim, ylim = (0.05, 4), (0.05, 4)
grid_n = 300

xs = np.linspace(xlim[0], xlim[1], grid_n)
ys = np.linspace(ylim[0], ylim[1], grid_n)
X, Y = np.meshgrid(xs, ys)
U = f_func(X, Y)
V = g_func(X, Y)
speed = np.sqrt(U**2 + V**2)
speed = np.where(speed == 0, 1e-10, speed)

fig, ax = plt.subplots(figsize=(9, 8))

# Closed orbits via conserved quantity contours
Hval = H(X, Y)
# Choose contour levels from orbits passing through specific initial conditions
h_levels = sorted(set(
    H(x0, alpha / beta) for x0 in [0.2, 0.4, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0]
))
ax.contour(X, Y, Hval, levels=h_levels, colors="slategray",
           linewidths=0.9, alpha=0.6)

# Nullclines
ax.contour(X, Y, U, levels=[0], colors="limegreen", linewidths=2, linestyles="--")
ax.contour(X, Y, V, levels=[0], colors="magenta", linewidths=2, linestyles="--")
ax.plot([], [], color="limegreen", ls="--", lw=2,
        label=r"prey nullcline ($\dot{x}=0$): $y = \alpha/\beta$")
ax.plot([], [], color="magenta", ls="--", lw=2,
        label=r"predator nullcline ($\dot{y}=0$): $x = \gamma/\delta$")

# Streamplot
qgrid = 30
Xq, Yq = np.meshgrid(np.linspace(xlim[0], xlim[1], qgrid),
                      np.linspace(ylim[0], ylim[1], qgrid))
Uq, Vq = f_func(Xq, Yq), g_func(Xq, Yq)
sq = np.sqrt(Uq**2 + Vq**2)
sq = np.where(sq == 0, 1e-10, sq)

norm = Normalize(vmin=sq.min(), vmax=sq.max())
strm = ax.streamplot(Xq, Yq, Uq, Vq, color=sq, cmap="coolwarm",
                     norm=norm, density=1.6, linewidth=0.8,
                     arrowsize=1.2, arrowstyle="->")
fig.colorbar(strm.lines, ax=ax, label="speed")

# Trajectory examples — integrate a few orbits to show direction
eq_x, eq_y = gamma_ / delta, alpha / beta
for x0_val in [0.2, 0.5, 1.2, 2.0]:
    sol = solve_ivp(lambda t, s: [f_func(s[0], s[1]), g_func(s[0], s[1])],
                    [0, 30], [x0_val, eq_y], max_step=0.05)
    ax.plot(sol.y[0], sol.y[1], color="gold", lw=1.8, alpha=0.8,
            zorder=3)

# Mark equilibria
marker_style = {
    "saddle": dict(marker="s", color="green"),
    "center": dict(marker="D", color="purple"),
}
for eq, (cls, eigs) in classifications.items():
    # Skip origin for visual clarity (it's at the axes boundary)
    if eq[0] < xlim[0] or eq[1] < ylim[0]:
        continue
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
ax.set_xlabel("x  (prey population)")
ax.set_ylabel("y  (predator population)")
ax.set_aspect("equal")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_title(
    "Lotka–Volterra Predator-Prey Phase Portrait\n"
    r"$\dot{x} = \alpha x - \beta xy$,  "
    r"$\dot{y} = \delta xy - \gamma y$"
    f"   (α={alpha}, β={beta}, δ={delta}, γ={gamma_})",
    fontsize=11)

plt.tight_layout()
plt.savefig("lotka_volterra.png", dpi=150)
plt.show()

# Print summary
print("\n=== Lotka–Volterra Equilibrium Points ===")
for eq, (cls, eigs) in classifications.items():
    eig_str = ", ".join(f"{e:.4g}" for e in eigs)
    print(f"  ({eq[0]:.4f}, {eq[1]:.4f})  →  {cls}  (eigenvalues: {eig_str})")
