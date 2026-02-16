"""
Homoclinic Orbit — Nonlinear Spring
====================================

    dx/dt = y
    dy/dt = -(k/m)(1 + x)x

This is a conservative (Hamiltonian) system with energy:
    H(x,y) = y²/2 + (k/m)(x²/2 + x³/3)

Parameters used:  k/m = 1
Equilibria:
    (0, 0)  — center  (eigenvalues ±i)
    (-1, 0) — saddle  (eigenvalues ±1)

The homoclinic orbit is the level curve H(x,y) = H(-1,0) = 1/6,
a trajectory that departs from the saddle and returns to it as t → ±∞.
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.integrate import solve_ivp

from phase_portrait import find_equilibria, classify_equilibrium


# ── Parameters ──
km = 1.0  # k/m

x, y = sp.symbols("x y")
f_expr = y                          # dx/dt
g_expr = -km * (1 + x) * x         # dy/dt

f_func = sp.lambdify((x, y), f_expr, modules="numpy")
g_func = sp.lambdify((x, y), g_expr, modules="numpy")

# ── Equilibria ──
equilibria = find_equilibria(f_expr, g_expr, x, y)
classifications = {}
for eq in equilibria:
    cls, eigs = classify_equilibrium(f_expr, g_expr, x, y, eq)
    classifications[eq] = (cls, eigs)

# ── Conserved energy ──
def H(xv, yv):
    return 0.5 * yv**2 + km * (0.5 * xv**2 + xv**3 / 3.0)

H_saddle = H(-1.0, 0.0)  # = km / 6

# ── Plot ──
xlim, ylim = (-2.5, 2.5), (-2.5, 2.5)
grid_n = 400
qgrid = 28

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

fig, ax = plt.subplots(figsize=(9, 9))

# Energy contours (closed orbits inside, open orbits outside)
Hval = H(X, Y)
# Levels inside the homoclinic orbit (bound orbits around center)
inner_levels = np.linspace(0.01, H_saddle - 0.005, 8)
# Levels outside the homoclinic orbit (unbounded)
outer_levels = np.linspace(H_saddle + 0.03, 1.2, 6)
ax.contour(X, Y, Hval, levels=inner_levels, colors="slategray",
           linewidths=0.7, alpha=0.5)
ax.contour(X, Y, Hval, levels=outer_levels, colors="slategray",
           linewidths=0.7, alpha=0.5)

# Nullclines
ax.contour(X, Y, U, levels=[0], colors="limegreen", linewidths=2, linestyles="--")
ax.contour(X, Y, V, levels=[0], colors="magenta", linewidths=2, linestyles="--")
ax.plot([], [], color="limegreen", ls="--", lw=2,
        label=r"x-nullcline ($\dot{x}=0$): $y=0$")
ax.plot([], [], color="magenta", ls="--", lw=2,
        label=r"y-nullcline ($\dot{y}=0$): $x=0,\;x=-1$")

# Streamplot
norm = Normalize(vmin=sq.min(), vmax=sq.max())
strm = ax.streamplot(Xq, Yq, Uq, Vq, color=sq, cmap="coolwarm",
                     norm=norm, density=1.4, linewidth=0.7,
                     arrowsize=1.2, arrowstyle="->")
fig.colorbar(strm.lines, ax=ax, label="speed")

# ── Homoclinic orbit ──
# The homoclinic orbit is H(x,y) = H_saddle = km/6.
# Solve for y: y = ±sqrt(2(H_saddle - km*(x²/2 + x³/3)))
x_hom = np.linspace(-2.0, -1e-6, 2000)
inside = 2.0 * (H_saddle - km * (0.5 * x_hom**2 + x_hom**3 / 3.0))
inside = np.clip(inside, 0, None)
y_hom = np.sqrt(inside)

# Plot the upper and lower branches
ax.plot(x_hom, y_hom, color="gold", lw=3.5, zorder=4, label="homoclinic orbit")
ax.plot(x_hom, -y_hom, color="gold", lw=3.5, zorder=4)

# Also verify with numerical integration: start just off the saddle along
# the unstable manifold
J = sp.Matrix([
    [sp.diff(f_expr, x), sp.diff(f_expr, y)],
    [sp.diff(g_expr, x), sp.diff(g_expr, y)],
])
J_at_saddle = J.subs({x: -1, y: 0})
eigenpairs = J_at_saddle.eigenvects()
for eigenval, mult, vecs in eigenpairs:
    ev = complex(eigenval)
    if ev.real > 0:
        unstable_vec = np.array([complex(vecs[0][0]),
                                  complex(vecs[0][1])]).real
        unstable_vec = unstable_vec / np.linalg.norm(unstable_vec)

eps = 1e-5
for sign in [1, -1]:
    x0 = -1.0 + sign * eps * unstable_vec[0]
    y0 = 0.0 + sign * eps * unstable_vec[1]

    def rhs(t, s):
        return [f_func(s[0], s[1]), g_func(s[0], s[1])]

    def near_saddle(t, s):
        return (s[0] + 1)**2 + s[1]**2 - 0.01
    near_saddle.terminal = True
    near_saddle.direction = -1  # trigger when approaching saddle

    sol = solve_ivp(rhs, [0, 50], [x0, y0], max_step=0.01,
                    rtol=1e-10, atol=1e-12, events=near_saddle)
    ax.plot(sol.y[0], sol.y[1], color="gold", lw=2, ls="--",
            alpha=0.6, zorder=4)

# ── Mark equilibria ──
marker_style = {
    "stable node":     dict(marker="o", color="blue"),
    "stable spiral":   dict(marker="o", color="cyan"),
    "unstable node":   dict(marker="^", color="red"),
    "unstable spiral": dict(marker="^", color="orange"),
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
    "Homoclinic Orbit — Nonlinear Spring\n"
    r"$\dot{x} = y$,  $\dot{y} = -(k/m)(1+x)x$"
    f"   (k/m = {km})",
    fontsize=11)

plt.tight_layout()
plt.savefig("homoclinic.png", dpi=150)
plt.show()

# Print summary
print("\n=== Equilibrium Points ===")
for eq, (cls, eigs) in classifications.items():
    eig_str = ", ".join(f"{e:.4g}" for e in eigs)
    print(f"  ({eq[0]:.4f}, {eq[1]:.4f})  →  {cls}  (eigenvalues: {eig_str})")
print(f"\nHomoclinic orbit energy: H = {H_saddle:.6f}")
