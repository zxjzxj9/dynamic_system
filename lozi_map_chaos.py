"""
Lozi Map — Orbit Structure and Chaos
=====================================

    x_{n+1} = 1 − a|x_n| + b y_n
    y_{n+1} = x_n

A piecewise-linear 2D discrete map that produces a strange attractor
similar in spirit to the Hénon map, but with sharp folds instead of
smooth curves due to the absolute value nonlinearity.

Classic chaotic parameters: a = 1.7, b = 0.5
"""

import numpy as np
import matplotlib.pyplot as plt


def lozi_iterate(x0, y0, a, b, n_iter):
    """Iterate the Lozi map, returning full trajectory."""
    xs = np.empty(n_iter)
    ys = np.empty(n_iter)
    x, y = x0, y0
    for i in range(n_iter):
        xs[i] = x
        ys[i] = y
        x_new = 1.0 - a * abs(x) + b * y
        y_new = x
        x, y = x_new, y_new
    return xs, ys


a = 1.7
b = 0.5

# ── Figure 1: Strange Attractor ──

print("Computing strange attractor...")
n_total = 51000
n_transient = 1000
xs, ys = lozi_iterate(0.0, 0.0, a, b, n_total)

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(xs[n_transient:], ys[n_transient:], s=0.08, c="black",
           alpha=0.5, edgecolors="none")
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
ax.set_title(
    "Lozi Map — Strange Attractor\n"
    r"$x_{n+1} = 1 - a\,|x_n| + b\,y_n, \quad y_{n+1} = x_n$"
    f"\n(a = {a}, b = {b})",
    fontsize=13)
ax.set_aspect("equal")
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("lozi_attractor.png", dpi=150)
print("Saved lozi_attractor.png")


# ── Figure 2: Sensitive Dependence on Initial Conditions ──

print("Computing sensitivity to initial conditions...")
eps = 1e-8
n_sens = 60

xs1, _ = lozi_iterate(0.0, 0.0, a, b, n_sens)
xs2, _ = lozi_iterate(eps, 0.0, a, b, n_sens)
separation = np.abs(xs1 - xs2)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ns = np.arange(n_sens)
ax1.plot(ns, xs1, "b-", lw=1.2, label="x₀ = 0")
ax1.plot(ns, xs2, "r--", lw=1.2, label=f"x₀ = {eps}")
ax1.set_ylabel("xₙ", fontsize=12)
ax1.set_title(
    "Sensitive Dependence on Initial Conditions\n"
    f"Two orbits with Δx₀ = {eps}",
    fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.2)

ax2.semilogy(ns, np.maximum(separation, 1e-16), "k-", lw=1.2)
ax2.set_xlabel("Iteration n", fontsize=12)
ax2.set_ylabel("|Δxₙ|", fontsize=12)
ax2.set_title("Separation between orbits (log scale)", fontsize=11)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("lozi_sensitivity.png", dpi=150)
print("Saved lozi_sensitivity.png")


# ── Figure 3: Bifurcation Diagram (sweep a, b = 0.5 fixed) ──

print("Computing bifurcation diagram (sweeping a)...")
a_values = np.concatenate([
    np.linspace(0.0, 1.4, 1500),
    np.linspace(1.4, 1.8, 3000),   # higher resolution in chaotic regime
])
n_transient_bif = 2000
n_record = 300
b_fixed = 0.5
bound = 100  # generous bound; clip display later

# Collect bounded orbits only
a_list = []
x_list = []

for i, av in enumerate(a_values):
    x, y = 0.1, 0.1
    diverged = False
    for _ in range(n_transient_bif):
        x_new = 1.0 - av * abs(x) + b_fixed * y
        y_new = x
        x, y = x_new, y_new
        if abs(x) > bound:
            diverged = True
            break
    if diverged:
        continue
    for j in range(n_record):
        x_new = 1.0 - av * abs(x) + b_fixed * y
        y_new = x
        x, y = x_new, y_new
        if abs(x) > bound:
            break
        a_list.append(av)
        x_list.append(x)

a_all = np.array(a_list)
x_all = np.array(x_list)

fig, ax = plt.subplots(figsize=(14, 8))
ax.scatter(a_all, x_all, s=0.02, c="black", alpha=0.25, edgecolors="none")
ax.axvline(1.7, color="crimson", lw=1.2, ls="--", alpha=0.7)
ax.annotate("a = 1.7\n(attractor shown above)",
            xy=(1.7, 0.98), xycoords=("data", "axes fraction"),
            fontsize=9, color="crimson", ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

ax.set_xlim(0.0, 1.8)
ax.set_ylim(-2, 2.5)
ax.set_xlabel("a  (nonlinearity parameter)", fontsize=12)
ax.set_ylabel("x*  (steady-state values)", fontsize=12)
ax.set_title(
    "Lozi Map — Bifurcation Diagram\n"
    r"$x_{n+1} = 1 - a\,|x_n| + b\,y_n, \quad y_{n+1} = x_n$"
    f"  (b = {b_fixed} fixed)",
    fontsize=13)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("lozi_bifurcation.png", dpi=150)
print("Saved lozi_bifurcation.png")

plt.show()

print("\n=== Lozi Map Summary ===")
print("  x_{n+1} = 1 − a|x_n| + b y_n")
print("  y_{n+1} = x_n")
print(f"\n  Parameters: a = {a}, b = {b}")
print("  The piecewise-linear absolute value creates sharp folds,")
print("  producing a strange attractor with fractal-like banded structure.")
