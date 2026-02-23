"""
Standard (Chirikov) Map — Poincaré-Birkhoff Theorem
====================================================

    y_{n+1} = y_n + (K / 2π) sin(2π x_n)   (mod 1)
    x_{n+1} = x_n + y_{n+1}                 (mod 1)

An area-preserving twist map on the torus [0,1) × [0,1).
At K = 0 the map is integrable (horizontal lines y = const).
As K increases, rational tori break up into chains of islands
(alternating elliptic and hyperbolic fixed points) — the
Poincaré-Birkhoff theorem. Irrational tori persist as KAM
curves until K becomes large enough to destroy them.

Classic parameter: K ≈ 0.8 shows KAM tori, island chains,
and thin chaotic layers coexisting.
"""

import numpy as np
import matplotlib.pyplot as plt

K = 0.8

# ── Iterate the standard map ──

def standard_map(x0, y0, K, n_iter):
    """Iterate the standard map on [0,1) × [0,1)."""
    xs = np.empty(n_iter)
    ys = np.empty(n_iter)
    x, y = x0, y0
    for i in range(n_iter):
        y = (y + K / (2.0 * np.pi) * np.sin(2.0 * np.pi * x)) % 1.0
        x = (x + y) % 1.0
        xs[i] = x
        ys[i] = y
    return xs, ys


# ── Generate orbits ──

print(f"Computing standard map orbits (K = {K})...")

fig, ax = plt.subplots(figsize=(10, 10))
cmap = plt.cm.coolwarm

# Main sweep: uniform y₀ values along x₀ = 0
n_orbits = 300
n_iter = 3000
y0_values = np.linspace(0.0, 1.0, n_orbits, endpoint=False)

for y0 in y0_values:
    xs, ys = standard_map(0.0, y0, K, n_iter)
    color = cmap(y0)
    ax.scatter(xs, ys, s=0.1, c=[color], alpha=0.6, edgecolors="none")

# Extra orbits seeded near resonances to fill island chains
# Period-1 fixed point is at (0, 0); nearby islands at y ≈ 0 (mod 1)
# Period-1/2 resonance at y ≈ 0.5; period-1/3 at y ≈ 1/3, 2/3
resonance_seeds = []
for y_center in [0.0, 1/3, 0.5, 2/3]:
    for dy in np.linspace(-0.04, 0.04, 20):
        for dx in np.linspace(-0.04, 0.04, 5):
            resonance_seeds.append((dx % 1.0, (y_center + dy) % 1.0))

for x0, y0 in resonance_seeds:
    xs, ys = standard_map(x0, y0, K, 4000)
    color = cmap(y0)
    ax.scatter(xs, ys, s=0.1, c=[color], alpha=0.6, edgecolors="none")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
ax.set_title(
    "Standard (Chirikov) Map — Poincaré-Birkhoff Theorem\n"
    r"$y_{n+1} = y_n + \frac{K}{2\pi}\sin(2\pi x_n)\ (\mathrm{mod}\ 1),"
    r"\quad x_{n+1} = x_n + y_{n+1}\ (\mathrm{mod}\ 1)$"
    f"\n(K = {K})",
    fontsize=13)
ax.set_aspect("equal")
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("standard_map.png", dpi=150)
print("Saved standard_map.png")

plt.show()

print("\n=== Standard Map Summary ===")
print(f"  K = {K}")
print("  KAM tori: smooth curves (irrational winding numbers survive)")
print("  Island chains: broken rational tori → elliptic + hyperbolic points")
print("  Chaotic layers: thin stochastic regions near hyperbolic points")
