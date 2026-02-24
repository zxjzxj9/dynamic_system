"""
Kicked Harmonic Oscillator — Stochastic Web
=============================================

    H = p²/2 + ω²x²/2 + K cos(x) Σ_n δ(t − nT)

A harmonic oscillator receiving periodic kicks. When the oscillator
frequency is resonant with the kick period (ωT = 2π/q for integer q),
the stroboscopic map produces a q-fold symmetric "stochastic web" —
a lattice of thin chaotic channels threading through phase space.

For q = 4 (quarter-turn resonance), the stroboscopic map is:
first apply the kick  p → p + K sin(x),
then free-rotate by π/2: (x, p) → (p, −x).

    x_{n+1} = p_n + K sin(x_n)
    p_{n+1} = −x_n

The web has square lattice symmetry, with chaotic channels forming
a grid that extends to infinity — enabling unbounded diffusion.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


K = 3.0

# ── Build diffusion map on a grid of initial conditions ──
# For each (x₀, p₀), iterate the map and measure the maximum
# displacement from the start. Points on the web diffuse far;
# points in cells stay trapped.

print(f"Computing stochastic web diffusion map (K = {K})...")

R = 14  # display range
res = 1000  # grid resolution
n_iter = 300  # iterations per point

x_grid = np.linspace(-R, R, res)
p_grid = np.linspace(-R, R, res)
X0, P0 = np.meshgrid(x_grid, p_grid)

# Iterate all grid points simultaneously (vectorized)
X = X0.copy()
P = P0.copy()
max_disp = np.zeros_like(X)

for _ in range(n_iter):
    X_new = P + K * np.sin(X)
    P_new = -X
    X, P = X_new, P_new
    disp = np.sqrt((X - X0)**2 + (P - P0)**2)
    max_disp = np.maximum(max_disp, disp)

# ── Plot ──

fig, ax = plt.subplots(figsize=(12, 12))

# Log scale for displacement; clamp to avoid log(0)
log_disp = np.log10(np.maximum(max_disp, 1e-6))

# Reversed inferno: dark = high diffusion (web), bright = trapped (cells)
im = ax.imshow(log_disp, extent=[-R, R, -R, R], origin="lower",
               cmap="inferno_r", aspect="equal",
               norm=mcolors.Normalize(vmin=0, vmax=2.5))

cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label(r"$\log_{10}$(max displacement)", fontsize=11)

ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("p", fontsize=12)
ax.set_title(
    "Kicked Harmonic Oscillator — Stochastic Web (q = 4)\n"
    r"$x_{n+1} = p_n + K\sin(x_n), \quad p_{n+1} = -x_n$"
    f"\n(K = {K})",
    fontsize=13)

plt.tight_layout()
plt.savefig("kicked_harmonic_web.png", dpi=150)
print("Saved kicked_harmonic_web.png")

plt.show()

print("\n=== Kicked Harmonic Oscillator Summary ===")
print(f"  K = {K}, q = 4 (square lattice symmetry)")
print("  Bright channels: stochastic web (chaotic diffusion)")
print("  Dark cells: regular islands (trapped KAM tori)")
print("  Web enables unbounded transport through phase space")
