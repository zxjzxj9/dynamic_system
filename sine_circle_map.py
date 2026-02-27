"""
Sine-Circle Map — Arnold Tongues and the Devil's Staircase
============================================================

    θ_{n+1} = θ_n + Ω − (K / 2π) sin(2π θ_n)   (mod 1)

A map of the circle to itself, parameterized by the bare rotation
number Ω and nonlinearity (coupling) K. At K = 0 it reduces to
rigid rotation by Ω. As K increases, mode-locking regions (Arnold
tongues) grow around every rational Ω. At the critical value K = 1,
the map becomes non-invertible and the tongues fill almost all of
parameter space — the winding number as a function of Ω becomes
a devil's staircase (a continuous, non-decreasing function that is
locally constant almost everywhere).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def winding_number(Omega, K, n_transient=1000, n_record=2000):
    """Compute the winding number W for given Ω and K (vectorized).

    Works with scalar or array inputs for Omega and K.
    Uses unwrapped angle to measure cumulative rotation.
    """
    theta = np.zeros_like(Omega, dtype=float)
    # Transient
    for _ in range(n_transient):
        theta = theta + Omega - K / (2 * np.pi) * np.sin(2 * np.pi * theta)
    theta_start = theta.copy()
    # Record
    for _ in range(n_record):
        theta = theta + Omega - K / (2 * np.pi) * np.sin(2 * np.pi * theta)
    W = (theta - theta_start) / n_record
    return W


# ══════════════════════════════════════════════════════════════════
# Figure 1: Devil's Staircase — Winding Number vs Ω
# ══════════════════════════════════════════════════════════════════

print("Computing devil's staircase (winding number vs Ω)...")

n_omega = 4000
Omega_vals = np.linspace(0, 1, n_omega)

K_curves = [
    (0.0,  "C7", "--", "K = 0  (rigid rotation)"),
    (0.5,  "C0", "-",  "K = 0.5"),
    (0.8,  "C2", "-",  "K = 0.8"),
    (1.0,  "C3", "-",  "K = 1.0  (critical, devil's staircase)"),
]

fig, ax = plt.subplots(figsize=(12, 8))

for K, color, ls, label in K_curves:
    print(f"  {label}...")
    K_arr = np.full_like(Omega_vals, K)
    W = winding_number(Omega_vals, K_arr)
    ax.plot(Omega_vals, W, color=color, ls=ls, lw=1.2, label=label)

# Mark some prominent plateaus at K=1
ax.axhline(0.5, color="gray", lw=0.5, alpha=0.3)
ax.axhline(1/3, color="gray", lw=0.5, alpha=0.3)
ax.axhline(2/3, color="gray", lw=0.5, alpha=0.3)

ax.set_xlabel(r"Bare rotation number $\Omega$", fontsize=12)
ax.set_ylabel(r"Winding number $W$", fontsize=12)
ax.set_title(
    "Sine-Circle Map — Devil's Staircase\n"
    r"$\theta_{n+1} = \theta_n + \Omega - \frac{K}{2\pi}"
    r"\sin(2\pi\theta_n)\ (\mathrm{mod}\ 1)$",
    fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("sine_circle_staircase.png", dpi=150)
print("Saved sine_circle_staircase.png")


# ══════════════════════════════════════════════════════════════════
# Figure 2: Arnold Tongues in (Ω, K) parameter space
# ══════════════════════════════════════════════════════════════════

print("Computing Arnold tongues (Ω–K parameter space)...")

n_omega_grid = 800
n_K_grid = 600

Omega_grid = np.linspace(0, 1, n_omega_grid)
K_grid = np.linspace(0, 1.5, n_K_grid)
OO, KK = np.meshgrid(Omega_grid, K_grid)

W_map = winding_number(OO, KK, n_transient=500, n_record=1000)

fig, ax = plt.subplots(figsize=(12, 8))

# Use a cyclic-friendly colormap; hsv wraps nicely for winding number
im = ax.imshow(W_map, extent=[0, 1, 0, 1.5], origin="lower",
               aspect="auto", cmap="hsv",
               norm=mcolors.Normalize(vmin=0, vmax=1),
               interpolation="nearest")

cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Winding number W", fontsize=11)

# Mark the critical line K = 1
ax.axhline(1.0, color="white", lw=1.5, ls="--", alpha=0.8)
ax.annotate("K = 1 (critical)", xy=(0.02, 1.02),
            fontsize=10, color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))

# Label some prominent tongues
for W_label, x_pos in [(r"$\frac{1}{2}$", 0.5),
                        (r"$\frac{1}{3}$", 0.333),
                        (r"$\frac{2}{3}$", 0.667)]:
    ax.annotate(f"W = {W_label}", xy=(x_pos, 0.05),
                fontsize=9, color="white", ha="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5))

ax.set_xlabel(r"Bare rotation number $\Omega$", fontsize=12)
ax.set_ylabel("Coupling strength K", fontsize=12)
ax.set_title(
    "Sine-Circle Map — Arnold Tongues\n"
    r"Winding number $W(\Omega, K)$ in parameter space",
    fontsize=13)

plt.tight_layout()
plt.savefig("sine_circle_tongues.png", dpi=150)
print("Saved sine_circle_tongues.png")

plt.show()

print("\n=== Sine-Circle Map Summary ===")
print("  θ_{n+1} = θ_n + Ω − (K/2π) sin(2πθ_n)  (mod 1)")
print("  K = 0: rigid rotation, W = Ω (diagonal line)")
print("  0 < K < 1: Arnold tongues grow around rational Ω values")
print("  K = 1 (critical): devil's staircase — tongues fill almost all Ω")
print("  K > 1: map non-invertible, tongues overlap → chaotic regions")
