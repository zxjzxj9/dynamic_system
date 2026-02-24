"""
Kicked Rotator — Resonance Overlap and the Onset of Chaos
==========================================================

    H = p²/2 + K cos(θ) Σ_n δ(t − n)

The kicked rotator is a periodically kicked free rotor. Its
stroboscopic Poincaré section (sampled once per kick) is the
standard (Chirikov) map:

    p_{n+1} = p_n + (K / 2π) sin(2π θ_n)   (mod 1)
    θ_{n+1} = θ_n + p_{n+1}                 (mod 1)

As the kick strength K increases, primary resonance islands
grow and eventually overlap (Chirikov criterion), destroying
the KAM tori that separate them. The critical value
K_c ≈ 0.9716 marks the breakup of the last KAM barrier and
the onset of global chaos (unbounded diffusion in momentum).
"""

import numpy as np
import matplotlib.pyplot as plt


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


# ── Parameters for each panel ──

K_values = [
    (0.0, "K = 0  (integrable)"),
    (0.5, "K = 0.5"),
    (0.8, "K = 0.8"),
    (0.9716, r"K = 0.9716  (critical, $K_c$)"),
    (1.5, "K = 1.5"),
    (3.0, "K = 3.0"),
]

n_orbits = 200
n_iter = 2000
cmap = plt.cm.coolwarm

# Resonance seeds near rational winding numbers
resonance_seeds = []
for y_center in [0.0, 1/3, 0.5, 2/3]:
    for dy in np.linspace(-0.04, 0.04, 15):
        for dx in np.linspace(-0.04, 0.04, 4):
            resonance_seeds.append((dx % 1.0, (y_center + dy) % 1.0))

# ── Generate figure ──

print("Computing kicked rotator phase portraits...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, (K, label) in enumerate(K_values):
    ax = axes[idx]
    print(f"  Panel {idx + 1}: {label}...")

    y0_values = np.linspace(0.0, 1.0, n_orbits, endpoint=False)

    # Main sweep
    for y0 in y0_values:
        xs, ys = standard_map(0.0, y0, K, n_iter)
        color = cmap(y0)
        ax.scatter(xs, ys, s=0.08, c=[color], alpha=0.5, edgecolors="none")

    # Resonance seeds (skip for K=0, nothing interesting happens)
    if K > 0:
        for x0, y0 in resonance_seeds:
            xs, ys = standard_map(x0, y0, K, 3000)
            color = cmap(y0)
            ax.scatter(xs, ys, s=0.08, c=[color], alpha=0.5, edgecolors="none")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(label, fontsize=12)
    ax.grid(True, alpha=0.2)

    if idx >= 3:
        ax.set_xlabel(r"$\theta$", fontsize=11)
    if idx % 3 == 0:
        ax.set_ylabel("p", fontsize=11)

fig.suptitle(
    "Kicked Rotator — Resonance Overlap and the Onset of Chaos\n"
    r"$H = \frac{p^2}{2} + K\cos(\theta)\,\sum_n \delta(t - n)$"
    r"$\qquad$Stroboscopic map: "
    r"$p_{n+1} = p_n + \frac{K}{2\pi}\sin(2\pi\theta_n),\;"
    r"\theta_{n+1} = \theta_n + p_{n+1}$",
    fontsize=13, y=1.01)

plt.tight_layout()
plt.savefig("kicked_rotator.png", dpi=150, bbox_inches="tight")
print("Saved kicked_rotator.png")

plt.show()

print("\n=== Kicked Rotator Summary ===")
print("  Increasing K grows resonance islands until they overlap.")
print("  K_c ≈ 0.9716: last KAM torus breaks → global chaos.")
print("  Below K_c: chaotic orbits confined between KAM barriers.")
print("  Above K_c: momentum diffuses freely through phase space.")
