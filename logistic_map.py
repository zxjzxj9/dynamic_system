"""
Logistic Map — Bifurcation Diagram & Self-Similarity
=====================================================

    x_{n+1} = r x_n (1 - x_n)

The logistic map is a discrete-time dynamical system that exhibits
the period-doubling route to chaos as r increases from ~3 to 4.

Key bifurcation points:
    r = 3.0      period-1 → period-2
    r ≈ 3.449    period-2 → period-4
    r ≈ 3.544    period-4 → period-8
    r ≈ 3.5699   onset of chaos (accumulation point)
    r ≈ 3.8284   period-3 window

Feigenbaum's constant δ ≈ 4.669 governs the geometric convergence
of successive bifurcation intervals.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def compute_bifurcation(r_min, r_max, n_r=10000, n_transient=1000, n_record=300):
    """Compute bifurcation data for the logistic map."""
    r_values = np.linspace(r_min, r_max, n_r)
    # Pre-allocate arrays
    r_all = np.repeat(r_values, n_record)
    x_all = np.empty_like(r_all)

    for i, r in enumerate(r_values):
        x = 0.5  # initial condition (avoid 0 and 1)
        # Transient iterations
        for _ in range(n_transient):
            x = r * x * (1.0 - x)
        # Record steady-state values
        for j in range(n_record):
            x = r * x * (1.0 - x)
            x_all[i * n_record + j] = x

    return r_all, x_all


# ── Bifurcation diagram ──

print("Computing full bifurcation diagram...")
r_all, x_all = compute_bifurcation(2.5, 4.0, n_r=10000)

fig, ax = plt.subplots(figsize=(14, 8))
ax.scatter(r_all, x_all, s=0.005, c="black", alpha=0.15, edgecolors="none")

# Mark key bifurcation points
bif_points = [
    (3.0, "period-1 → 2\nr = 3"),
    (3.449, "period-2 → 4\nr ≈ 3.449"),
    (3.5699, "onset of chaos\nr ≈ 3.570"),
    (3.8284, "period-3 window\nr ≈ 3.828"),
]

for r_bif, label in bif_points:
    ax.axvline(r_bif, color="crimson", lw=0.8, ls="--", alpha=0.6)
    ax.annotate(label, xy=(r_bif, 0.98), xycoords=("data", "axes fraction"),
                fontsize=7.5, color="crimson", ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

# Annotate Feigenbaum's constant
# δ = (r_{n-1} - r_{n-2}) / (r_n - r_{n-1}) → 4.669...
r1 = 3.0
r2 = 3.449490
r3 = 3.544090
delta_approx = (r2 - r1) / (r3 - r2)
ax.annotate(
    f"Feigenbaum's constant\nδ = lim Δrₙ/Δrₙ₊₁ ≈ 4.669\n"
    f"(first ratio: {delta_approx:.3f})",
    xy=(3.45, 0.15), fontsize=9,
    bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="goldenrod", alpha=0.9))

ax.set_xlim(2.5, 4.0)
ax.set_ylim(0, 1)
ax.set_xlabel("r  (growth rate parameter)", fontsize=12)
ax.set_ylabel("x*  (steady-state population)", fontsize=12)
ax.set_title(
    "Logistic Map Bifurcation Diagram\n"
    r"$x_{n+1} = r\, x_n\,(1 - x_n)$",
    fontsize=13)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("logistic_map_bifurcation.png", dpi=150)
print("Saved logistic_map_bifurcation.png")


# ── Self-similarity zoom series ──

zoom_regions = [
    (2.5, 4.0, 0, 1, "Full bifurcation diagram"),
    (3.4, 3.6, 0.3, 0.95, "Period-doubling cascade"),
    (3.82, 3.86, 0.42, 0.6, "Period-3 window"),
    (3.845, 3.857, 0.44, 0.56, "Period-3 cascade (deep zoom)"),
]

# Rectangle colors linking each panel to the next zoom
rect_colors = ["dodgerblue", "forestgreen", "darkorange"]

print("Computing self-similarity zoom panels...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, (r_min, r_max, x_lo, x_hi, title) in enumerate(zoom_regions):
    ax = axes[idx]
    print(f"  Panel {idx + 1}: r ∈ [{r_min}, {r_max}]...")
    r_data, x_data = compute_bifurcation(r_min, r_max, n_r=8000,
                                          n_transient=1200, n_record=400)
    ax.scatter(r_data, x_data, s=0.01, c="black", alpha=0.12, edgecolors="none")

    ax.set_xlim(r_min, r_max)
    ax.set_ylim(x_lo, x_hi)
    ax.set_xlabel("r", fontsize=10)
    ax.set_ylabel("x*", fontsize=10)
    ax.set_title(f"{title}\nr ∈ [{r_min}, {r_max}]", fontsize=11)
    ax.grid(True, alpha=0.2)

    # Draw rectangle showing next zoom region
    if idx < len(zoom_regions) - 1:
        nr_min, nr_max, nx_lo, nx_hi, _ = zoom_regions[idx + 1]
        # Only draw if next region is visible in current panel
        if nr_min >= r_min and nr_max <= r_max and nx_lo >= x_lo and nx_hi <= x_hi:
            rect = Rectangle((nr_min, nx_lo), nr_max - nr_min, nx_hi - nx_lo,
                              linewidth=2, edgecolor=rect_colors[idx],
                              facecolor=rect_colors[idx], alpha=0.12)
            ax.add_patch(rect)
            ax.plot([nr_min, nr_min, nr_max, nr_max, nr_min],
                    [nx_lo, nx_hi, nx_hi, nx_lo, nx_lo],
                    color=rect_colors[idx], lw=2, ls="--")

plt.suptitle(
    "Self-Similarity in the Logistic Map\n"
    "Each panel zooms into the highlighted region of the previous panel",
    fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("logistic_map_self_similarity.png", dpi=150, bbox_inches="tight")
print("Saved logistic_map_self_similarity.png")

plt.show()

# Print summary
print("\n=== Logistic Map Summary ===")
print("  x_{n+1} = r * x_n * (1 - x_n)")
print(f"\n  Key bifurcation points:")
for r_bif, label in bif_points:
    print(f"    r = {r_bif:<8.4f}  {label.replace(chr(10), ', ')}")
print(f"\n  Feigenbaum's constant δ ≈ 4.669")
print(f"  First ratio approximation: (r2-r1)/(r3-r2) = {delta_approx:.4f}")
