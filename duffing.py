"""
Forced Duffing Oscillator — Route to Chaos
============================================

    ẍ + δẋ − x + x³ = a·sin(ωt)

As a first-order system:
    ẋ = y
    ẏ = −δy + x − x³ + a·sin(ωt)

The double-well potential V(x) = −x²/2 + x⁴/4 has minima at x = ±1.
Without forcing (a=0), the system has simple periodic orbits in each well.
As forcing amplitude a increases, the orbit undergoes a boundary crisis —
it escapes from one well and begins wandering chaotically between both.
Periodic windows reappear within the chaotic regime.

Parameters: δ = 0.25 (damping), ω = 1.0 (driving frequency)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ── Parameters ──

delta = 0.25   # damping
omega = 1.0    # driving frequency
T = 2 * np.pi / omega  # driving period

# Initial condition — start in the right well
x0, y0 = 1.0, 0.0


def duffing(t, state, a):
    """Duffing oscillator right-hand side."""
    x, y = state
    dxdt = y
    dydt = -delta * y + x - x**3 + a * np.sin(omega * t)
    return [dxdt, dydt]


# ── Figure 1: Phase Portraits (2×2 grid) ──

print("Generating phase portraits...")

amplitudes = [0.00, 0.25, 0.30, 0.50]
titles = [
    r"$a = 0.00$ — unforced (well orbit)",
    r"$a = 0.25$ — period-1 (single well)",
    r"$a = 0.30$ — chaos (inter-well crisis)",
    r"$a = 0.50$ — periodic window (large orbit)",
]

n_transient = 200   # driving periods to discard
n_collect = 300     # driving periods to collect

fig1, axes = plt.subplots(2, 2, figsize=(14, 12))

for idx, (a, title) in enumerate(zip(amplitudes, titles)):
    ax = axes[idx // 2, idx % 2]
    print(f"  a = {a:.2f} ...")

    t_end = (n_transient + n_collect) * T
    sol = solve_ivp(
        lambda t, s: duffing(t, s, a),
        (0.0, t_end), [x0, y0],
        method="RK45", max_step=0.01,
        rtol=1e-9, atol=1e-12,
        dense_output=True,
    )

    # Discard transient
    t_start = n_transient * T
    mask = sol.t >= t_start
    t_plot = sol.t[mask]
    x_plot = sol.y[0, mask]
    y_plot = sol.y[1, mask]

    # Phase trajectory
    ax.plot(x_plot, y_plot, lw=0.3, alpha=0.7, color="steelblue")

    # Stroboscopic Poincaré section: sample at t = n·T
    n_start = n_transient
    n_end = n_transient + n_collect
    t_strobe = np.arange(n_start, n_end + 1) * T
    strobe_vals = sol.sol(t_strobe)
    ax.scatter(strobe_vals[0], strobe_vals[1], s=12, c="crimson",
               zorder=5, edgecolors="black", linewidths=0.3, label="Poincaré")

    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("y (= ẋ)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9, loc="upper right")

fig1.suptitle(
    "Forced Duffing Oscillator — Phase Portraits\n"
    r"$\ddot{x} + \delta\dot{x} - x + x^3 = a\sin(\omega t)$"
    f"    (δ={delta}, ω={omega:.1f})",
    fontsize=14, y=0.98,
)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("duffing_phase_portraits.png", dpi=150, bbox_inches="tight")
print("Saved duffing_phase_portraits.png")

# ── Figure 2: Bifurcation Diagram (stroboscopic) ──

print("\nGenerating bifurcation diagram...")

a_values = np.linspace(0.1, 0.7, 600)
n_transient_bif = 200
n_collect_bif = 300

all_a = []
all_x = []

for i, a in enumerate(a_values):
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(a_values)} ...")

    t_end = (n_transient_bif + n_collect_bif) * T
    sol = solve_ivp(
        lambda t, s, a=a: duffing(t, s, a),
        (0.0, t_end), [x0, y0],
        method="RK45", max_step=0.01,
        rtol=1e-8, atol=1e-10,
        dense_output=True,
    )

    # Stroboscopic sampling after transient
    t_strobe = np.arange(n_transient_bif, n_transient_bif + n_collect_bif + 1) * T
    x_strobe = sol.sol(t_strobe)[0]

    all_a.extend([a] * len(x_strobe))
    all_x.extend(x_strobe)

all_a = np.array(all_a)
all_x = np.array(all_x)

fig2, ax2 = plt.subplots(figsize=(14, 8))
ax2.scatter(all_a, all_x, s=0.05, c="black", alpha=0.3, rasterized=True)
ax2.set_xlabel("Forcing amplitude  a", fontsize=12)
ax2.set_ylabel("Stroboscopic  x  (sampled at  t = nT)", fontsize=12)
ax2.set_title(
    "Duffing Oscillator — Bifurcation Diagram (stroboscopic)\n"
    r"$\ddot{x} + \delta\dot{x} - x + x^3 = a\sin(\omega t)$"
    f"    (δ={delta}, ω={omega:.1f})",
    fontsize=14,
)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("duffing_bifurcation.png", dpi=150, bbox_inches="tight")
print("Saved duffing_bifurcation.png")

plt.show()

# ── Summary ──

print("\n=== Duffing Oscillator Summary ===")
print(f"  Parameters: δ = {delta}, ω = {omega}")
print(f"  Double-well potential: V(x) = -x²/2 + x⁴/4  (minima at x = ±1)")
print(f"\n  Phase portraits show transition:")
print(f"    a = 0.00 → unforced well oscillation")
print(f"    a = 0.25 → period-1 orbit (confined to one well)")
print(f"    a = 0.30 → chaotic inter-well motion (boundary crisis)")
print(f"    a = 0.50 → periodic window (large-amplitude period-1)")
print(f"\n  Bifurcation diagram sweeps a ∈ [0.1, 0.7] with {len(a_values)} values")
print(f"  Chaos onset via boundary crisis at a ≈ 0.26")
