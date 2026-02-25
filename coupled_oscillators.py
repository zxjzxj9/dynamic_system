"""
Coupled Oscillators — Phase Locking and Frequency Locking
==========================================================

    dθ₁/dt = ω₁ + K sin(θ₂ − θ₁)
    dθ₂/dt = ω₂ + K sin(θ₁ − θ₂)

Two Kuramoto oscillators with different natural frequencies ω₁, ω₂
coupled with strength K. The phase difference Δθ = θ₁ − θ₂ obeys:

    dΔθ/dt = Δω − 2K sin(Δθ)

where Δω = ω₁ − ω₂. When K ≥ |Δω|/2, the oscillators phase-lock
(Δθ → const) and frequency-lock (effective frequencies become equal).
The critical coupling is K_c = |Δω|/2.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ── Parameters ──

omega1 = 1.0
omega2 = 1.5
Delta_omega = omega1 - omega2  # = -0.5
K_c = abs(Delta_omega) / 2.0   # = 0.25


def kuramoto_2(t, theta, K):
    """RHS for two coupled Kuramoto oscillators."""
    th1, th2 = theta
    dth1 = omega1 + K * np.sin(th2 - th1)
    dth2 = omega2 + K * np.sin(th1 - th2)
    return [dth1, dth2]


# ══════════════════════════════════════════════════════════════════
# Figure 1: Phase Locking Time Series
# ══════════════════════════════════════════════════════════════════

print("Computing phase locking time series...")

K_panels = [
    (0.1,  f"K = 0.1  (below K_c, unlocked)"),
    (0.25, f"K = 0.25  (critical, K_c)"),
    (0.6,  f"K = 0.6  (above K_c, locked)"),
]

T_end = 100.0
t_eval = np.linspace(0, T_end, 5000)
theta0 = [0.0, 0.0]

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

for idx, (K, label) in enumerate(K_panels):
    ax = axes[idx]
    sol = solve_ivp(kuramoto_2, [0, T_end], theta0, args=(K,),
                    t_eval=t_eval, rtol=1e-10, atol=1e-12)
    delta_theta = sol.y[0] - sol.y[1]

    ax.plot(sol.t, delta_theta, color="C0", lw=1.0)
    ax.set_ylabel(r"$\Delta\theta(t)$", fontsize=11)
    ax.set_title(label, fontsize=11)
    ax.grid(True, alpha=0.2)

    if K >= K_c:
        # Mark the steady-state value
        final_val = delta_theta[-1]
        ax.axhline(final_val, color="crimson", ls="--", lw=0.8, alpha=0.7)

axes[-1].set_xlabel("t", fontsize=12)

fig.suptitle(
    "Coupled Kuramoto Oscillators — Phase Locking\n"
    r"$\dot{\theta}_i = \omega_i + K\sin(\theta_j - \theta_i)$"
    f"   ($\\omega_1 = {omega1},\\ \\omega_2 = {omega2},\\ "
    f"K_c = |\\Delta\\omega|/2 = {K_c}$)",
    fontsize=13)

plt.tight_layout()
plt.savefig("coupled_phase_locking.png", dpi=150)
print("Saved coupled_phase_locking.png")


# ══════════════════════════════════════════════════════════════════
# Figure 2: Frequency Locking Transition
# ══════════════════════════════════════════════════════════════════

print("Computing frequency locking transition...")

K_values = np.linspace(0.001, 1.0, 500)
T_total = 2000.0
T_measure_start = 1000.0

omega_eff_1 = np.empty(len(K_values))
omega_eff_2 = np.empty(len(K_values))

for i, K in enumerate(K_values):
    sol = solve_ivp(kuramoto_2, [0, T_total], theta0, args=(K,),
                    t_eval=[T_measure_start, T_total],
                    rtol=1e-10, atol=1e-12)
    dt = sol.t[-1] - sol.t[0]
    omega_eff_1[i] = (sol.y[0, -1] - sol.y[0, 0]) / dt
    omega_eff_2[i] = (sol.y[1, -1] - sol.y[1, 0]) / dt

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(K_values, omega_eff_1, "C0-", lw=1.5,
        label=rf"$\omega_1^{{\mathrm{{eff}}}}$  ($\omega_1 = {omega1}$)")
ax.plot(K_values, omega_eff_2, "C3-", lw=1.5,
        label=rf"$\omega_2^{{\mathrm{{eff}}}}$  ($\omega_2 = {omega2}$)")
ax.axvline(K_c, color="gray", ls="--", lw=1.0, alpha=0.7)
ax.annotate(f"$K_c = {K_c}$", xy=(K_c, omega1 - 0.05),
            fontsize=10, ha="right", color="gray",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

ax.set_xlabel("Coupling strength K", fontsize=12)
ax.set_ylabel("Effective frequency", fontsize=12)
ax.set_title(
    "Coupled Kuramoto Oscillators — Frequency Locking\n"
    r"Effective frequencies $\omega_i^{\mathrm{eff}}$ vs coupling K"
    f"   ($\\omega_1 = {omega1},\\ \\omega_2 = {omega2}$)",
    fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("coupled_freq_locking.png", dpi=150)
print("Saved coupled_freq_locking.png")


# ══════════════════════════════════════════════════════════════════
# Figure 3: Arnold Tongue
# ══════════════════════════════════════════════════════════════════

print("Computing Arnold tongue...")

dw_range = np.linspace(-2.0, 2.0, 800)
K_range = np.linspace(0.0, 1.5, 600)
DW, KK = np.meshgrid(dw_range, K_range)

# Analytical locking condition: K >= |Δω|/2
locked = KK >= np.abs(DW) / 2.0

fig, ax = plt.subplots(figsize=(10, 8))

ax.contourf(DW, KK, locked.astype(float), levels=[0.5, 1.5],
            colors=["#4C72B0"], alpha=0.4)
ax.contour(DW, KK, locked.astype(float), levels=[0.5],
           colors=["#4C72B0"], linewidths=1.5)

# Theoretical boundary
dw_pos = np.linspace(0, 2.0, 200)
ax.plot(dw_pos, dw_pos / 2, "k--", lw=1.2, label=r"$K = |\Delta\omega|/2$")
ax.plot(-dw_pos, dw_pos / 2, "k--", lw=1.2)

# Mark the parameters used in Figures 1 & 2
ax.plot(Delta_omega, 0.1, "ro", ms=7, zorder=5, label="K = 0.1 (unlocked)")
ax.plot(Delta_omega, 0.25, "gs", ms=7, zorder=5, label="K = 0.25 (critical)")
ax.plot(Delta_omega, 0.6, "b^", ms=7, zorder=5, label="K = 0.6 (locked)")

ax.set_xlabel(r"Frequency detuning $\Delta\omega = \omega_1 - \omega_2$",
              fontsize=12)
ax.set_ylabel("Coupling strength K", fontsize=12)
ax.set_title(
    "Arnold Tongue — Locking Region in Parameter Space\n"
    r"Shaded: oscillators phase-locked ($K \geq |\Delta\omega|/2$)",
    fontsize=13)
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("coupled_arnold_tongue.png", dpi=150)
print("Saved coupled_arnold_tongue.png")

plt.show()

print("\n=== Coupled Oscillators Summary ===")
print(f"  ω₁ = {omega1}, ω₂ = {omega2}, Δω = {Delta_omega}")
print(f"  Critical coupling: K_c = |Δω|/2 = {K_c}")
print("  Below K_c: phase difference drifts (unlocked)")
print("  Above K_c: phase difference → constant (phase locked)")
print("  Effective frequencies merge at K_c (frequency locking)")
print("  Arnold tongue: V-shaped locking region in (Δω, K) space")
