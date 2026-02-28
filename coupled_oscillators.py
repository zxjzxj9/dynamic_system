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


# ══════════════════════════════════════════════════════════════════
# Figure 4: Phase Space — Attractor and Repeller
# ══════════════════════════════════════════════════════════════════

print("Computing phase space attractor/repeller diagram...")

dtheta = np.linspace(-np.pi, np.pi, 1000)

K_phase_values = [
    (0.1,  "C3", "--", f"K = 0.1 < K_c (no fixed points)"),
    (0.25, "gray", "-.", f"K = 0.25 = K_c (saddle-node)"),
    (0.4,  "C0", "-",  f"K = 0.4 > K_c"),
    (0.6,  "C2", "-",  f"K = 0.6 > K_c"),
]

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 10),
                                      gridspec_kw={"height_ratios": [3, 2]})

# ── Top panel: dΔθ/dt vs Δθ ──

ax_top.axhline(0, color="black", lw=0.5)

for K, color, ls, label in K_phase_values:
    ddtheta = Delta_omega - 2 * K * np.sin(dtheta)
    ax_top.plot(dtheta, ddtheta, color=color, ls=ls, lw=1.8, label=label)

    # Mark fixed points for locked cases
    if K >= K_c:
        ratio = Delta_omega / (2 * K)
        ratio = np.clip(ratio, -1, 1)
        # Stable fixed point (attractor)
        fp_stable = np.arcsin(ratio)
        # Unstable fixed point (repeller)
        fp_unstable = np.pi - np.arcsin(ratio)
        # Wrap to [-π, π]
        if fp_unstable > np.pi:
            fp_unstable -= 2 * np.pi

        ax_top.plot(fp_stable, 0, "o", color=color, ms=10, mfc=color,
                    mec="black", mew=1.2, zorder=5)
        ax_top.plot(fp_unstable, 0, "o", color=color, ms=10, mfc="white",
                    mec=color, mew=2.0, zorder=5)

# Legend entries for fixed point markers
ax_top.plot([], [], "ko", ms=8, mfc="black", label="Attractor (stable)")
ax_top.plot([], [], "o", color="black", ms=8, mfc="white", mew=2.0,
            label="Repeller (unstable)")

ax_top.set_xlabel(r"Phase difference $\Delta\theta$", fontsize=12)
ax_top.set_ylabel(r"$d\Delta\theta / dt$", fontsize=12)
ax_top.set_title(
    r"Phase Space: $d\Delta\theta/dt = \Delta\omega - 2K\sin(\Delta\theta)$"
    "\n"
    r"Saddle-node bifurcation on a circle"
    f"   ($\\Delta\\omega = {Delta_omega}$)",
    fontsize=13)
ax_top.set_xlim(-np.pi, np.pi)
ax_top.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax_top.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$",
                         r"$\pi/2$", r"$\pi$"])
ax_top.legend(fontsize=9, loc="upper right")
ax_top.grid(True, alpha=0.2)

# Add flow arrows on the K=0.6 curve
K_arrow = 0.6
ddtheta_arrow = Delta_omega - 2 * K_arrow * np.sin(dtheta)
ratio_arrow = np.clip(Delta_omega / (2 * K_arrow), -1, 1)
fp_s = np.arcsin(ratio_arrow)
fp_u = np.pi - np.arcsin(ratio_arrow)
if fp_u > np.pi:
    fp_u -= 2 * np.pi

# Arrow positions: between repeller and attractor on each side
arrow_positions = [
    (fp_s + fp_u) / 2,            # between attractor and repeller (flowing toward attractor)
    fp_s - 0.8,                    # left of attractor (flowing toward attractor)
    fp_u + 0.5 if fp_u + 0.5 < np.pi else fp_u - 2.5,  # near repeller (flowing away)
]
for ap in arrow_positions:
    val = Delta_omega - 2 * K_arrow * np.sin(ap)
    direction = 1 if val > 0 else -1
    ax_top.annotate("", xy=(ap + direction * 0.15, val),
                    xytext=(ap - direction * 0.15, val),
                    arrowprops=dict(arrowstyle="->", color="C2", lw=2.0))

# ── Bottom panel: flow on the circle for K = 0.6 ──

K_circ = 0.6
ratio_circ = np.clip(Delta_omega / (2 * K_circ), -1, 1)
fp_stable_circ = np.arcsin(ratio_circ)
fp_unstable_circ = np.pi - np.arcsin(ratio_circ)
if fp_unstable_circ > np.pi:
    fp_unstable_circ -= 2 * np.pi

# Draw the circle
theta_circle = np.linspace(0, 2 * np.pi, 300)
R = 1.0
cx, cy = 0.0, 0.0
ax_bot.plot(cx + R * np.cos(theta_circle), cy + R * np.sin(theta_circle),
            "k-", lw=1.5)
ax_bot.set_aspect("equal")

# Mark attractor and repeller on circle
ax_bot.plot(R * np.cos(fp_stable_circ), R * np.sin(fp_stable_circ),
            "o", color="C2", ms=14, mfc="C2", mec="black", mew=1.5, zorder=5)
ax_bot.annotate("Attractor\n" + rf"$\Delta\theta^* = {fp_stable_circ:.2f}$",
                xy=(R * np.cos(fp_stable_circ), R * np.sin(fp_stable_circ)),
                xytext=(R * np.cos(fp_stable_circ) - 0.9,
                        R * np.sin(fp_stable_circ) - 0.5),
                fontsize=10, ha="center",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

ax_bot.plot(R * np.cos(fp_unstable_circ), R * np.sin(fp_unstable_circ),
            "o", color="C2", ms=14, mfc="white", mec="C2", mew=2.5, zorder=5)
ax_bot.annotate("Repeller\n" + rf"$\Delta\theta^* = {fp_unstable_circ:.2f}$",
                xy=(R * np.cos(fp_unstable_circ), R * np.sin(fp_unstable_circ)),
                xytext=(R * np.cos(fp_unstable_circ) + 0.9,
                        R * np.sin(fp_unstable_circ) + 0.5),
                fontsize=10, ha="center",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

# Draw flow arrows around the circle
n_arrows = 12
for i in range(n_arrows):
    angle = -np.pi + (2 * np.pi * i / n_arrows)
    # Skip if too close to fixed points
    if min(abs(angle - fp_stable_circ), abs(angle - fp_unstable_circ)) < 0.3:
        continue
    flow = Delta_omega - 2 * K_circ * np.sin(angle)
    # Arrow tangent to circle
    tangent_x = -np.sin(angle)
    tangent_y = np.cos(angle)
    sign = 1 if flow > 0 else -1
    arrow_len = 0.12
    x0 = R * np.cos(angle)
    y0 = R * np.sin(angle)
    dx = sign * arrow_len * tangent_x
    dy = sign * arrow_len * tangent_y
    ax_bot.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="C0",
                                    lw=1.8, mutation_scale=15))

ax_bot.set_xlim(-2.0, 2.0)
ax_bot.set_ylim(-1.6, 1.6)
ax_bot.set_title(
    f"Flow on the Phase Circle (K = {K_circ})\n"
    "All orbits flow toward the attractor, away from the repeller",
    fontsize=12)
ax_bot.axis("off")

plt.tight_layout()
plt.savefig("coupled_phase_space.png", dpi=150)
print("Saved coupled_phase_space.png")


# ══════════════════════════════════════════════════════════════════
# Figure 5: Critical Slowing Down (fixed K, increasing |Δω|)
# ══════════════════════════════════════════════════════════════════

print("Computing critical slowing down...")

K_fixed = 0.6
Dw_critical = 2 * K_fixed  # = 1.2

# Panel layout: top = time series, bottom left = relaxation time, bottom right = Arnold tongue path
fig = plt.figure(figsize=(14, 12))
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.35, wspace=0.3)
ax_ts = fig.add_subplot(gs[0, :])
ax_tau = fig.add_subplot(gs[1, 0])
ax_tongue = fig.add_subplot(gs[1, 1])

# ── Top panel: time series of Δθ(t) for several Δω ──

Dw_values = [0.2, 0.6, 0.9, 1.05, 1.15, 1.19]
T_end_csd = 150.0
t_eval_csd = np.linspace(0, T_end_csd, 8000)
colors_csd = plt.cm.viridis(np.linspace(0.1, 0.9, len(Dw_values)))

# Start with a perturbation away from the fixed point
theta0_csd = [np.pi / 3, 0.0]  # initial phase difference ≈ π/3

for i, Dw in enumerate(Dw_values):
    omega1_csd = 1.0
    omega2_csd = 1.0 - Dw  # so Δω = ω₁ − ω₂ = Dw

    def rhs_csd(t, theta, K=K_fixed, w1=omega1_csd, w2=omega2_csd):
        th1, th2 = theta
        return [w1 + K * np.sin(th2 - th1),
                w2 + K * np.sin(th1 - th2)]

    sol = solve_ivp(rhs_csd, [0, T_end_csd], theta0_csd,
                    t_eval=t_eval_csd, rtol=1e-10, atol=1e-12)
    delta_theta = sol.y[0] - sol.y[1]

    # Compute the expected equilibrium
    ratio = Dw / (2 * K_fixed)
    if abs(ratio) <= 1:
        eq_val = np.arcsin(ratio)
    else:
        eq_val = None

    label = rf"$\Delta\omega = {Dw}$"
    if abs(Dw - 1.19) < 0.01:
        label += r"  ($\approx \Delta\omega_c$)"

    ax_ts.plot(sol.t, delta_theta, color=colors_csd[i], lw=1.3, label=label)

ax_ts.set_xlabel("t", fontsize=12)
ax_ts.set_ylabel(r"$\Delta\theta(t)$", fontsize=12)
ax_ts.set_title(
    f"Critical Slowing Down — Phase Relaxation (K = {K_fixed})\n"
    rf"$\Delta\omega_c = 2K = {Dw_critical}$: "
    "relaxation time diverges as the locking boundary is approached",
    fontsize=13)
ax_ts.legend(fontsize=9, loc="upper right")
ax_ts.grid(True, alpha=0.2)

# ── Bottom-left: relaxation time vs Δω ──

# Theoretical: near the saddle-node, linearized decay rate is
# γ = sqrt((2K)² − Δω²), so τ = 1/γ = 1/sqrt(4K² − Δω²)
Dw_sweep = np.linspace(0.0, Dw_critical - 0.001, 1000)
gamma_theory = np.sqrt((2 * K_fixed)**2 - Dw_sweep**2)
tau_theory = 1.0 / gamma_theory

# Numerical: measure time for |Δθ(t) - Δθ*| to drop below threshold
Dw_num = np.linspace(0.05, Dw_critical - 0.01, 80)
tau_num = np.empty(len(Dw_num))
threshold = 0.01

for i, Dw in enumerate(Dw_num):
    omega2_n = 1.0 - Dw
    ratio = Dw / (2 * K_fixed)
    eq_val = np.arcsin(np.clip(ratio, -1, 1))

    def rhs_n(t, theta, K=K_fixed, w2=omega2_n):
        th1, th2 = theta
        return [1.0 + K * np.sin(th2 - th1),
                w2 + K * np.sin(th1 - th2)]

    sol = solve_ivp(rhs_n, [0, 500.0], theta0_csd,
                    t_eval=np.linspace(0, 500.0, 20000),
                    rtol=1e-10, atol=1e-12)
    delta_theta_n = sol.y[0] - sol.y[1]
    # Find first time |Δθ − eq| < threshold
    err = np.abs(delta_theta_n - eq_val)
    locked_idx = np.where(err < threshold)[0]
    if len(locked_idx) > 0:
        tau_num[i] = sol.t[locked_idx[0]]
    else:
        tau_num[i] = 500.0  # didn't converge

ax_tau.plot(Dw_sweep, tau_theory, "C3-", lw=1.5,
            label=r"Theory: $\tau = 1/\sqrt{4K^2 - \Delta\omega^2}$")
ax_tau.plot(Dw_num, tau_num, "ko", ms=3, alpha=0.7,
            label="Numerical (time to converge)")
ax_tau.axvline(Dw_critical, color="gray", ls="--", lw=1.0, alpha=0.7)
ax_tau.annotate(rf"$\Delta\omega_c = {Dw_critical}$",
                xy=(Dw_critical, 0.9), xycoords=("data", "axes fraction"),
                fontsize=10, ha="right", color="gray")
ax_tau.set_xlabel(r"$\Delta\omega$", fontsize=12)
ax_tau.set_ylabel(r"Relaxation time $\tau$", fontsize=12)
ax_tau.set_title("Relaxation Time Divergence", fontsize=12)
ax_tau.set_ylim(0, 60)
ax_tau.legend(fontsize=9)
ax_tau.grid(True, alpha=0.2)

# ── Bottom-right: Arnold tongue with path marked ──

dw_range_t = np.linspace(-2.0, 2.0, 400)
K_range_t = np.linspace(0.0, 1.5, 300)
DW_t, KK_t = np.meshgrid(dw_range_t, K_range_t)
locked_t = KK_t >= np.abs(DW_t) / 2.0

ax_tongue.contourf(DW_t, KK_t, locked_t.astype(float), levels=[0.5, 1.5],
                    colors=["#4C72B0"], alpha=0.3)
ax_tongue.contour(DW_t, KK_t, locked_t.astype(float), levels=[0.5],
                  colors=["#4C72B0"], linewidths=1.5)

# Theoretical boundary
dw_pos_t = np.linspace(0, 2.0, 200)
ax_tongue.plot(dw_pos_t, dw_pos_t / 2, "k--", lw=1.0)
ax_tongue.plot(-dw_pos_t, dw_pos_t / 2, "k--", lw=1.0)

# Draw the path: horizontal line at K = 0.6, Δω from 0 to 1.2
ax_tongue.axhline(K_fixed, color="C3", lw=2.0, ls="-", alpha=0.5)
ax_tongue.annotate("", xy=(Dw_critical, K_fixed),
                   xytext=(0.0, K_fixed),
                   arrowprops=dict(arrowstyle="-|>", color="C3", lw=2.0))

# Mark the sampled Δω values
for Dw in Dw_values:
    ax_tongue.plot(Dw, K_fixed, "o", color="C3", ms=5, mec="black",
                   mew=0.5, zorder=5)

# Mark critical point
ax_tongue.plot(Dw_critical, K_fixed, "*", color="gold", ms=14,
               mec="black", mew=1.0, zorder=6)
ax_tongue.annotate(rf"$\Delta\omega_c = {Dw_critical}$",
                   xy=(Dw_critical, K_fixed),
                   xytext=(Dw_critical + 0.15, K_fixed + 0.15),
                   fontsize=10, ha="left",
                   arrowprops=dict(arrowstyle="->", color="black", lw=1.0))

ax_tongue.set_xlabel(r"$\Delta\omega$", fontsize=12)
ax_tongue.set_ylabel("K", fontsize=12)
ax_tongue.set_title(f"Path through Arnold tongue (K = {K_fixed})", fontsize=12)
ax_tongue.set_xlim(-1.5, 2.0)
ax_tongue.set_ylim(0, 1.2)
ax_tongue.grid(True, alpha=0.2)

plt.savefig("coupled_critical_slowing.png", dpi=150)
print("Saved coupled_critical_slowing.png")

plt.show()

print("\n=== Coupled Oscillators Summary ===")
print(f"  ω₁ = {omega1}, ω₂ = {omega2}, Δω = {Delta_omega}")
print(f"  Critical coupling: K_c = |Δω|/2 = {K_c}")
print("  Below K_c: phase difference drifts (unlocked)")
print("  Above K_c: phase difference → constant (phase locked)")
print("  Effective frequencies merge at K_c (frequency locking)")
print("  Arnold tongue: V-shaped locking region in (Δω, K) space")
