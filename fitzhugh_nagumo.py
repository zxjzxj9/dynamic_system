"""
FitzHugh–Nagumo Model — Excitability and Relaxation Oscillations
=================================================================

    dv/dt = v − v³/3 − w + I
    dw/dt = ε(v + a − bw)

A simplified 2D model of neuronal excitability. The fast variable v
represents membrane potential; the slow variable w represents recovery.
External current I controls the regime:

    - Small I: excitable (stable equilibrium, threshold response)
    - Intermediate I: oscillatory (limit cycle via Hopf bifurcation)
    - Large I: excitable again (stable equilibrium restored)

Parameters: a = 0.7, b = 0.8, ε = 0.08
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.integrate import solve_ivp


# ── Parameters ──

a = 0.7
b = 0.8
eps = 0.08


def fhn_rhs(t, state, I):
    v, w = state
    dv = v - v**3 / 3 - w + I
    dw = eps * (v + a - b * w)
    return [dv, dw]


def v_nullcline(v, I):
    """w = v − v³/3 + I"""
    return v - v**3 / 3 + I


def w_nullcline(v):
    """w = (v + a) / b"""
    return (v + a) / b


# ══════════════════════════════════════════════════════════════════
# Figure 1: Phase Portraits at different I
# ══════════════════════════════════════════════════════════════════

print("Computing phase portraits...")

I_panels = [
    (0.0,  "I = 0  (excitable)"),
    (0.34, "I = 0.34  (near Hopf)"),
    (0.5,  "I = 0.5  (oscillatory)"),
    (1.0,  "I = 1.0  (relaxation oscillations)"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

v_range = np.linspace(-3.0, 3.0, 400)

for idx, (I_val, label) in enumerate(I_panels):
    ax = axes[idx]
    print(f"  Panel {idx + 1}: {label}...")

    # Nullclines
    w_vnull = v_nullcline(v_range, I_val)
    w_wnull = w_nullcline(v_range)

    ax.plot(v_range, w_vnull, "limegreen", lw=2.5, ls="--",
            label=r"$\dot{v}=0$: $w = v - v^3/3 + I$")
    ax.plot(v_range, w_wnull, "magenta", lw=2.5, ls="--",
            label=r"$\dot{w}=0$: $w = (v+a)/b$")

    # Streamplot
    vlim, wlim = (-2.5, 2.5), (-1.0, 3.5)
    qgrid = 30
    Vq, Wq = np.meshgrid(np.linspace(vlim[0], vlim[1], qgrid),
                          np.linspace(wlim[0], wlim[1], qgrid))
    Uq = Vq - Vq**3 / 3 - Wq + I_val
    Vdq = eps * (Vq + a - b * Wq)
    sq = np.sqrt(Uq**2 + Vdq**2)
    sq = np.where(sq == 0, 1e-10, sq)

    ax.streamplot(Vq, Wq, Uq, Vdq, color=sq, cmap="Blues",
                  norm=Normalize(vmin=0, vmax=np.percentile(sq, 90)),
                  density=1.5, linewidth=0.6, arrowsize=0.8)

    # Find equilibrium numerically (intersection of nullclines)
    # v - v³/3 + I = (v + a)/b  →  solve
    from scipy.optimize import brentq
    def eq_func(v, I=I_val):
        return v - v**3 / 3 + I - (v + a) / b

    # Find all roots by scanning
    eq_vs = []
    v_scan = np.linspace(-3, 3, 1000)
    f_scan = np.array([eq_func(v) for v in v_scan])
    for i in range(len(f_scan) - 1):
        if f_scan[i] * f_scan[i + 1] < 0:
            v_eq = brentq(eq_func, v_scan[i], v_scan[i + 1])
            eq_vs.append(v_eq)

    for v_eq in eq_vs:
        w_eq = w_nullcline(v_eq)
        # Jacobian
        J = np.array([
            [1 - v_eq**2, -1],
            [eps, -eps * b]
        ])
        eigs = np.linalg.eigvals(J)
        re_parts = eigs.real

        if all(r < 0 for r in re_parts):
            if any(abs(e.imag) > 1e-6 for e in eigs):
                cls = "stable spiral"
                marker, color = "o", "blue"
            else:
                cls = "stable node"
                marker, color = "o", "blue"
        elif all(r > 0 for r in re_parts):
            if any(abs(e.imag) > 1e-6 for e in eigs):
                cls = "unstable spiral"
                marker, color = "^", "red"
            else:
                cls = "unstable node"
                marker, color = "^", "red"
        else:
            cls = "saddle"
            marker, color = "s", "green"

        ax.plot(v_eq, w_eq, marker=marker, color=color, ms=10,
                mec="black", mew=1.5, zorder=5)
        eig_str = ", ".join(f"{e:.3g}" for e in eigs)
        ax.annotate(f"{cls}\nλ = {eig_str}",
                    xy=(v_eq, w_eq), xytext=(12, 12),
                    textcoords="offset points", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85),
                    arrowprops=dict(arrowstyle="->", color="black"))

    # Sample trajectory
    # For excitable: start near equilibrium with a kick
    if I_val < 0.33:
        v0_traj = eq_vs[0] + 0.5 if eq_vs else 0.0
        w0_traj = w_nullcline(eq_vs[0]) if eq_vs else 0.0
        T_traj = 120
    else:
        # For oscillatory: start away from equilibrium
        v0_traj = -1.5
        w0_traj = -0.5
        T_traj = 200

    sol = solve_ivp(fhn_rhs, [0, T_traj], [v0_traj, w0_traj],
                    args=(I_val,), max_step=0.1, rtol=1e-8, atol=1e-10)
    # Discard transient for limit cycle
    n_skip = len(sol.t) // 3
    ax.plot(sol.y[0][n_skip:], sol.y[1][n_skip:], "gold", lw=2.0,
            alpha=0.9, zorder=4)
    # Show start for excitable
    if I_val < 0.33:
        ax.plot(sol.y[0][0], sol.y[1][0], "*", color="gold", ms=12,
                mec="black", mew=0.8, zorder=6)

    ax.set_xlim(vlim)
    ax.set_ylim(wlim)
    ax.set_title(label, fontsize=12)
    ax.grid(True, alpha=0.2)
    if idx >= 2:
        ax.set_xlabel("v  (membrane potential)", fontsize=11)
    if idx % 2 == 0:
        ax.set_ylabel("w  (recovery variable)", fontsize=11)
    if idx == 0:
        ax.legend(fontsize=7, loc="upper left")

fig.suptitle(
    "FitzHugh–Nagumo — Phase Portraits\n"
    r"$\dot{v} = v - v^3/3 - w + I$,  "
    r"$\dot{w} = \varepsilon(v + a - bw)$"
    f"   (a={a}, b={b}, ε={eps})",
    fontsize=13, y=1.01)

plt.tight_layout()
plt.savefig("fhn_phase_portraits.png", dpi=150, bbox_inches="tight")
print("Saved fhn_phase_portraits.png")


# ══════════════════════════════════════════════════════════════════
# Figure 2: Time Series — excitable vs oscillatory
# ══════════════════════════════════════════════════════════════════

print("Computing time series...")

fig, axes = plt.subplots(2, 2, figsize=(16, 8))

# Top row: excitable (I=0), perturbation → single spike
I_exc = 0.0
# Find equilibrium
v_eq_exc = brentq(lambda v: v - v**3/3 + I_exc - (v+a)/b, -3, 3)
w_eq_exc = w_nullcline(v_eq_exc)

# Start at equilibrium, then model a brief current pulse
def fhn_pulse(t, state):
    v, w = state
    I_t = I_exc + (2.0 if 5.0 < t < 6.0 else 0.0)  # brief pulse
    return [v - v**3/3 - w + I_t, eps * (v + a - b * w)]

T_exc = 120.0
sol_exc = solve_ivp(fhn_pulse, [0, T_exc], [v_eq_exc, w_eq_exc],
                    max_step=0.1, rtol=1e-8, atol=1e-10)

axes[0, 0].plot(sol_exc.t, sol_exc.y[0], "C0-", lw=1.2, label="v(t)")
axes[0, 0].plot(sol_exc.t, sol_exc.y[1], "C3-", lw=1.2, label="w(t)")
axes[0, 0].axvspan(5, 6, color="gold", alpha=0.3, label="Stimulus pulse")
axes[0, 0].set_ylabel("v, w", fontsize=11)
axes[0, 0].set_title("Excitable Regime (I = 0) — Single Spike", fontsize=11)
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.2)

# Phase portrait of the spike
axes[0, 1].plot(v_range, v_nullcline(v_range, I_exc), "limegreen",
                lw=2, ls="--")
axes[0, 1].plot(v_range, w_nullcline(v_range), "magenta", lw=2, ls="--")
axes[0, 1].plot(sol_exc.y[0], sol_exc.y[1], "C0-", lw=1.5)
axes[0, 1].plot(sol_exc.y[0][0], sol_exc.y[1][0], "o", color="C0",
                ms=8, mec="black", zorder=5)
axes[0, 1].set_xlim(-2.5, 2.5)
axes[0, 1].set_ylim(-1.0, 3.0)
axes[0, 1].set_title("Phase Portrait — Spike Trajectory", fontsize=11)
axes[0, 1].grid(True, alpha=0.2)

# Bottom row: oscillatory (I=0.5), sustained spiking
I_osc = 0.5
T_osc = 250.0
sol_osc = solve_ivp(fhn_rhs, [0, T_osc], [-1.0, -0.5],
                    args=(I_osc,), max_step=0.1, rtol=1e-8, atol=1e-10)

# Discard transient
t_start = 50.0
mask = sol_osc.t >= t_start

axes[1, 0].plot(sol_osc.t[mask], sol_osc.y[0][mask], "C0-", lw=1.2,
                label="v(t)")
axes[1, 0].plot(sol_osc.t[mask], sol_osc.y[1][mask], "C3-", lw=1.2,
                label="w(t)")
axes[1, 0].set_xlabel("t", fontsize=11)
axes[1, 0].set_ylabel("v, w", fontsize=11)
axes[1, 0].set_title("Oscillatory Regime (I = 0.5) — Sustained Spiking",
                      fontsize=11)
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.2)

# Phase portrait of limit cycle
n_skip_osc = np.searchsorted(sol_osc.t, t_start)
axes[1, 1].plot(v_range, v_nullcline(v_range, I_osc), "limegreen",
                lw=2, ls="--")
axes[1, 1].plot(v_range, w_nullcline(v_range), "magenta", lw=2, ls="--")
axes[1, 1].plot(sol_osc.y[0][n_skip_osc:], sol_osc.y[1][n_skip_osc:],
                "C0-", lw=1.5)
axes[1, 1].set_xlim(-2.5, 2.5)
axes[1, 1].set_ylim(-1.0, 3.0)
axes[1, 1].set_title("Phase Portrait — Limit Cycle", fontsize=11)
axes[1, 1].grid(True, alpha=0.2)

fig.suptitle(
    "FitzHugh–Nagumo — Time Series and Phase Trajectories\n"
    f"(a={a}, b={b}, ε={eps})",
    fontsize=13)

plt.tight_layout()
plt.savefig("fhn_time_series.png", dpi=150)
print("Saved fhn_time_series.png")


# ══════════════════════════════════════════════════════════════════
# Figure 3: Bifurcation Diagram — sweep I
# ══════════════════════════════════════════════════════════════════

print("Computing bifurcation diagram (sweeping I)...")

I_sweep = np.linspace(-0.5, 2.0, 1000)

# Equilibria
v_eq_all = []
stability_all = []

for I_val in I_sweep:
    def eq_f(v, I=I_val):
        return v - v**3/3 + I - (v + a) / b

    v_scan = np.linspace(-3, 3, 500)
    f_scan = np.array([eq_f(v) for v in v_scan])
    roots = []
    for i in range(len(f_scan) - 1):
        if f_scan[i] * f_scan[i + 1] < 0:
            roots.append(brentq(eq_f, v_scan[i], v_scan[i + 1]))

    for v_r in roots:
        J = np.array([[1 - v_r**2, -1], [eps, -eps * b]])
        eigs = np.linalg.eigvals(J)
        stable = all(e.real < 0 for e in eigs)
        v_eq_all.append((I_val, v_r, stable))

# Limit cycle amplitude: sweep I, integrate, record min/max v
I_lc = np.linspace(0.0, 1.5, 200)
lc_vmin = []
lc_vmax = []
lc_I = []

for I_val in I_lc:
    sol = solve_ivp(fhn_rhs, [0, 500], [-1.0, -0.5], args=(I_val,),
                    max_step=0.2, rtol=1e-8, atol=1e-10)
    # Use last portion
    n_last = len(sol.t) // 2
    v_last = sol.y[0][n_last:]
    v_amp = np.max(v_last) - np.min(v_last)
    if v_amp > 0.05:  # oscillating
        lc_vmin.append(np.min(v_last))
        lc_vmax.append(np.max(v_last))
        lc_I.append(I_val)

fig, ax = plt.subplots(figsize=(12, 7))

# Plot equilibria
I_stable = [p[0] for p in v_eq_all if p[2]]
v_stable = [p[1] for p in v_eq_all if p[2]]
I_unstable = [p[0] for p in v_eq_all if not p[2]]
v_unstable = [p[1] for p in v_eq_all if not p[2]]

ax.plot(I_stable, v_stable, "C0.", ms=2, label="Stable equilibrium")
ax.plot(I_unstable, v_unstable, "C3.", ms=2, alpha=0.5,
        label="Unstable equilibrium")

# Plot limit cycle envelope
if lc_I:
    ax.plot(lc_I, lc_vmax, "C2-", lw=2, label="Limit cycle max(v)")
    ax.plot(lc_I, lc_vmin, "C2-", lw=2, label="Limit cycle min(v)")
    ax.fill_between(lc_I, lc_vmin, lc_vmax, color="C2", alpha=0.1)

# Mark approximate Hopf bifurcation points
# Hopf occurs where equilibrium changes stability
hopf_I = []
for i in range(1, len(v_eq_all)):
    if v_eq_all[i-1][2] != v_eq_all[i][2]:
        # Check they're on the same branch (similar v)
        if abs(v_eq_all[i][1] - v_eq_all[i-1][1]) < 0.5:
            hopf_I.append(v_eq_all[i][0])

for I_h in hopf_I:
    ax.axvline(I_h, color="gray", ls=":", lw=1.0, alpha=0.5)
    ax.annotate(f"Hopf\nI ≈ {I_h:.2f}",
                xy=(I_h, 2.0), fontsize=9, ha="center", color="gray",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

ax.set_xlabel("External current I", fontsize=12)
ax.set_ylabel("v  (membrane potential)", fontsize=12)
ax.set_title(
    "FitzHugh–Nagumo — Bifurcation Diagram\n"
    "Hopf bifurcation: stable equilibrium → limit cycle → stable equilibrium",
    fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("fhn_bifurcation.png", dpi=150)
print("Saved fhn_bifurcation.png")

plt.show()

print("\n=== FitzHugh-Nagumo Summary ===")
print(f"  Parameters: a = {a}, b = {b}, ε = {eps}")
print("  dv/dt = v − v³/3 − w + I  (fast, membrane potential)")
print("  dw/dt = ε(v + a − bw)      (slow, recovery)")
print("  Small I: excitable — threshold perturbation → single spike")
print("  Intermediate I: oscillatory — sustained spiking (limit cycle)")
print("  Large I: excitable again — equilibrium restored")
print("  Transition via Hopf bifurcation (subcritical or supercritical)")
