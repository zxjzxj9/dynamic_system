"""
Supply–Demand Oscillator — Attractor, Repeller, and Limit Cycle
===============================================================

A Kaldor-type nonlinear market model where speculative demand
creates an S-shaped excess demand function:

    dP/dt = α · E(P, Q)         (price adjusts to excess demand)
    dQ/dt = P − γQ              (production responds to price, decays)

The excess demand  E(P, Q) = D(P) − Q  where demand is nonlinear:

    D(P) = D₀ + μP − P³/3      (Kaldor/van der Pol-type)

The cubic term captures speculative markets: at moderate prices,
trend-following (μP) dominates and demand INCREASES with price.
At extreme prices, mean-reversion (−P³/3) dominates.

Jacobian at equilibrium (P*, Q*):
    J = [[α(μ − P*²),  −α],
         [1,            −γ]]

    tr(J) = α(μ − P*²) − γ
    det(J) = −αγ(μ − P*²) + α > 0  (when α > 0, γ > 0)

Hopf bifurcation when tr(J) = 0:
    α_c = γ / (μ − P*²)

For P* near 0 and μ > 0:
    α < α_c  →  stable spiral (ATTRACTOR)
    α = α_c  →  Hopf bifurcation
    α > α_c  →  unstable spiral (REPELLER) + stable LIMIT CYCLE
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ── Parameters ──

D0 = 0.0       # demand intercept (shifts equilibrium)
mu = 1.0        # speculative strength (trend-following)
gamma = 0.2     # production decay / adjustment rate


def demand(P):
    """D(P) = D₀ + μP − P³/3   (S-shaped speculative demand)"""
    return D0 + mu * P - P**3 / 3


def demand_deriv(P):
    """D'(P) = μ − P²"""
    return mu - P**2


def rhs(t, state, alpha):
    P, Q = state
    E = demand(P) - Q       # excess demand
    dP = alpha * E           # price adjustment
    dQ = P - gamma * Q       # production adjustment
    return [dP, dQ]


def find_equilibrium():
    """At equilibrium: dQ/dt = 0 → Q* = P*/γ,
    dP/dt = 0 → D(P*) = Q* = P*/γ.
    So: D₀ + μP* − P*³/3 = P*/γ
    """
    def eq(P):
        return D0 + mu * P - P**3 / 3 - P / gamma

    P_scan = np.linspace(-5, 5, 2000)
    f_scan = np.array([eq(P) for P in P_scan])
    roots = []
    for i in range(len(f_scan) - 1):
        if f_scan[i] * f_scan[i + 1] < 0:
            roots.append(brentq(eq, P_scan[i], P_scan[i + 1]))

    # Also check if P=0 is a root
    if abs(eq(0)) < 1e-10:
        roots.append(0.0)

    equilibria = []
    for P_eq in sorted(set([round(r, 8) for r in roots])):
        Q_eq = P_eq / gamma
        equilibria.append((P_eq, Q_eq))

    return equilibria


def classify_equilibrium(P_eq, Q_eq, alpha):
    """Classify via Jacobian eigenvalues."""
    dD = demand_deriv(P_eq)

    J = np.array([
        [alpha * dD, -alpha],
        [1.0,        -gamma]
    ])
    eigs = np.linalg.eigvals(J)

    re_parts = eigs.real
    if all(r < 0 for r in re_parts):
        if any(abs(e.imag) > 1e-6 for e in eigs):
            cls = "stable spiral"
        else:
            cls = "stable node"
        marker, color = "o", "#2196F3"
    elif all(r > 0 for r in re_parts):
        if any(abs(e.imag) > 1e-6 for e in eigs):
            cls = "unstable spiral"
        else:
            cls = "unstable node"
        marker, color = "^", "#F44336"
    else:
        cls = "saddle"
        marker, color = "s", "#4CAF50"

    return cls, marker, color, eigs, J


# ── Compute equilibrium and Hopf point ──

equilibria = find_equilibrium()
print("Equilibria:")
for P_e, Q_e in equilibria:
    dD = demand_deriv(P_e)
    print(f"  P* = {P_e:.4f}, Q* = {Q_e:.4f}, D'(P*) = {dD:.4f}")

# Hopf bifurcation: tr(J) = α(μ − P*²) − γ = 0
P_star, Q_star = equilibria[0]
dD_star = demand_deriv(P_star)
alpha_hopf = gamma / dD_star if dD_star > 0 else float('inf')
print(f"\nHopf bifurcation at α_c = γ/D'(P*) = {gamma}/{dD_star:.4f} = {alpha_hopf:.4f}")


# ══════════════════════════════════════════════════════════════════
# Figure 1: Phase Portraits — Attractor → Hopf → Repeller + Cycle
# ══════════════════════════════════════════════════════════════════

print("\nComputing phase portraits...")

alpha_panels = [
    (0.1,  r"$\alpha = 0.1$  (stable node — attractor)"),
    (0.15, r"$\alpha = 0.15$  (stable spiral — damped oscillations)"),
    (0.4,  r"$\alpha = 0.4$  (unstable spiral — repeller + limit cycle)"),
    (1.5,  r"$\alpha = 1.5$  (strong repeller + large limit cycle)"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, (alpha_val, label) in enumerate(alpha_panels):
    ax = axes[idx]
    print(f"  Panel {idx + 1}: α = {alpha_val}...")

    Plim = (-4.0, 4.0)
    Qlim = (-10.0, 10.0)

    P_range = np.linspace(Plim[0], Plim[1], 500)

    # dP/dt = 0 nullcline: Q = D(P) = D₀ + μP − P³/3
    Q_Pnull = demand(P_range)

    # dQ/dt = 0 nullcline: Q = P/γ
    Q_Qnull = P_range / gamma

    ax.plot(P_range, Q_Pnull, "limegreen", lw=2.5, ls="--",
            label=r"$\dot{P}=0$: $Q = D(P)$  (excess demand = 0)")
    ax.plot(P_range, Q_Qnull, "magenta", lw=2.5, ls="--",
            label=r"$\dot{Q}=0$: $Q = P/\gamma$  (production steady)")

    # Streamplot
    qgrid = 30
    Pg, Qg = np.meshgrid(np.linspace(Plim[0], Plim[1], qgrid),
                          np.linspace(Qlim[0], Qlim[1], qgrid))
    UP = alpha_val * (demand(Pg) - Qg)
    UQ = Pg - gamma * Qg
    speed = np.sqrt(UP**2 + UQ**2)
    speed = np.where(speed == 0, 1e-10, speed)

    ax.streamplot(Pg, Qg, UP, UQ, color=speed, cmap="Blues",
                  norm=Normalize(vmin=0, vmax=np.percentile(speed, 90)),
                  density=1.5, linewidth=0.6, arrowsize=0.8)

    # Equilibria
    for P_e, Q_e in equilibria:
        if Plim[0] <= P_e <= Plim[1] and Qlim[0] <= Q_e <= Qlim[1]:
            cls, marker, mc, eigs, J = classify_equilibrium(P_e, Q_e, alpha_val)
            ax.plot(P_e, Q_e, marker=marker, color=mc, ms=12,
                    mec="black", mew=1.5, zorder=5)
            eig_str = ", ".join(f"{e:.3f}" for e in eigs)
            # Position annotation to avoid overlap
            offset_y = 20 if Q_e < (Qlim[1] + Qlim[0]) / 2 else -30
            ax.annotate(f"{cls}\nλ = {eig_str}",
                        xy=(P_e, Q_e), xytext=(15, offset_y),
                        textcoords="offset points", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  alpha=0.85),
                        arrowprops=dict(arrowstyle="->", color="black"))

    # Trajectories from various initial conditions
    ics = [
        (-2.5, -5.0), (2.5, 5.0), (-3.0, 5.0), (3.0, -5.0),
        (0.5, 3.0), (-0.5, -3.0),
    ]
    traj_colors = ["#FF9800", "#E91E63", "#9C27B0",
                   "#00BCD4", "#CDDC39", "#795548"]

    T_sim = 200 if alpha_val < 0.3 else 300
    for ic, tc in zip(ics, traj_colors):
        sol = solve_ivp(rhs, [0, T_sim], list(ic), args=(alpha_val,),
                        max_step=0.05, rtol=1e-9, atol=1e-11)
        if alpha_val >= 0.3:
            n_trans = len(sol.t) // 3
            ax.plot(sol.y[0][:n_trans], sol.y[1][:n_trans], color=tc,
                    lw=0.8, alpha=0.3, zorder=3)
            ax.plot(sol.y[0][n_trans:], sol.y[1][n_trans:], color=tc,
                    lw=2.0, alpha=0.9, zorder=4)
        else:
            ax.plot(sol.y[0], sol.y[1], color=tc,
                    lw=1.8, alpha=0.85, zorder=4)
        ax.plot(sol.y[0][0], sol.y[1][0], "*", color=tc,
                ms=8, mec="black", mew=0.5, zorder=6)

    ax.set_xlim(Plim)
    ax.set_ylim(Qlim)
    ax.set_title(label, fontsize=12)
    ax.grid(True, alpha=0.2)
    if idx >= 2:
        ax.set_xlabel("P  (price deviation)", fontsize=11)
    if idx % 2 == 0:
        ax.set_ylabel("Q  (quantity deviation)", fontsize=11)
    if idx == 0:
        ax.legend(fontsize=7, loc="upper left")

fig.suptitle(
    "Supply–Demand Oscillator — Phase Portraits\n"
    r"$\dot{P} = \alpha(D_0 + \mu P - P^3/3 - Q)$,  "
    r"$\dot{Q} = P - \gamma Q$"
    f"   (μ={mu}, γ={gamma})\n"
    f"Hopf bifurcation at α_c = {alpha_hopf:.3f}",
    fontsize=12, y=1.02)

plt.tight_layout()
plt.savefig("supply_demand_phase.png", dpi=150, bbox_inches="tight")
print("Saved supply_demand_phase.png")


# ══════════════════════════════════════════════════════════════════
# Figure 2: Demand Curve and Stability Analysis
# ══════════════════════════════════════════════════════════════════

print("\nComputing demand/supply analysis...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

P_range = np.linspace(-3, 3, 500)

# Panel 1: The S-shaped demand vs linear supply/production line
ax1.plot(P_range, demand(P_range), "C0-", lw=2.5,
         label=r"Demand: $D(P) = \mu P - P^3/3$")
ax1.plot(P_range, P_range / gamma, "C2-", lw=2.5,
         label=r"Production equilibrium: $Q = P/\gamma$")

ax1.fill_between(P_range, demand(P_range), P_range / gamma,
                 where=(demand(P_range) > P_range / gamma),
                 color="C3", alpha=0.1, label="Excess demand (price rises)")
ax1.fill_between(P_range, demand(P_range), P_range / gamma,
                 where=(demand(P_range) < P_range / gamma),
                 color="C0", alpha=0.1, label="Excess supply (price falls)")

for P_e, Q_e in equilibria:
    ax1.plot(P_e, Q_e, "ko", ms=10, zorder=5)
    ax1.annotate(f"({P_e:.1f}, {Q_e:.1f})",
                 xy=(P_e, Q_e), xytext=(10, 10),
                 textcoords="offset points", fontsize=9,
                 bbox=dict(fc="white", alpha=0.8))

ax1.set_xlabel("P  (price deviation)", fontsize=12)
ax1.set_ylabel("Q  (quantity)", fontsize=12)
ax1.set_title("S-shaped Demand vs Production Line\n"
              "(speculative demand increases with price near P = 0)",
              fontsize=11)
ax1.legend(fontsize=8, loc="upper left")
ax1.grid(True, alpha=0.2)

# Panel 2: D'(P) showing the destabilizing region
dD_range = demand_deriv(P_range)
ax2.plot(P_range, dD_range, "C0-", lw=2.5, label=r"$D'(P) = \mu - P^2$")
ax2.axhline(0, color="black", lw=0.5)

ax2.fill_between(P_range, 0, dD_range, where=(dD_range > 0),
                 color="C3", alpha=0.15,
                 label=r"$D'(P) > 0$: trend-following dominates")
ax2.fill_between(P_range, 0, dD_range, where=(dD_range <= 0),
                 color="C0", alpha=0.15,
                 label=r"$D'(P) \leq 0$: mean-reversion dominates")

# Mark sqrt(mu) boundaries
P_crit = np.sqrt(mu)
ax2.axvline(P_crit, color="gray", ls=":", lw=1.5, alpha=0.7)
ax2.axvline(-P_crit, color="gray", ls=":", lw=1.5, alpha=0.7)
ax2.annotate(f"P = ±√μ = ±{P_crit:.2f}", xy=(P_crit, 0.05),
             fontsize=9, color="gray")

ax2.plot(P_star, dD_star, "ko", ms=8, zorder=5)
ax2.annotate(f"D'(P*) = {dD_star:.2f}\n"
             f"α_c = γ/D' = {alpha_hopf:.3f}",
             xy=(P_star, dD_star), xytext=(40, -30),
             textcoords="offset points", fontsize=10,
             bbox=dict(fc="white", alpha=0.85),
             arrowprops=dict(arrowstyle="->"))

ax2.set_xlabel("P  (price deviation)", fontsize=12)
ax2.set_ylabel("D'(P)", fontsize=12)
ax2.set_title("Demand Slope — Positive Near Origin = Destabilizing\n"
              r"Hopf bifurcation when $\alpha > \gamma / D'(P^*)$",
              fontsize=11)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("supply_demand_curves.png", dpi=150, bbox_inches="tight")
print("Saved supply_demand_curves.png")


# ══════════════════════════════════════════════════════════════════
# Figure 3: Bifurcation Diagram — sweep α
# ══════════════════════════════════════════════════════════════════

print("\nComputing bifurcation diagram...")

alpha_sweep = np.linspace(0.01, 2.0, 500)

eq_data = []
for alpha_val in alpha_sweep:
    for P_e, Q_e in equilibria:
        cls, _, _, eigs, _ = classify_equilibrium(P_e, Q_e, alpha_val)
        stable = "stable" in cls
        eq_data.append((alpha_val, P_e, Q_e, stable, cls))

# Limit cycle amplitude via simulation
print("  Sweeping α for limit cycle envelope...")
lc_data = []
alpha_lc = np.linspace(0.15, 2.0, 200)

for alpha_val in alpha_lc:
    sol = solve_ivp(rhs, [0, 1000], [0.1, 0.1], args=(alpha_val,),
                    max_step=0.1, rtol=1e-9, atol=1e-11)
    n_last = len(sol.t) * 3 // 4
    P_last = sol.y[0][n_last:]
    amp = np.max(P_last) - np.min(P_last)
    if amp > 0.05:
        lc_data.append((alpha_val, np.min(P_last), np.max(P_last)))

fig, ax = plt.subplots(figsize=(12, 7))

# Plot equilibria
# The central equilibrium at P*=0
a_s = [d[0] for d in eq_data if d[3] and abs(d[1] - P_star) < 0.1]
P_s = [d[1] for d in eq_data if d[3] and abs(d[1] - P_star) < 0.1]
a_u = [d[0] for d in eq_data if not d[3] and abs(d[1] - P_star) < 0.1]
P_u = [d[1] for d in eq_data if not d[3] and abs(d[1] - P_star) < 0.1]

# Other equilibria (saddles)
a_saddle = [d[0] for d in eq_data if "saddle" in d[4]]
P_saddle = [d[1] for d in eq_data if "saddle" in d[4]]

if a_s:
    ax.plot(a_s, P_s, "C0.", ms=3, label="Stable equilibrium (attractor)")
if a_u:
    ax.plot(a_u, P_u, "C3.", ms=3, alpha=0.6,
            label="Unstable equilibrium (repeller)")
if a_saddle:
    ax.plot(a_saddle, P_saddle, "C4.", ms=2, alpha=0.4,
            label="Saddle points")

if lc_data:
    lc_a = [d[0] for d in lc_data]
    lc_pmin = [d[1] for d in lc_data]
    lc_pmax = [d[2] for d in lc_data]
    ax.plot(lc_a, lc_pmax, "C2-", lw=2.5, label="Limit cycle max(P)")
    ax.plot(lc_a, lc_pmin, "C2-", lw=2.5, label="Limit cycle min(P)")
    ax.fill_between(lc_a, lc_pmin, lc_pmax, color="C2", alpha=0.1)

ax.axvline(alpha_hopf, color="gray", ls=":", lw=1.5)
ax.annotate(f"Hopf bifurcation\nα_c = {alpha_hopf:.3f}",
            xy=(alpha_hopf, 3.5), fontsize=11, ha="center",
            color="#555",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))

ax.annotate("ATTRACTOR\n(stable spiral)\nprices converge",
            xy=(alpha_hopf * 0.4, 0.5), fontsize=10,
            ha="center", color="#2196F3", fontweight="bold")
ax.annotate("REPELLER + LIMIT CYCLE\n(unstable spiral inside\nstable periodic orbit)",
            xy=(alpha_hopf + 0.7, 2.5), fontsize=10,
            ha="center", color="#F44336", fontweight="bold")

ax.set_xlabel(r"Price adjustment speed  $\alpha$", fontsize=12)
ax.set_ylabel("P  (price deviation)", fontsize=12)
ax.set_title(
    "Supply–Demand Bifurcation Diagram\n"
    r"Supercritical Hopf: attractor $\to$ repeller + stable limit cycle"
    f"  (γ = {gamma}, μ = {mu})",
    fontsize=13)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("supply_demand_bifurcation.png", dpi=150, bbox_inches="tight")
print("Saved supply_demand_bifurcation.png")


# ══════════════════════════════════════════════════════════════════
# Figure 4: Time Series — the three regimes
# ══════════════════════════════════════════════════════════════════

print("\nComputing time series...")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

alpha_ts = [0.1, 0.4, 1.5]
titles_ts = [
    r"$\alpha = 0.1 < \alpha_c$ — ATTRACTOR: "
    "prices spiral inward to equilibrium",
    r"$\alpha = 0.4 > \alpha_c$ — REPELLER + LIMIT CYCLE: "
    "sustained boom-bust oscillations",
    r"$\alpha = 1.5 \gg \alpha_c$ — STRONG REPELLER: "
    "large-amplitude relaxation oscillations",
]

for idx, (a_val, title) in enumerate(zip(alpha_ts, titles_ts)):
    sol = solve_ivp(rhs, [0, 200], [0.5, 0.5], args=(a_val,),
                    max_step=0.02, rtol=1e-9, atol=1e-11)

    axes[idx].plot(sol.t, sol.y[0], "#2196F3", lw=1.5,
                   label="P(t) — price")
    axes[idx].plot(sol.t, sol.y[1], "#F44336", lw=1.5,
                   label="Q(t) — quantity")
    axes[idx].axhline(P_star, color="gray", ls=":", lw=1, alpha=0.5,
                      label=f"P* = {P_star:.1f}")

    axes[idx].set_ylabel("P, Q", fontsize=11)
    axes[idx].set_title(title, fontsize=10)
    axes[idx].legend(fontsize=8, loc="upper right")
    axes[idx].grid(True, alpha=0.2)

axes[-1].set_xlabel("t  (time)", fontsize=11)

fig.suptitle(
    "Supply–Demand Dynamics — Time Series\n"
    "From stable attractor through Hopf bifurcation to boom-bust limit cycles",
    fontsize=12)

plt.tight_layout()
plt.savefig("supply_demand_timeseries.png", dpi=150, bbox_inches="tight")
print("Saved supply_demand_timeseries.png")

plt.show()

print("\n=== Supply–Demand Oscillator Summary ===")
print(f"  D(P) = {D0} + {mu}P − P³/3   (S-shaped speculative demand)")
print(f"  dP/dt = α(D(P) − Q),   dQ/dt = P − {gamma}Q")
print()
print(f"  Equilibrium: P* = {P_star:.3f}, Q* = {Q_star:.3f}")
print(f"  D'(P*) = μ − P*² = {dD_star:.4f} > 0")
print(f"  Hopf bifurcation at α_c = γ/D'(P*) = {alpha_hopf:.4f}")
print()
print("  The fixed point is UNSTABLE (repeller) when α > α_c:")
print("    tr(J) = α·D'(P*) − γ = α·(μ − P*²) − γ > 0")
print("    Eigenvalues have positive real part → spiral outward")
print()
print("  But nonlinear saturation (the −P³/3 term) prevents escape,")
print("  creating a stable LIMIT CYCLE that traps all trajectories.")
print("  The repeller pushes trajectories OUT, the cycle pulls them IN.")
print()
print("  Economic interpretation:")
print("    - Near equilibrium: speculative trend-following (μP) dominates")
print("      → positive feedback → prices accelerate AWAY from equilibrium")
print("    - Far from equilibrium: mean-reversion (−P³/3) dominates")
print("      → negative feedback → prices pulled BACK toward center")
print("    → perpetual boom-bust oscillations")
