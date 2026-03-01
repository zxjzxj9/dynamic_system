"""
Lotka–Volterra Predator-Prey Phase Portrait
============================================

    dx/dt = αx  - βxy   =  x(α - βy)     (prey)
    dy/dt = δxy - γy     =  y(δx - γ)     (predator)

Parameters used:  α = 1, β = 1.5, δ = 1.25, γ = 1
Equilibria:
    (0, 0)              — saddle
    (γ/δ, α/β) = (0.8, 0.667) — center  (conserved quantity → closed orbits)
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.integrate import solve_ivp

from phase_portrait import (find_equilibria, classify_equilibrium)


# ── Parameters ──
alpha, beta, delta, gamma_ = 1.0, 1.5, 1.25, 1.0

x, y = sp.symbols("x y")
f_expr = alpha * x - beta * x * y      # prey  dx/dt
g_expr = delta * x * y - gamma_ * y    # predator  dy/dt

f_func = sp.lambdify((x, y), f_expr, modules="numpy")
g_func = sp.lambdify((x, y), g_expr, modules="numpy")

# ── Equilibria ──
equilibria = find_equilibria(f_expr, g_expr, x, y)
classifications = {}
for eq in equilibria:
    cls, eigs = classify_equilibrium(f_expr, g_expr, x, y, eq)
    classifications[eq] = (cls, eigs)

# ── Conserved quantity  H(x,y) = δx - γ ln(x) + βy - α ln(y) ──
def H(xv, yv):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (delta * xv - gamma_ * np.log(xv)
                + beta * yv - alpha * np.log(yv))

# ── Plot ──
xlim, ylim = (0.05, 4), (0.05, 4)
grid_n = 300

xs = np.linspace(xlim[0], xlim[1], grid_n)
ys = np.linspace(ylim[0], ylim[1], grid_n)
X, Y = np.meshgrid(xs, ys)
U = f_func(X, Y)
V = g_func(X, Y)
speed = np.sqrt(U**2 + V**2)
speed = np.where(speed == 0, 1e-10, speed)

fig, ax = plt.subplots(figsize=(9, 8))

# Closed orbits via conserved quantity contours
Hval = H(X, Y)
# Choose contour levels from orbits passing through specific initial conditions
h_levels = sorted(set(
    H(x0, alpha / beta) for x0 in [0.2, 0.4, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0]
))
ax.contour(X, Y, Hval, levels=h_levels, colors="slategray",
           linewidths=0.9, alpha=0.6)

# Nullclines
ax.contour(X, Y, U, levels=[0], colors="limegreen", linewidths=2, linestyles="--")
ax.contour(X, Y, V, levels=[0], colors="magenta", linewidths=2, linestyles="--")
ax.plot([], [], color="limegreen", ls="--", lw=2,
        label=r"prey nullcline ($\dot{x}=0$): $y = \alpha/\beta$")
ax.plot([], [], color="magenta", ls="--", lw=2,
        label=r"predator nullcline ($\dot{y}=0$): $x = \gamma/\delta$")

# Streamplot
qgrid = 30
Xq, Yq = np.meshgrid(np.linspace(xlim[0], xlim[1], qgrid),
                      np.linspace(ylim[0], ylim[1], qgrid))
Uq, Vq = f_func(Xq, Yq), g_func(Xq, Yq)
sq = np.sqrt(Uq**2 + Vq**2)
sq = np.where(sq == 0, 1e-10, sq)

norm = Normalize(vmin=sq.min(), vmax=sq.max())
strm = ax.streamplot(Xq, Yq, Uq, Vq, color=sq, cmap="coolwarm",
                     norm=norm, density=1.6, linewidth=0.8,
                     arrowsize=1.2, arrowstyle="->")
fig.colorbar(strm.lines, ax=ax, label="speed")

# Trajectory examples — integrate a few orbits to show direction
eq_x, eq_y = gamma_ / delta, alpha / beta
for x0_val in [0.2, 0.5, 1.2, 2.0]:
    sol = solve_ivp(lambda t, s: [f_func(s[0], s[1]), g_func(s[0], s[1])],
                    [0, 30], [x0_val, eq_y], max_step=0.05)
    ax.plot(sol.y[0], sol.y[1], color="gold", lw=1.8, alpha=0.8,
            zorder=3)

# Mark equilibria
marker_style = {
    "saddle": dict(marker="s", color="green"),
    "center": dict(marker="D", color="purple"),
}
for eq, (cls, eigs) in classifications.items():
    # Skip origin for visual clarity (it's at the axes boundary)
    if eq[0] < xlim[0] or eq[1] < ylim[0]:
        continue
    style = marker_style.get(cls, dict(marker="x", color="black"))
    ax.plot(eq[0], eq[1], marker=style["marker"], color=style["color"],
            markersize=12, markeredgecolor="black", markeredgewidth=1.5,
            label=f"{cls} ({eq[0]:.1f}, {eq[1]:.1f})", zorder=5)
    eig_str = ", ".join(f"{e:.3g}" for e in eigs)
    ax.annotate(f"{cls}\nλ = {eig_str}",
                xy=eq, xytext=(14, 14), textcoords="offset points",
                fontsize=8, bbox=dict(boxstyle="round,pad=0.3",
                                      fc="white", alpha=0.85),
                arrowprops=dict(arrowstyle="->", color="black"))

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel("x  (prey population)")
ax.set_ylabel("y  (predator population)")
ax.set_aspect("equal")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_title(
    "Lotka–Volterra Predator-Prey Phase Portrait\n"
    r"$\dot{x} = \alpha x - \beta xy$,  "
    r"$\dot{y} = \delta xy - \gamma y$"
    f"   (α={alpha}, β={beta}, δ={delta}, γ={gamma_})",
    fontsize=11)

plt.tight_layout()
plt.savefig("lotka_volterra.png", dpi=150)
plt.show()

# Print summary
print("\n=== Lotka–Volterra Equilibrium Points ===")
for eq, (cls, eigs) in classifications.items():
    eig_str = ", ".join(f"{e:.4g}" for e in eigs)
    print(f"  ({eq[0]:.4f}, {eq[1]:.4f})  →  {cls}  (eigenvalues: {eig_str})")


# ══════════════════════════════════════════════════════════════════
# Allee Effect: Strong extinction threshold
#   dx/dt = x(α − βy)(x/θ − 1)
#   dy/dt = −y(γ − δx)
# ══════════════════════════════════════════════════════════════════

print("\n\n========================================")
print("  Allee Effect — Extinction Threshold")
print("========================================\n")

theta_values = [0.15, 0.4, 0.7]

# Fixed points analysis:
# On x-axis (y=0): x=0, x=θ (Allee threshold)
# On y-axis (x=0): y=0
# Interior: y*=α/β, x*=γ/δ (same as classic, provided x*>θ)
#   Also x=θ gives dx/dt=0 for any y → vertical nullcline at x=θ
# So nullclines for dx/dt=0: x=0, x=θ, y=α/β
# Nullcline for dy/dt=0: y=0, x=γ/δ
# Interior equilibria: (γ/δ, α/β) if γ/δ ≠ θ, plus (θ, 0)

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

for idx, theta in enumerate(theta_values):
    ax = axes[idx]
    print(f"Computing Allee model with θ = {theta}...")

    def f_allee(xv, yv, th=theta):
        return xv * (alpha - beta * yv) * (xv / th - 1)

    def g_allee(xv, yv):
        return -yv * (gamma_ - delta * xv)

    def rhs_allee(t, s, th=theta):
        xv, yv = s
        if xv < 1e-12:
            xv = 0.0
        if yv < 1e-12:
            yv = 0.0
        return [xv * (alpha - beta * yv) * (xv / th - 1),
                -yv * (gamma_ - delta * xv)]

    # Grid
    xlim_a, ylim_a = (0.01, 3.5), (0.01, 3.0)
    grid_a = 300
    xs_a = np.linspace(xlim_a[0], xlim_a[1], grid_a)
    ys_a = np.linspace(ylim_a[0], ylim_a[1], grid_a)
    Xa, Ya = np.meshgrid(xs_a, ys_a)
    Ua = f_allee(Xa, Ya)
    Va = g_allee(Xa, Ya)

    # Nullclines
    ax.contour(Xa, Ya, Ua, levels=[0], colors="limegreen",
               linewidths=2, linestyles="--")
    ax.contour(Xa, Ya, Va, levels=[0], colors="magenta",
               linewidths=2, linestyles="--")

    # Streamplot
    qgrid_a = 35
    Xq_a, Yq_a = np.meshgrid(np.linspace(xlim_a[0], xlim_a[1], qgrid_a),
                               np.linspace(ylim_a[0], ylim_a[1], qgrid_a))
    Uq_a = f_allee(Xq_a, Yq_a)
    Vq_a = g_allee(Xq_a, Yq_a)
    sq_a = np.sqrt(Uq_a**2 + Vq_a**2)
    sq_a = np.where(sq_a == 0, 1e-10, sq_a)

    norm_a = Normalize(vmin=0, vmax=np.percentile(sq_a, 95))
    ax.streamplot(Xq_a, Yq_a, Uq_a, Vq_a, color=sq_a, cmap="coolwarm",
                  norm=norm_a, density=1.8, linewidth=0.7,
                  arrowsize=1.0, arrowstyle="->")

    # Sample trajectories — survival vs extinction
    # Green: survive (start above threshold)
    # Red: go extinct (start below threshold)
    traj_ics = []
    # Above threshold — various starting points
    for x0 in [theta + 0.3, theta + 0.8, 1.5, 2.5]:
        for y0 in [0.3, 0.7, 1.0, 1.5]:
            traj_ics.append((x0, y0, "C2", 0.6))
    # Below threshold
    for x0 in [theta * 0.3, theta * 0.5, theta * 0.8]:
        for y0 in [0.3, 0.7, 1.2]:
            traj_ics.append((x0, y0, "C3", 0.6))

    for x0, y0, color, alpha_t in traj_ics:
        if x0 < xlim_a[0] or y0 < ylim_a[0]:
            continue
        sol = solve_ivp(rhs_allee, [0, 40], [x0, y0],
                        max_step=0.02, rtol=1e-8, atol=1e-10)
        # Clip to positive quadrant
        xsol = np.clip(sol.y[0], 0, None)
        ysol = np.clip(sol.y[1], 0, None)
        ax.plot(xsol, ysol, color=color, lw=1.0, alpha=alpha_t, zorder=3)

    # Mark fixed points
    # (0, 0)
    ax.plot(0, 0, "s", color="green", ms=10, mec="black", mew=1.5,
            zorder=5)
    # (θ, 0) — saddle
    ax.plot(theta, 0, "s", color="orange", ms=10, mec="black", mew=1.5,
            zorder=5)
    ax.annotate(rf"$(\theta, 0) = ({theta}, 0)$" + "\nsaddle",
                xy=(theta, 0), xytext=(theta + 0.15, 0.35),
                fontsize=8, bbox=dict(boxstyle="round,pad=0.2",
                                      fc="white", alpha=0.85),
                arrowprops=dict(arrowstyle="->", color="black"))

    # Interior equilibrium (γ/δ, α/β) if it exists and γ/δ > θ
    x_star = gamma_ / delta
    y_star = alpha / beta
    if x_star > theta:
        # Jacobian at interior point to classify
        # f = x(α-βy)(x/θ-1), g = -y(γ-δx)
        # df/dx = (α-βy)(2x/θ - 1) + ... evaluated at equilibrium
        # Use numerical Jacobian
        eps_j = 1e-6
        J = np.zeros((2, 2))
        f0 = rhs_allee(0, [x_star, y_star])
        for j in range(2):
            s_p = [x_star, y_star]
            s_p[j] += eps_j
            f_p = rhs_allee(0, s_p)
            J[0, j] = (f_p[0] - f0[0]) / eps_j
            J[1, j] = (f_p[1] - f0[1]) / eps_j
        eigs_int = np.linalg.eigvals(J)
        re_parts = eigs_int.real

        if all(r < 0 for r in re_parts):
            cls_int = "stable"
            marker_int = "o"
            color_int = "blue"
        elif all(r > 0 for r in re_parts):
            cls_int = "unstable"
            marker_int = "^"
            color_int = "red"
        elif any(r < 0 for r in re_parts) and any(r > 0 for r in re_parts):
            cls_int = "saddle"
            marker_int = "s"
            color_int = "green"
        else:
            cls_int = "center"
            marker_int = "D"
            color_int = "purple"

        if np.any(np.abs(eigs_int.imag) > 1e-6):
            cls_int = ("stable spiral" if all(r < 0 for r in re_parts)
                       else "unstable spiral" if all(r > 0 for r in re_parts)
                       else cls_int)

        ax.plot(x_star, y_star, marker=marker_int, color=color_int,
                ms=12, mec="black", mew=1.5, zorder=5)
        eig_str = ", ".join(f"{e:.3g}" for e in eigs_int)
        ax.annotate(f"({x_star:.2f}, {y_star:.2f})\n{cls_int}\nλ = {eig_str}",
                    xy=(x_star, y_star), xytext=(x_star + 0.3, y_star + 0.4),
                    fontsize=7, bbox=dict(boxstyle="round,pad=0.2",
                                          fc="white", alpha=0.85),
                    arrowprops=dict(arrowstyle="->", color="black"))

        print(f"  θ = {theta}: interior eq ({x_star:.3f}, {y_star:.3f}) "
              f"is {cls_int}, λ = {eig_str}")
    else:
        print(f"  θ = {theta}: interior eq ({x_star:.3f}, {y_star:.3f}) "
              f"is BELOW threshold (x* < θ) — no coexistence!")
        ax.annotate(f"x* = γ/δ = {x_star:.2f} < θ = {theta}\n"
                    "No coexistence!",
                    xy=(x_star, y_star), fontsize=8, color="red",
                    bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow",
                              alpha=0.9))

    # Shade extinction region
    ax.axvspan(xlim_a[0], theta, color="red", alpha=0.05)

    ax.set_xlim(xlim_a)
    ax.set_ylim(ylim_a)
    ax.set_xlabel("x  (prey)", fontsize=11)
    if idx == 0:
        ax.set_ylabel("y  (predator)", fontsize=11)
    ax.set_title(rf"$\theta = {theta}$", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

# Legend for the whole figure
axes[0].plot([], [], color="limegreen", ls="--", lw=2, label=r"$\dot{x}=0$ nullcline")
axes[0].plot([], [], color="magenta", ls="--", lw=2, label=r"$\dot{y}=0$ nullcline")
axes[0].plot([], [], color="C2", lw=1.5, label="Survival trajectories")
axes[0].plot([], [], color="C3", lw=1.5, label="Extinction trajectories")
axes[0].legend(fontsize=8, loc="upper right")

fig.suptitle(
    "Lotka–Volterra with Allee Effect (Extinction Threshold)\n"
    r"$\dot{x} = x(\alpha - \beta y)(x/\theta - 1)$,  "
    r"$\dot{y} = -y(\gamma - \delta x)$"
    f"   (α={alpha}, β={beta}, δ={delta}, γ={gamma_})",
    fontsize=13, y=1.02)

plt.tight_layout()
plt.savefig("lotka_volterra_allee.png", dpi=150, bbox_inches="tight")
print("Saved lotka_volterra_allee.png")

plt.show()

print("\n=== Allee Effect Summary ===")
print(f"  Parameters: α={alpha}, β={beta}, δ={delta}, γ={gamma_}")
print(f"  Interior equilibrium: ({gamma_/delta:.3f}, {alpha/beta:.3f})")
print(f"  Threshold values tested: {theta_values}")
print("  Below threshold (x < θ): prey decline → both species go extinct")
print("  Above threshold (x > θ): coexistence possible if equilibrium is stable")
print("  The Allee effect breaks the closed orbits of classic Lotka-Volterra")
