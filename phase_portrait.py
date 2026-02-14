import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def find_equilibria(f_expr, g_expr, x, y, search_range=(-10, 10)):
    """Find equilibrium points by solving f(x,y)=0, g(x,y)=0."""
    solutions = sp.solve([f_expr, g_expr], [x, y], dict=True)
    real_equilibria = []
    for sol in solutions:
        xval = complex(sol[x])
        yval = complex(sol[y])
        if abs(xval.imag) < 1e-10 and abs(yval.imag) < 1e-10:
            real_equilibria.append((float(xval.real), float(yval.real)))
    return real_equilibria


def classify_equilibrium(f_expr, g_expr, x, y, eq_point):
    """Classify an equilibrium point using the Jacobian eigenvalues."""
    J = sp.Matrix([
        [sp.diff(f_expr, x), sp.diff(f_expr, y)],
        [sp.diff(g_expr, x), sp.diff(g_expr, y)],
    ])
    J_at_eq = J.subs({x: eq_point[0], y: eq_point[1]})
    eigenvalues = [complex(ev) for ev in J_at_eq.eigenvals().keys()]

    real_parts = [ev.real for ev in eigenvalues]

    if all(r < -1e-10 for r in real_parts):
        if all(abs(ev.imag) < 1e-10 for ev in eigenvalues):
            return "stable node", eigenvalues
        else:
            return "stable spiral", eigenvalues
    elif all(r > 1e-10 for r in real_parts):
        if all(abs(ev.imag) < 1e-10 for ev in eigenvalues):
            return "unstable node", eigenvalues
        else:
            return "unstable spiral", eigenvalues
    elif any(r > 1e-10 for r in real_parts) and any(r < -1e-10 for r in real_parts):
        return "saddle", eigenvalues
    else:
        if all(abs(r) < 1e-10 for r in real_parts):
            return "center", eigenvalues
        return "non-isolated / degenerate", eigenvalues


def plot_phase_portrait(f_expr, g_expr, x, y,
                        xlim=(-3, 3), ylim=(-3, 3), density=1.5,
                        grid_n=30, title=None):
    """
    Plot the phase portrait for the 2D system:
        dx/dt = f(x, y)
        dy/dt = g(x, y)
    """
    f_func = sp.lambdify((x, y), f_expr, modules="numpy")
    g_func = sp.lambdify((x, y), g_expr, modules="numpy")

    # --- Find and classify equilibria ---
    equilibria = find_equilibria(f_expr, g_expr, x, y)
    classifications = {}
    for eq in equilibria:
        cls, eigs = classify_equilibrium(f_expr, g_expr, x, y, eq)
        classifications[eq] = (cls, eigs)

    # --- Set up grid ---
    xs = np.linspace(xlim[0], xlim[1], grid_n)
    ys = np.linspace(ylim[0], ylim[1], grid_n)
    X, Y = np.meshgrid(xs, ys)
    U = f_func(X, Y)
    V = g_func(X, Y)

    speed = np.sqrt(U**2 + V**2)
    speed = np.where(speed == 0, 1e-10, speed)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 8))

    # Streamplot with speed-based coloring
    norm = Normalize(vmin=speed.min(), vmax=speed.max())
    strm = ax.streamplot(X, Y, U, V, color=speed, cmap="coolwarm",
                         norm=norm, density=density, linewidth=0.8,
                         arrowsize=1.2, arrowstyle='->')
    fig.colorbar(strm.lines, ax=ax, label="speed |f|")

    # Quiver (light background arrows)
    ax.quiver(X, Y, U / speed, V / speed, alpha=0.15, scale=35, width=0.003,
              color="gray")

    # --- Mark equilibria ---
    marker_style = {
        "stable node":     dict(marker="o", color="blue",   label="stable node (attractor)"),
        "stable spiral":   dict(marker="o", color="cyan",   label="stable spiral (attractor)"),
        "unstable node":   dict(marker="^", color="red",    label="unstable node (repeller)"),
        "unstable spiral": dict(marker="^", color="orange",  label="unstable spiral (repeller)"),
        "saddle":          dict(marker="s", color="green",   label="saddle"),
        "center":          dict(marker="D", color="purple",  label="center"),
    }
    seen_labels = set()
    for eq, (cls, eigs) in classifications.items():
        style = marker_style.get(cls, dict(marker="x", color="black", label=cls))
        lbl = style["label"] if style["label"] not in seen_labels else None
        seen_labels.add(style["label"])
        ax.plot(eq[0], eq[1], marker=style["marker"], color=style["color"],
                markersize=12, markeredgecolor="black", markeredgewidth=1.5,
                label=lbl, zorder=5)
        eig_str = ", ".join(f"{e:.3g}" for e in eigs)
        ax.annotate(f"{cls}\nλ = {eig_str}",
                    xy=eq, xytext=(12, 12), textcoords="offset points",
                    fontsize=8, bbox=dict(boxstyle="round,pad=0.3",
                                          fc="white", alpha=0.85),
                    arrowprops=dict(arrowstyle="->", color="black"))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    if title is None:
        title = (f"Phase Portrait\n"
                 f"$\\dot{{x}} = {sp.latex(f_expr)}$,  "
                 f"$\\dot{{y}} = {sp.latex(g_expr)}$")
    ax.set_title(title, fontsize=12)

    plt.tight_layout()
    plt.savefig("phase_portrait.png", dpi=150)
    plt.show()

    # Print summary
    print("\n=== Equilibrium Points ===")
    for eq, (cls, eigs) in classifications.items():
        eig_str = ", ".join(f"{e:.4g}" for e in eigs)
        print(f"  ({eq[0]:.4f}, {eq[1]:.4f})  →  {cls}  (eigenvalues: {eig_str})")


# ──────────────────────────────────────────────
# Example:  dx/dt = x + exp(-y),  dy/dt = -y
# ──────────────────────────────────────────────
if __name__ == "__main__":
    x, y = sp.symbols("x y")

    f = x + sp.exp(-y)      # dx/dt
    g = -y                   # dy/dt

    plot_phase_portrait(f, g, x, y, xlim=(-4, 4), ylim=(-4, 4), density=1.8)
