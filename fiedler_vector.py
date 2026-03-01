"""
Erdős-Rényi Graph — Fiedler Vector and Algebraic Connectivity
===============================================================

The graph Laplacian L = D − A (degree matrix minus adjacency matrix)
encodes the connectivity structure of a graph. Its eigenvalues
0 = λ₁ ≤ λ₂ ≤ ... ≤ λ_n reveal how well-connected the graph is:

- λ₂ (the Fiedler value / algebraic connectivity): measures how
  "hard" it is to disconnect the graph. λ₂ = 0 iff the graph is
  disconnected.
- v₂ (the Fiedler vector): the eigenvector of λ₂, gives the optimal
  relaxed bisection of the graph. Nodes with similar v₂ components
  are "close" in the graph's connectivity structure; sign(v₂) yields
  a spectral partition that approximately minimizes the edge cut.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx


# ── Generate a connected ER graph ──

n = 50
p = 0.10
seed = 42

print(f"Generating Erdős-Rényi graph G({n}, {p})...")
np.random.seed(seed)
rng = np.random.RandomState(seed)

# Regenerate until connected
while True:
    G = nx.erdos_renyi_graph(n, p, seed=rng)
    if nx.is_connected(G):
        break

print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

# ── Compute graph Laplacian and its spectrum ──

L = nx.laplacian_matrix(G).toarray().astype(float)
eigenvalues, eigenvectors = np.linalg.eigh(L)

# Fiedler value and vector (second-smallest eigenvalue)
lambda2 = eigenvalues[1]
fiedler = eigenvectors[:, 1]

# Normalize so max |v₂| = 1 for consistent coloring
fiedler = fiedler / np.max(np.abs(fiedler))

print(f"  λ₂ (algebraic connectivity) = {lambda2:.4f}")
print(f"  Fiedler vector range: [{fiedler.min():.3f}, {fiedler.max():.3f}]")

# Spring layout (fixed seed for reproducibility)
pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(n), iterations=100)


# ══════════════════════════════════════════════════════════════════
# Figure 1: Graph colored by Fiedler vector
# ══════════════════════════════════════════════════════════════════

print("Drawing graph colored by Fiedler vector...")

fig, ax = plt.subplots(figsize=(10, 10))

# Draw edges
nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=0.8, edge_color="gray")

# Draw nodes colored by Fiedler vector component
node_colors = fiedler
vmax = 1.0
sm = plt.cm.ScalarMappable(cmap="coolwarm",
                            norm=mcolors.Normalize(vmin=-vmax, vmax=vmax))
sm.set_array([])

node_cmap = plt.cm.coolwarm
node_rgba = node_cmap(mcolors.Normalize(vmin=-vmax, vmax=vmax)(fiedler))

# Node size proportional to |v₂|
node_sizes = 100 + 400 * np.abs(fiedler)

nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_rgba,
                       node_size=node_sizes, edgecolors="black",
                       linewidths=0.5)
nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_color="black")

cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label(r"Fiedler vector component $v_{2,i}$", fontsize=11)

ax.set_title(
    f"Erdős-Rényi Graph G({n}, {p}) — Fiedler Vector\n"
    r"Nodes colored by $v_{2,i}$: "
    f"$\\lambda_2 = {lambda2:.4f}$",
    fontsize=13)
ax.axis("off")

plt.tight_layout()
plt.savefig("fiedler_graph.png", dpi=150)
print("Saved fiedler_graph.png")


# ══════════════════════════════════════════════════════════════════
# Figure 2: Spectral Partitioning
# ══════════════════════════════════════════════════════════════════

print("Drawing spectral partitioning...")

fig, (ax_graph, ax_bar) = plt.subplots(1, 2, figsize=(16, 8),
                                        gridspec_kw={"width_ratios": [1, 1]})

# ── Left: graph with binary partition ──

partition = np.sign(fiedler)
# Handle exact zeros (unlikely but possible)
partition[partition == 0] = 1

colors_part = np.where(partition > 0, "#E24A33", "#348ABD")
set_A = set(np.where(partition > 0)[0])
set_B = set(np.where(partition <= 0)[0])

# Classify edges
cut_edges = [(u, v) for u, v in G.edges()
             if (u in set_A and v in set_B) or (u in set_B and v in set_A)]
internal_edges = [(u, v) for u, v in G.edges() if (u, v) not in cut_edges]

nx.draw_networkx_edges(G, pos, edgelist=internal_edges, ax=ax_graph,
                       alpha=0.4, width=1.0, edge_color="gray")
nx.draw_networkx_edges(G, pos, edgelist=cut_edges, ax=ax_graph,
                       alpha=0.6, width=1.5, edge_color="gold",
                       style="dashed")
nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_color=colors_part,
                       node_size=200, edgecolors="black", linewidths=0.8)
nx.draw_networkx_labels(G, pos, ax=ax_graph, font_size=6)

n_cut = len(cut_edges)
ax_graph.set_title(
    f"Spectral Bisection via sign($v_2$)\n"
    f"Red: $v_{{2,i}} > 0$ ({len(set_A)} nodes), "
    f"Blue: $v_{{2,i}} < 0$ ({len(set_B)} nodes)\n"
    f"Cut edges (gold dashed): {n_cut}",
    fontsize=12)
ax_graph.axis("off")

# ── Right: sorted Fiedler vector bar chart ──

sorted_idx = np.argsort(fiedler)
sorted_fiedler = fiedler[sorted_idx]
bar_colors = np.where(sorted_fiedler > 0, "#E24A33", "#348ABD")

ax_bar.bar(range(n), sorted_fiedler, color=bar_colors, edgecolor="black",
           linewidth=0.3, width=1.0)
ax_bar.axhline(0, color="black", lw=1.0)
ax_bar.set_xlabel("Node (sorted by Fiedler component)", fontsize=11)
ax_bar.set_ylabel(r"$v_{2,i}$", fontsize=12)
ax_bar.set_title(
    "Sorted Fiedler Vector Components\n"
    "The sign change defines the spectral partition",
    fontsize=12)
ax_bar.grid(True, alpha=0.2, axis="y")

plt.tight_layout()
plt.savefig("fiedler_partition.png", dpi=150)
print("Saved fiedler_partition.png")


# ══════════════════════════════════════════════════════════════════
# Figure 3: Laplacian Spectrum — well-connected vs barely-connected
# ══════════════════════════════════════════════════════════════════

print("Computing Laplacian spectra for comparison...")

configs = [
    (n, 0.15, "Well-connected: G(50, 0.15)"),
    (n, 0.05, "Barely-connected: G(50, 0.05)"),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

for idx, (n_g, p_g, label) in enumerate(configs):
    ax = axes[idx]
    rng_g = np.random.RandomState(42 + idx)

    # Generate connected graph
    for attempt in range(100):
        G_g = nx.erdos_renyi_graph(n_g, p_g, seed=rng_g)
        if nx.is_connected(G_g):
            break
    else:
        # If we can't find a connected one, just use the last one
        print(f"  Warning: couldn't find connected graph for p={p_g}")

    L_g = nx.laplacian_matrix(G_g).toarray().astype(float)
    evals_g = np.linalg.eigvalsh(L_g)
    lambda2_g = evals_g[1]

    print(f"  {label}: λ₂ = {lambda2_g:.4f}, edges = {G_g.number_of_edges()}")

    ax.bar(range(len(evals_g)), evals_g, color="C0", alpha=0.6,
           edgecolor="C0", linewidth=0.5, width=0.8)
    # Highlight λ₂
    ax.bar(1, evals_g[1], color="C3", edgecolor="black", linewidth=1.0,
           width=0.8, zorder=5)
    ax.annotate(rf"$\lambda_2 = {lambda2_g:.4f}$",
                xy=(1, lambda2_g), xytext=(8, lambda2_g + 1),
                fontsize=11, color="C3", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="C3", lw=1.5))

    ax.set_xlabel("Eigenvalue index", fontsize=11)
    ax.set_ylabel(r"$\lambda_i$", fontsize=12)
    ax.set_title(f"{label}\n({G_g.number_of_edges()} edges)", fontsize=12)
    ax.grid(True, alpha=0.2, axis="y")

fig.suptitle(
    "Laplacian Spectrum — Algebraic Connectivity\n"
    r"$\lambda_2$ measures how well-connected the graph is "
    r"($\lambda_2 = 0$ ⟺ disconnected)",
    fontsize=13)

plt.tight_layout()
plt.savefig("fiedler_spectrum.png", dpi=150)
print("Saved fiedler_spectrum.png")


# ══════════════════════════════════════════════════════════════════
# Figure 4: Diffusion from a single node under different p
# ══════════════════════════════════════════════════════════════════

print("Computing diffusion from a single node...")

from scipy.linalg import expm

n_diff = 50
source_node = 0  # pick node 0 as source
t_max = 5.0
t_vals = np.linspace(0, t_max, 500)

p_values = [0.05, 0.10, 0.15, 0.25, 0.40]

fig, (ax_source, ax_avg) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

for p_d in p_values:
    rng_d = np.random.RandomState(42)
    # Generate connected graph
    for _ in range(200):
        G_d = nx.erdos_renyi_graph(n_diff, p_d, seed=rng_d)
        if nx.is_connected(G_d):
            break

    L_d = nx.laplacian_matrix(G_d).toarray().astype(float)
    evals_d = np.linalg.eigvalsh(L_d)
    lambda2_d = evals_d[1]

    # Initial condition: u(0) = δ_source (1 at source, 0 elsewhere)
    u0 = np.zeros(n_diff)
    u0[source_node] = 1.0

    # Eigendecomposition for efficient exp(-Lt)
    evals_full, evecs_full = np.linalg.eigh(L_d)
    # u(t) = V exp(-Λt) V^T u0
    coeffs = evecs_full.T @ u0  # projection onto eigenbasis

    u_source = np.empty(len(t_vals))
    u_max = np.empty(len(t_vals))
    u_equil = 1.0 / n_diff  # equilibrium value

    for i, t in enumerate(t_vals):
        u_t = evecs_full @ (coeffs * np.exp(-evals_full * t))
        u_source[i] = u_t[source_node]
        u_max[i] = np.max(u_t)

    label = (rf"p = {p_d}  ($\lambda_2 = {lambda2_d:.3f}$, "
             f"{G_d.number_of_edges()} edges)")
    ax_source.plot(t_vals, u_source, lw=1.8, label=label)
    ax_avg.plot(t_vals, u_max, lw=1.8, label=label)

    print(f"  p = {p_d}: λ₂ = {lambda2_d:.4f}, edges = {G_d.number_of_edges()}")

# Equilibrium line
ax_source.axhline(1.0 / n_diff, color="black", ls=":", lw=1.0, alpha=0.5)
ax_source.annotate(f"Equilibrium = 1/n = {1/n_diff:.3f}",
                   xy=(t_max * 0.7, 1.0 / n_diff + 0.005),
                   fontsize=9, color="black", alpha=0.7)
ax_avg.axhline(1.0 / n_diff, color="black", ls=":", lw=1.0, alpha=0.5)

ax_source.set_ylabel(r"$u_{\rm source}(t)$", fontsize=12)
ax_source.set_title(
    "Heat Diffusion from a Single Node on ER Graphs\n"
    r"$du/dt = -Lu$, initial condition: $u(0) = \delta_{\rm source}$",
    fontsize=13)
ax_source.set_yscale("log")
ax_source.set_ylim(1.0 / n_diff * 0.5, 1.2)
ax_source.legend(fontsize=9)
ax_source.grid(True, alpha=0.2)

ax_avg.set_xlabel("Time t", fontsize=12)
ax_avg.set_ylabel(r"$\max_i\, u_i(t)$", fontsize=12)
ax_avg.set_title(
    "Maximum node value — convergence to equilibrium\n"
    r"Decay rate governed by $\lambda_2$: larger $\lambda_2$ → faster mixing",
    fontsize=12)
ax_avg.set_yscale("log")
ax_avg.set_ylim(1.0 / n_diff * 0.5, 1.2)
ax_avg.legend(fontsize=9)
ax_avg.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("fiedler_diffusion.png", dpi=150)
print("Saved fiedler_diffusion.png")


# ══════════════════════════════════════════════════════════════════
# Figure 5: Snapshots of diffusion on the graph
# ══════════════════════════════════════════════════════════════════

print("Drawing diffusion snapshots on graph...")

# Use the main graph G (p=0.1)
L_main = nx.laplacian_matrix(G).toarray().astype(float)
evals_main, evecs_main = np.linalg.eigh(L_main)

u0_snap = np.zeros(n)
u0_snap[source_node] = 1.0
coeffs_snap = evecs_main.T @ u0_snap

snapshot_times = [0.0, 0.1, 0.3, 0.5, 1.0, 3.0]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

sm_diff = plt.cm.ScalarMappable(
    cmap="hot_r",
    norm=mcolors.LogNorm(vmin=1e-3, vmax=1.0))
sm_diff.set_array([])

for idx, t_snap in enumerate(snapshot_times):
    ax = axes[idx]
    u_snap = evecs_main @ (coeffs_snap * np.exp(-evals_main * t_snap))
    u_snap = np.clip(u_snap, 1e-4, None)  # clip for log colormap

    node_rgba_snap = plt.cm.hot_r(
        mcolors.LogNorm(vmin=1e-3, vmax=1.0)(u_snap))

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=0.6,
                           edge_color="gray")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_rgba_snap,
                           node_size=150, edgecolors="black", linewidths=0.4)
    # Highlight source
    ax.scatter(*pos[source_node], s=250, facecolors="none",
               edgecolors="lime", linewidths=2.0, zorder=10)

    ax.set_title(f"t = {t_snap}", fontsize=12)
    ax.axis("off")

fig.suptitle(
    f"Diffusion Snapshots on G({n}, {p})\n"
    r"$du/dt = -Lu$, source node circled in green"
    f"  ($\\lambda_2 = {lambda2:.4f}$)",
    fontsize=13)

# Shared colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cbar = fig.colorbar(sm_diff, cax=cbar_ax)
cbar.set_label(r"$u_i(t)$  (log scale)", fontsize=11)

plt.tight_layout(rect=[0, 0, 0.91, 0.95])
plt.savefig("fiedler_diffusion_snapshots.png", dpi=150)
print("Saved fiedler_diffusion_snapshots.png")

plt.show()

print("\n=== Fiedler Vector Summary ===")
print(f"  Graph: G({n}, {p}), {G.number_of_nodes()} nodes, "
      f"{G.number_of_edges()} edges")
print(f"  λ₂ (algebraic connectivity) = {lambda2:.4f}")
print("  Fiedler vector v₂: eigenvector of λ₂ of the graph Laplacian L = D − A")
print("  sign(v₂) gives the spectral bisection — approximate min-cut partition")
print(f"  Spectral cut: {n_cut} edges between the two groups")
print("  λ₂ → 0: graph is nearly disconnected (bottleneck)")
print("  λ₂ large: graph is well-connected (robust)")
