# 2D Phase Portrait Generator for Dynamical Systems

Phase portrait generator for 2D autonomous dynamical systems using SymPy, NumPy, and Matplotlib.

## Features

- Symbolic equilibrium finding via SymPy
- Automatic classification of fixed points (stable/unstable nodes, spirals, saddles, centers) using Jacobian eigenvalues
- Streamplot colored by flow speed with quiver field overlay
- Annotated equilibria with type and eigenvalue labels

## Usage

Edit the system at the bottom of `phase_portrait.py`:

```python
x, y = sp.symbols("x y")
f = x + sp.exp(-y)   # dx/dt
g = -y                # dy/dt
plot_phase_portrait(f, g, x, y, xlim=(-4, 4), ylim=(-4, 4))
```

Run:

```bash
python phase_portrait.py
```

## Example

System: $\dot{x} = x + e^{-y}$, $\dot{y} = -y$

![Phase Portrait](phase_portrait.png)

Equilibrium at (-1, 0) — **saddle point** with eigenvalues λ = -1, 1.

## Dependencies

- sympy
- numpy
- matplotlib
