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

## Lotka–Volterra Predator-Prey Example

The classic predator-prey model:

$$\dot{x} = \alpha x - \beta x y, \quad \dot{y} = \delta x y - \gamma y$$

with parameters α=1, β=0.5, δ=0.25, γ=0.5.

```bash
python lotka_volterra.py
```

![Lotka–Volterra Phase Portrait](lotka_volterra.png)

**Equilibria:**
- **(0, 0)** — saddle point (eigenvalues λ = -0.5, 1). The origin is an unstable fixed point where both species are extinct; any small perturbation drives the system away.
- **(2, 2)** — center (eigenvalues λ = ±0.5i). Purely imaginary eigenvalues produce closed orbits — the populations oscillate periodically with no damping.

**Key features visible in the portrait:**
- **Green dashed line** (prey nullcline, `dx/dt = 0`): the horizontal line `y = α/β = 2`. Above it prey declines; below it prey grows.
- **Magenta dashed line** (predator nullcline, `dy/dt = 0`): the vertical line `x = γ/δ = 2`. Left of it predators decline; right of it predators grow.
- **Gray contours**: level curves of the conserved quantity `H(x,y) = δx − γ ln x + βy − α ln y`, confirming that orbits are closed.
- **Gold trajectories**: sample orbits showing the counter-clockwise cycling — prey peak is followed by predator peak with a phase lag.

## Van der Pol Oscillator Example

A nonlinear oscillator with self-sustaining oscillations:

$$\dot{x} = y, \quad \dot{y} = \mu(1 - x^2)y - x$$

with parameter μ = 1.5.

```bash
python van_der_pol.py
```

![Van der Pol Phase Portrait](van_der_pol.png)

**Equilibrium:**
- **(0, 0)** — unstable spiral (eigenvalues λ = 0.75 ± 0.661i). The origin repels all nearby trajectories outward.

**Key features visible in the portrait:**
- **Gold closed curve** (limit cycle): the unique stable periodic orbit. All trajectories — whether starting inside or outside — converge to this cycle. This is the hallmark of the Van der Pol oscillator.
- **Orange trajectory** (from inside): starts near the origin and spirals outward toward the limit cycle.
- **Blue trajectory** (from outside): starts far from the origin and spirals inward toward the limit cycle.
- **Green dashed line** (x-nullcline, `dx/dt = 0`): the line `y = 0`.
- **Magenta dashed curve** (y-nullcline, `dy/dt = 0`): the cubic `y = x / [μ(1 − x²)]`, with vertical asymptotes at x = ±1.

The coexistence of an unstable equilibrium with a stable limit cycle is a classic example of a **Hopf bifurcation** — for μ > 0 the system always settles into sustained oscillations regardless of initial conditions.

### Limit Cycles Under Different μ

![Van der Pol Limit Cycles](van_der_pol_limit_cycles.png)

The parameter μ controls the strength of nonlinear damping and dramatically shapes the limit cycle:

- **Small μ** (0.2, 0.5): the cycle is nearly circular — the oscillator behaves almost like a simple harmonic oscillator with a gentle amplitude-limiting nonlinearity.
- **Moderate μ** (1.0, 1.5, 2.0): the cycle elongates and develops visible asymmetry as the nonlinear term becomes significant.
- **Large μ** (3.0, 5.0): the cycle becomes a sharp-cornered "relaxation oscillation" — the system spends most of its time slowly drifting along the nullcline branches, punctuated by rapid jumps between them. The velocity spikes grow taller while the displacement amplitude stays near x ≈ ±2.

## Dependencies

- sympy
- numpy
- matplotlib
- scipy
