# Coupled Oscillators — Phase and Frequency Locking

## Model

Two Kuramoto oscillators:

```
dθ₁/dt = ω₁ + K sin(θ₂ − θ₁)
dθ₂/dt = ω₂ + K sin(θ₁ − θ₂)
```

Parameters: ω₁ = 1.0, ω₂ = 1.5, Δω = 0.5. Critical coupling K_c = |Δω|/2 = 0.25.

## Figures

### Figure 1 — Phase Locking Time Series (`coupled_phase_locking.png`)
- 3 rows × 1 column
- K = 0.1 (below K_c), K = 0.25 (critical), K = 0.6 (above K_c)
- Each panel: Δθ(t) = θ₁(t) − θ₂(t) over ~100 time units
- Below K_c: Δθ drifts; above K_c: Δθ converges to constant

### Figure 2 — Frequency Locking Transition (`coupled_freq_locking.png`)
- Single panel, sweep K from 0 to 1.0 (~500 steps)
- Measure effective frequency via (θ(t_end) − θ(t_start)) / Δt over last 100 time units
- Plot ω₁_eff and ω₂_eff vs K; vertical dashed line at K_c = 0.25

### Figure 3 — Arnold Tongue (`coupled_arnold_tongue.png`)
- Analytical: locking when K ≥ |Δω|/2
- Grid of (Δω, K), color locked vs unlocked
- Overlay theoretical boundary K = |Δω|/2

## Files
- Create: `coupled_oscillators.py`
- Generate: `coupled_phase_locking.png`, `coupled_freq_locking.png`, `coupled_arnold_tongue.png`
- Modify: `README.md`
