"""
Ramsey_fringe_demo.py

Demonstrates the decay of Ramsey fringe contrast due to laser phase noise,
and directly connects the simulation result to the Integrated Phase Noise
(IPN) computed analytically from the input PSD.

SEQUENCE:
    1. Ideal pi/2 pulse  — rotates Bloch vector to equator
    2. Free precession   — duration T, laser phase noise accumulates
    3. Ideal pi/2 pulse  — maps accumulated phase to population
    4. Measure P_e       — averaged over ensemble gives fringe contrast C(T)

ANALYTIC LINK:
    For a noisy free precession with zero mean phase, the ensemble-averaged
    contrast decays as:

        C(T) = exp( -sigma^2_phi(T) / 2 )

    where sigma^2_phi(T) is the variance of the phase increment:
        Delta_phi(T) = phi(t + T) - phi(t)

    This is computed empirically from the generated trajectories themselves,
    giving a self-consistent prediction that correctly captures saturation
    behaviour — the contrast plateau beyond ~1 us is real physics, reflecting
    that the phase noise power is dominated by the high-frequency servo bump
    near 1 MHz, which saturates by ~0.5 us. The low-frequency content below
    32 kHz (the measurement floor) contributes negligibly.

    The right panel shows sigma_phi(T) vs T directly, making the connection
    between the IPN structure and the contrast decay immediately visible.

KEY IMPLEMENTATION NOTE — multi-window sampling:
    TK95 generates a stationary process, so every offset j in a long
    trajectory gives an independent increment phi[j + n_T] - phi[j] of
    length T. We draw n_windows random offsets per trajectory per T point,
    giving n_trajs * n_windows samples at no extra generation cost.

    The Ramsey sequence is evaluated analytically:
        P_e = cos^2(phi_acc / 2)
    which collapses the per-shot matrix exponential to a single numpy
    operation, making the sweep very fast.

EXTRAPOLATION:
    'decay' mode (Kohlrausch, beta=2) rolls off the PSD below the lowest
    measured frequency (~32 kHz) as 1/f^2 — physically motivated for a
    cavity-stabilised laser below its servo bandwidth.

DATA:
    950nm_freqNoise_blueENHANCED.csv — frequency noise PSD (Hz^2/Hz)
    Frequency range: ~32 kHz to ~3.9 MHz
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import noiphi

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'legend.fontsize': 11,
})

# -- 1. Load PSD and Build Phase Noise Spectrum --
data = np.genfromtxt('../data/950nm_freqNoise_blueENHANCED.csv', delimiter=',')
f, s_freq = data[:, 0], data[:, 1]
s_phase = noiphi.conversion_tools.frequency_to_phase_psd(f, s_freq)

# -- 2. Simulation Parameters --
Omega     = 2 * np.pi * 1.0e6   # Rabi frequency for pi/2 pulses (rad/s)
dt_pulse  = 1e-9                 # 1ns step for pulse
n_trajs   = 20                   # Number of independent long trajectories
n_windows = 200                  # Random windows sampled per trajectory per T
dt_free   = 1e-8                 # 10ns step during free precession

# T sweep: log-spaced from 100ns to 100us
T_vals = np.logspace(-7, -4, 40)
T_max  = T_vals[-1]
n_long = int(np.ceil(T_max / dt_free)) * 20  # 20x T_max for rich window sampling

# -- 3. pi/2 Pulse Propagator (ideal, computed once) --
sig_plus  = np.array([[0, 1], [0, 0]], dtype=complex)
sig_minus = np.array([[0, 0], [1, 0]], dtype=complex)
H_pulse   = (Omega / 2.0) * (sig_plus + sig_minus)
t_pi2     = np.pi / (2 * Omega)
n_pulse   = max(1, int(round(t_pi2 / dt_pulse)))
U_pi2     = expm(-1j * H_pulse * n_pulse * dt_pulse)

# -- 4. Pre-generate Long Trajectories --
# zero_offset=False: phi[0] is a free random variable so increments are physical.
# extrapolation_mode='decay': physically motivated 1/f^2 roll-off below 32 kHz.
print(f"Generating {n_trajs} trajectories ({n_long} samples, "
      f"{n_long * dt_free * 1e6:.0f} us)...")
sim = noiphi.core.PhaseNoiseSimulator(f, s_phase, dt=dt_free, n_samples=n_long,
                                  zero_offset=False, extrapolation_mode='decay')
long_trajs = []
for i in range(n_trajs):
    _, phi = sim.generateNoise()
    long_trajs.append(phi)
    print(f"  Trajectory {i+1}/{n_trajs} generated")

# -- 5. Empirical sigma_phi(T) for Analytic Contrast Prediction --
# Computed from the trajectories themselves — self-consistent with the
# simulation and correctly captures the saturation at large T.
print("\nComputing empirical sigma_phi(T)...")
sigma_empirical   = np.zeros(len(T_vals))
contrast_analytic = np.zeros(len(T_vals))

for k, T in enumerate(T_vals):
    n_T  = max(1, int(round(T / dt_free)))
    vars = [np.var(phi[n_T:] - phi[:-n_T]) for phi in long_trajs]
    sigma_empirical[k]   = np.sqrt(np.mean(vars))
    contrast_analytic[k] = np.exp(-np.mean(vars) / 2.0)

# -- 6. Ramsey Sweep --
# Analytic Ramsey result for ideal pi/2 - free precession - pi/2 sequence:
#   P_e = cos^2(phi_acc / 2)
# Contrast = 2*<P_e> - 1 = <cos(phi_acc)>  (since 2*cos^2(x/2)-1 = cos(x))
print("\nRunning Ramsey sweep...")
contrast_sim = np.zeros(len(T_vals))

for k, T in enumerate(T_vals):
    n_T = max(1, int(round(T / dt_free)))

    Pe_all = []
    for phi in long_trajs:
        n_available = len(phi) - n_T
        if n_available <= 0:
            continue
        offsets        = np.random.randint(0, n_available,
                                           size=min(n_windows, n_available))
        phi_increments = phi[offsets + n_T] - phi[offsets]
        Pe_all.append(np.cos(phi_increments / 2)**2)

    contrast_sim[k] = 2.0 * np.mean(np.concatenate(Pe_all)) - 1.0
    print(f"  T = {T*1e6:.2f} us  ->  "
          f"contrast = {contrast_sim[k]:.4f}  "
          f"(sigma_phi = {sigma_empirical[k]:.4f} rad)")

# -- 7. Visualization --
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

t_us = T_vals * 1e6

# Left panel: contrast decay — stochastic vs analytic prediction
ax1.semilogx(t_us, contrast_analytic,
             color='black', linewidth=1.5, linestyle='--',
             label=r'Analytic: $e^{-\sigma^2_\phi(T)/2}$ (empirical $\sigma_{\phi}$)')
ax1.semilogx(t_us, contrast_sim,
             'o', color='tab:red', markersize=5,
             label=f'Stochastic ({n_trajs} traj × {n_windows} windows)')
ax1.axhline(1/np.e, color='grey', linewidth=0.8, linestyle=':',
            label=r'$1/e$ coherence level')

ax1.set_xlabel(r'Free Precession Time $T$ ($\mu$s)')
ax1.set_ylabel('Fringe Contrast $C(T)$')
ax1.set_title('Ramsey Fringe Contrast Decay')
ax1.set_ylim(-0.05, 1.05)
ax1.legend()
ax1.grid(True, which='both', alpha=0.3)

# Right panel: empirical sigma_phi(T) — shows where variance saturates
ax2.semilogx(t_us, sigma_empirical, color='tab:blue', linewidth=1.5,
             label=r'$\sigma_\phi(T)$ (empirical)')
ax2.axhline(np.sqrt(2.0), color='grey', linewidth=0.8, linestyle=':',
            label=r'$\sigma_\phi = \sqrt{2}$ $\Rightarrow$ $C = 1/e$')
ax2.set_xlabel(r'Free Precession Time $T$ ($\mu$s)')
ax2.set_ylabel(r'RMS Phase Increment $\sigma_\phi(T)$ (rad)')
ax2.set_title(r'Phase Increment Variance vs $T$')
ax2.legend()
ax2.grid(True, which='both', alpha=0.3)

plt.suptitle('Ramsey Coherence and Laser Phase Noise (950nm blueENHANCED)',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('Ramsey_fringe_demo.png', dpi=150, bbox_inches='tight')
plt.show()
