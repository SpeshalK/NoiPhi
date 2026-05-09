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

    where sigma^2_phi(T) is the variance of the phase increment Delta_phi(T)
    = phi(T) - phi(0), obtained by integrating the phase PSD from high
    frequency down to 1/T:

        sigma^2_phi(T) = 2 * integral_{1/T}^{f_max} S_phi(f) df

    This is exactly what noiphi.analysis_tools.integrated_phase_noise
    computes (cumulatively from high frequency downward), so the IPN curve
    evaluated at f = 1/T gives the analytic contrast prediction directly.

KEY IMPLEMENTATION NOTE — long trajectory approach:
    TK95 generates a stationary process, so any single sample phi[k] is
    drawn from the same marginal distribution regardless of k. This means
    phi[-1] alone carries no information about the dark time T.

    The physically correct quantity is the phase INCREMENT:
        Delta_phi(T) = phi[n_T] - phi[0]
    whose variance genuinely grows with T as noise accumulates.

    We therefore generate n_trajs long trajectories upfront (spanning the
    full T sweep range) and index into them at each T value. This is also
    faster — trajectories are generated once, not once per T point.

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

# -- 2. Analytic Contrast Prediction from IPN --
# Use the raw input data frequencies directly — NOT the extrapolated linear
# grid from the simulator. The extrapolated floor adds artificial low-frequency
# power that dominates the integral and flattens ipn_rms across all frequencies.
#
# integrated_phase_noise integrates from f_max downward, so:
#   ipn_rms[0]  -> total jitter over full band  (largest T, most jitter)
#   ipn_rms[-1] -> jitter over nothing           (smallest T, ~zero jitter)
# Contrast: C(T) = exp(-ipn_rms(1/T)^2 / 2)
ipn_rms = noiphi.analysis_tools.integrated_phase_noise(f, s_phase)

T_analytic        = 1.0 / f                      # f ascending -> T descending
contrast_analytic = np.exp(-ipn_rms**2 / 2.0)

# Sort by ascending T for clean plotting
sort_idx          = np.argsort(T_analytic)
T_analytic        = T_analytic[sort_idx]
contrast_analytic = contrast_analytic[sort_idx]
ipn_rms_sorted    = ipn_rms[sort_idx]

# -- 3. Simulation Parameters --
Omega    = 2 * np.pi * 1.0e6   # Rabi frequency for pi/2 pulses (rad/s)
dt_pulse = 1e-9                 # 1ns step for pulse
n_trajs  = 50
dt_free  = 1e-8                 # 10ns step during free precession

# T sweep: log-spaced from 100ns to 100us
T_vals = np.logspace(-7, -4, 40)
T_max  = T_vals[-1]
n_long = int(np.ceil(T_max / dt_free)) + 1   # Samples needed to cover full sweep

# pi/2 pulse propagator (ideal, computed once)
sig_plus  = np.array([[0, 1], [0, 0]], dtype=complex)
sig_minus = np.array([[0, 0], [1, 0]], dtype=complex)
sig_z     = np.array([[1, 0], [0, -1]], dtype=complex)
H_pulse   = (Omega / 2.0) * (sig_plus + sig_minus)
t_pi2     = np.pi / (2 * Omega)
n_pulse   = max(1, int(round(t_pi2 / dt_pulse)))
U_pi2     = expm(-1j * H_pulse * n_pulse * dt_pulse)

# -- 4. Pre-generate Long Trajectories --
# Each trajectory spans T_max, giving access to phi[n_T] - phi[0]
# for any T in the sweep via simple indexing. zero_offset=False so
# that phi[0] is a free random variable and increments are physical.
print(f"Generating {n_trajs} trajectories of length {n_long} samples ({T_max*1e6:.0f} us)...")
sim = noiphi.core.NoiseSimulator(f, s_phase, dt=dt_free, n_samples=n_long,
                                  zero_offset=False)
long_trajs = []
for i in range(n_trajs):
    _, phi = sim.generateNoise()
    long_trajs.append(phi)
    print(f"  Trajectory {i+1}/{n_trajs} generated")

# -- 5. Ramsey Sweep --
contrast_sim = np.zeros(len(T_vals))

for k, T in enumerate(T_vals):
    n_T = max(1, int(round(T / dt_free)))

    Pe_shots = np.zeros(n_trajs)
    for i, phi in enumerate(long_trajs):

        # Phase INCREMENT over dark time T — this is the physically meaningful
        # quantity whose variance grows with T
        phi_acc = phi[n_T] - phi[0]

        state  = np.array([1, 0], dtype=complex)
        state  = U_pi2 @ state
        U_free = expm(-1j * phi_acc * sig_z / 2)
        state  = U_free @ state
        state  = U_pi2 @ state

        Pe_shots[i] = np.abs(state[1])**2

    contrast_sim[k] = 2 * np.mean(Pe_shots) - 1.0
    print(f"  T = {T*1e6:.2f} us  ->  contrast = {contrast_sim[k]:.4f}")

# -- 6. Visualization --
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left panel: contrast decay
ax1.semilogx(T_analytic * 1e6, contrast_analytic,
             color='black', linewidth=1.5, linestyle='--',
             label=r'Analytic: $e^{-\sigma^2_\phi(T)/2}$')
ax1.semilogx(T_vals * 1e6, contrast_sim,
             'o', color='tab:red', markersize=5,
             label=f'Stochastic ensemble (N={n_trajs})')
ax1.axhline(1/np.e, color='grey', linewidth=0.8, linestyle=':',
            label=r'$1/e$ coherence level')

ax1.set_xlabel(r'Free Precession Time $T$ ($\mu$s)')
ax1.set_ylabel('Fringe Contrast $C(T)$')
ax1.set_title('Ramsey Fringe Contrast Decay')
ax1.set_ylim(-0.1, 1.1)
ax1.legend()
ax1.grid(True, which='both', alpha=0.3)

# Right panel: IPN vs T — same x-axis for direct comparison
ax2.semilogx(T_analytic * 1e6, ipn_rms_sorted, color='tab:blue', linewidth=1.5,
             label=r'IPN: $\sigma_\phi(T)$')
ax2.axhline(np.sqrt(2.0), color='grey', linewidth=0.8, linestyle=':',
            label=r'$\sigma_\phi = \sqrt{2}$ $\Rightarrow$ $C = 1/e$')
ax2.set_xlabel(r'Free Precession Time $T$ ($\mu$s)')
ax2.set_ylabel(r'RMS Phase Jitter $\sigma_\phi$ (rad)')
ax2.set_title('Integrated Phase Noise vs $T$')
ax2.legend()
ax2.grid(True, which='both', alpha=0.3)

plt.suptitle('Ramsey Coherence and Laser Phase Noise (950nm blueENHANCED)',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('Ramsey_fringe_demo.png', dpi=150, bbox_inches='tight')
plt.show()
