"""
Rabi_Oscill_Phase_Noisy.py

Demonstrates the impact of laser phase noise on single-atom Rabi oscillations
using an ensemble of independent noise trajectories, directly comparable to
what is measured in a real Rydberg experiment.

PHYSICS:
    A two-level atom is driven by a laser with Rabi frequency Omega. In the
    ideal case the population oscillates coherently between ground and excited
    states at frequency Omega. When the laser carries phase noise phi(t), the
    drive term acquires a stochastic phase:

        H(t) = (Omega/2) * (exp(i*phi(t)) * sigma_+ + exp(-i*phi(t)) * sigma_-)

    Each noise realisation produces a slightly different trajectory. Averaging
    over the ensemble recovers the experimentally observed signal, where
    coherent oscillations are progressively washed out by dephasing — producing
    a decaying envelope on top of the Rabi oscillations.

OUTPUT:
    - Individual noisy trajectories in faint blue (single shots)
    - Noiseless reference in black dashed
    - Ensemble mean <P_e(t)> in bold red with ±1σ band
    - Noise-induced deviation panel below

CONNECTION TO IPN:
    The dephasing rate is set by the integrated phase noise (IPN) of the laser.
    High-frequency servo bumps near 1 MHz (visible in the IPN from
    noise_analysis.py) are the primary contributors — they accumulate phase
    error on the timescale of a single Rabi cycle, directly limiting coherence.

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
sim = noiphi.core.PhaseNoiseSimulator(f, s_phase, dt=dt_free, n_samples=n_long,
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
