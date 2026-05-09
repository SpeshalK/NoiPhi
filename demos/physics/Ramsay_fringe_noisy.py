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

    where sigma^2_phi(T) is the phase variance accumulated over time T,
    obtained by integrating the phase PSD from high frequency down to 1/T:

        sigma^2_phi(T) = 2 * integral_{1/T}^{f_max} S_phi(f) df

    This is exactly what noiphi.analysis_tools.integrated_phase_noise
    computes (cumulatively from high frequency downward), so the IPN curve
    evaluated at f = 1/T gives the analytic contrast prediction directly.

    This demo shows that the stochastic ensemble and the analytic IPN
    prediction agree — validating both the simulator and the analysis tools.

DATA:
    950nm_freqNoise_blueENHANCED.csv — frequency noise PSD (Hz^2/Hz)
    Frequency range: ~32 kHz to ~3.9 MHz
    Expected coherence timescale: ~30 us (set by low-frequency edge ~32 kHz)
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
# Use the raw input data frequencies directly — NOT the extrapolated linear grid.
# The extrapolated floor adds artificial low-frequency power that dominates the
# integral and makes ipn_rms flat across all frequencies, masking the real structure.
#
# integrated_phase_noise integrates from f_max downward, so:
#   ipn_rms[0]  -> jitter integrated over full band (large T, most jitter)
#   ipn_rms[-1] -> jitter integrated over nothing  (small T, ~zero jitter)
# Contrast decays as C(T) = exp(-sigma^2_phi(T) / 2) = exp(-ipn_rms(1/T)^2 / 2)
f_pos   = f
psd_pos = s_phase
ipn_rms = noiphi.analysis_tools.integrated_phase_noise(f_pos, psd_pos)


T_analytic        = 1.0 / f_pos
contrast_analytic = np.exp(-ipn_rms**2 / 2.0)

# Sort by ascending T for clean plotting (f is ascending, so T is descending)
sort_idx          = np.argsort(T_analytic)
T_analytic        = T_analytic[sort_idx]
contrast_analytic = contrast_analytic[sort_idx]
ipn_rms_sorted    = ipn_rms[sort_idx]

# -- 3. Ramsey Simulation Parameters --
Omega   = 2 * np.pi * 1.0e6    # Rabi frequency for pi/2 pulses (rad/s)
dt_pulse = 1e-9                 # 1ns timestep for pulse (high resolution)
n_trajs  = 50

# pi/2 pulse duration: Omega * t_pi2 = pi/2
t_pi2    = np.pi / (2 * Omega)
n_pulse  = max(1, int(round(t_pi2 / dt_pulse)))

# Operators
sig_plus  = np.array([[0, 1], [0, 0]], dtype=complex)
sig_minus = np.array([[0, 0], [1, 0]], dtype=complex)
H_pulse   = (Omega / 2.0) * (sig_plus + sig_minus)
U_pi2     = expm(-1j * H_pulse * n_pulse * dt_pulse)   # Ideal pi/2 propagator

# Precession times to sweep — log spaced from 100ns to 100us
T_vals = np.logspace(-7, -4, 40)   # seconds

# -- 4. Ensemble Ramsey Sweep --
contrast_sim = np.zeros(len(T_vals))

for k, T in enumerate(T_vals):

    dt_free = 1e-8                          # 10ns timestep during free precession
    n_free  = max(1, int(round(T / dt_free)))

    # Initialise a simulator for this precession duration
    sim = noiphi.core.NoiseSimulator(f, s_phase, dt=dt_free, n_samples=n_free)

    Pe_shots = np.zeros(n_trajs)
    
    T_test = 30e-6  # 30 us
    dt_free = 1e-8
    n_free = int(round(T_test / dt_free))
    sim_test = noiphi.core.NoiseSimulator(f, s_phase, dt=dt_free, n_samples=n_free)

    phi_endpoints = []
    for _ in range(200):
        _, phi = sim_test.generateNoise()
        phi_endpoints.append(phi[-1])

    phi_endpoints = np.array(phi_endpoints)
    print(f"phi[-1] std = {np.std(phi_endpoints):.4f} rad")
    print(f"Expected sigma from IPN = {ipn_rms[0]:.4f} rad")
    print(f"Expected contrast = {np.exp(-np.std(phi_endpoints)**2 / 2):.4f}")

    for i in range(n_trajs):
        _, phi = sim.generateNoise()

        state = np.array([1, 0], dtype=complex)   # Ground state

        # First pi/2 pulse (ideal)
        state = U_pi2 @ state

        # Free precession: the total accumulated phase over the dark window
        # is phi[-1] - phi[0]. Since zero_offset=True, phi[0]=0, so the
        # accumulated phase is simply phi[-1]. Applied as a single sigma_z
        # rotation — no inner loop needed.
        phi_acc = phi[-1]
        U_free  = expm(-1j * phi_acc * np.array([[1, 0], [0, -1]], dtype=complex) / 2)
        state   = U_free @ state

        # Second pi/2 pulse (ideal)
        state = U_pi2 @ state

        Pe_shots[i] = np.abs(state[1])**2

    contrast_sim[k] = 2 * np.mean(Pe_shots) - 1.0   # Maps [0,1] -> [-1,1] contrast
    print(f"  T = {T*1e6:.2f} us  ->  contrast = {contrast_sim[k]:.4f}")

# -- 5. Visualization --
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left panel: contrast decay
ax1.semilogx(T_analytic * 1e6, contrast_analytic,
             color='black', linewidth=1.5, linestyle='--',
             label='Analytic: $e^{-\\sigma^2_\\phi(T)/2}$')
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

# Right panel: IPN vs T — same x-axis as contrast panel for direct comparison
ax2.semilogx(T_analytic * 1e6, ipn_rms_sorted, color='tab:blue', linewidth=1.5,
             label='IPN (RMS jitter)')
ax2.set_xlabel(r'Free Precession Time $T$ ($\mu$s)')
ax2.set_ylabel('RMS Phase Jitter $\sigma_\phi$ (rad)')
ax2.set_title('Integrated Phase Noise vs $T$')
ax2.grid(True, which='both', alpha=0.3)

# Mark the 1/e coherence threshold: C = 1/e when sigma^2/2 = 1 -> sigma = sqrt(2)
sigma_1e = np.sqrt(2.0)
ax2.axhline(sigma_1e, color='grey', linewidth=0.8, linestyle=':',
            label=r'$\sigma_\phi = \sqrt{2}$ ($\Rightarrow$ $C = 1/e$)')
ax2.legend()

plt.suptitle('Ramsey Coherence and Laser Phase Noise (950nm blueENHANCED)',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('Ramsey_fringe_demo.png', dpi=150, bbox_inches='tight')
plt.show()
