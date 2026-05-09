"""
Rabi_Oscill_Ensemble.py

Demonstrates the effect of laser phase noise on single-atom Rabi oscillations
using an ensemble of independent noise trajectories.

Each trajectory is a unique stochastic realisation of the same experimental
PSD. Averaging over the ensemble recovers the experimentally measured signal,
where coherent oscillations are washed out by dephasing to produce a smooth
decay — directly analogous to what is observed in a real Rydberg experiment.

OUTPUT:
- Individual trajectories plotted in light grey (each a single shot)
- Ensemble mean <P_e(t)> plotted in bold
- ±1 standard deviation band shaded around the mean
- Noiseless reference plotted for comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import noiphi

# -- 1. Load PSD and Initialise Simulator --
data = np.genfromtxt('../data/950nm_freqNoise_blueENHANCED.csv', delimiter=',')
f, s_freq = data[:, 0], data[:, 1]
s_phase = noiphi.conversion_tools.frequency_to_phase_psd(f, s_freq)

# 10ns resolution to capture the 1MHz servo bump; 5000 steps = 50μs
dt       = 1e-8
n_steps  = 5000
n_trajs  = 50        # Number of independent noise realisations

sim = noiphi.core.NoiseSimulator(f, s_phase, dt=dt, n_samples=n_steps)
time = np.arange(n_steps) * dt

# -- 2. Physics Setup --
Omega = 2 * np.pi * 1.0e6  # 1 MHz Rabi frequency

sig_plus  = np.array([[0, 1], [0, 0]], dtype=complex)
sig_minus = np.array([[0, 0], [1, 0]], dtype=complex)
H_clean   = (Omega / 2.0) * (sig_plus + sig_minus)
U_clean   = expm(-1j * H_clean * dt)   # Constant, compute once

# -- 3. Noiseless Reference --
state_ref = np.array([1, 0], dtype=complex)
probs_ref = []
for _ in range(n_steps):
    state_ref = U_clean @ state_ref
    probs_ref.append(np.abs(state_ref[1])**2)
probs_ref = np.array(probs_ref)

# -- 4. Ensemble Evolution --
all_probs = np.zeros((n_trajs, n_steps))

for i in range(n_trajs):
    _, phi = sim.generateNoise()
    state  = np.array([1, 0], dtype=complex)

    for j, p in enumerate(phi):
        H = (Omega / 2.0) * (np.exp(1j * p) * sig_plus + np.exp(-1j * p) * sig_minus)
        state = expm(-1j * H * dt) @ state
        all_probs[i, j] = np.abs(state[1])**2

    print(f"  Trajectory {i+1}/{n_trajs} complete")

ensemble_mean = np.mean(all_probs, axis=0)
ensemble_std  = np.std(all_probs,  axis=0)

# -- 5. Visualization --
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
t_us = time * 1e6

# Top panel: individual trajectories + ensemble mean
for i in range(n_trajs):
    ax1.plot(t_us, all_probs[i], color='steelblue', alpha=0.12, linewidth=0.7)

ax1.plot(t_us, probs_ref,      color='black',     linewidth=1.2, linestyle='--', label='Noiseless')
ax1.plot(t_us, ensemble_mean,  color='tab:red',   linewidth=1.8, label=f'Ensemble mean (N={n_trajs})')
ax1.fill_between(t_us,
                 ensemble_mean - ensemble_std,
                 ensemble_mean + ensemble_std,
                 color='tab:red', alpha=0.2, label=r'$\pm 1\sigma$')

ax1.set_ylabel("Excited State Population $P_e(t)$")
ax1.set_title("Single-Atom Rabi Oscillations Under Laser Phase Noise")
ax1.set_ylim(-0.05, 1.15)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Bottom panel: noise-induced deviation from ideal
deviation = ensemble_mean - probs_ref
ax2.plot(t_us, deviation, color='tab:orange', linewidth=1.5, label='Mean deviation from noiseless')
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax2.fill_between(t_us, deviation, 0, color='tab:orange', alpha=0.2)
ax2.set_xlabel(r"Time ($\mu$s)")
ax2.set_ylabel(r"$\langle P_e \rangle - P_e^{\mathrm{clean}}$")
ax2.set_title("Noise-Induced Deviation")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Rabi_Oscill_Ensemble.png', dpi=150)
plt.show()
