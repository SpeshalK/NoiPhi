"""
many_body_Ising_grEvo.py

Simulates the ground state population dynamics of an N-atom Rydberg chain
under global laser phase noise, using an ensemble of independent noise
trajectories to recover the experimentally measured average signal.

Note on trajectory count:
    The many-body Hilbert space scales as 2^N, making each expm_multiply
    call significantly more expensive than the single-qubit case. n_trajs550
    is a practical default for a demo; increase cautiously for larger N.

OUTPUT:
- Individual noisy trajectories in light blue
- Noiseless reference in black dashed
- Ensemble mean in bold red with ±1σ band
- Noise-induced deviation panel below
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import expm_multiply
import noiphi
from main_Rydbuild import buildRydHamil

# -- 1. Physical Parameters --
N        = 4
V_int    = 2 * np.pi * 0.0 * 1e6   # Interaction strength (rad/s)
Omega    = 2 * np.pi * 2.0 * 1e6   # Rabi frequency (rad/s)
Delta    = 2 * np.pi * 0.0 * 1e6   # Detuning (rad/s)
dt       = 1e-8                     # 10ns timestep
n_steps  = 500                      # Total timesteps
n_trajs  = 50                       # See note above
time     = np.arange(n_steps) * dt

# -- 2. Build Many-Body Hamiltonian Components --
Hx_plus, Hx_minus, H_delta, H_int = buildRydHamil(N, C=3)

H_static        = -(Delta * H_delta) + (V_int * H_int)
H_dynamic_clean = (Omega / 2.0) * (Hx_plus + Hx_minus)
H_clean         = H_static + H_dynamic_clean   # Constant — compute once

# -- 3. Initialise Noise Simulator --
data = np.genfromtxt('../data/950nm_freqNoise_blueENHANCED.csv', delimiter=',')
f, s_freq = data[:, 0], data[:, 1]
s_phase = noiphi.conversion_tools.frequency_to_phase_psd(f, s_freq)
sim = noiphi.core.PhaseNoiseSimulator(f, s_phase, dt=dt, n_samples=n_steps)

# -- 4. Noiseless Reference --
state_ref = np.zeros(2**N, dtype=complex)
state_ref[0] = 1.0
probs_ref = []
for _ in range(n_steps):
    state_ref = expm_multiply(-1j * H_clean * dt, state_ref)
    probs_ref.append(np.abs(state_ref[0])**2)
probs_ref = np.array(probs_ref)

# -- 5. Ensemble Evolution --
all_probs = np.zeros((n_trajs, n_steps))

for i in range(n_trajs):
    print(f"Trajectory {i+1}/{n_trajs}...")
    _, phi = sim.generateNoise()

    state = np.zeros(2**N, dtype=complex)
    state[0] = 1.0

    for j, p in enumerate(phi):
        H_noisy = H_static + (Omega / 2.0) * (
            np.exp( 1j * p) * Hx_plus +
            np.exp(-1j * p) * Hx_minus
        )
        state = expm_multiply(-1j * H_noisy * dt, state)
        all_probs[i, j] = np.abs(state[0])**2

ensemble_mean = np.mean(all_probs, axis=0)
ensemble_std  = np.std(all_probs,  axis=0)

# -- 6. Visualization --
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
t_us = time * 1e6

# Top panel: trajectories + mean + reference
for i in range(n_trajs):
    ax1.plot(t_us, all_probs[i], color='steelblue', alpha=0.2, linewidth=0.7)

ax1.plot(t_us, probs_ref,     color='black',   linewidth=1.2, linestyle='--', label='Noiseless')
ax1.plot(t_us, ensemble_mean, color='tab:red', linewidth=1.8, label=f'Ensemble mean (N_traj={n_trajs})')
ax1.fill_between(t_us,
                 ensemble_mean - ensemble_std,
                 ensemble_mean + ensemble_std,
                 color='tab:red', alpha=0.2, label=r'$\pm 1\sigma$')

ax1.set_ylabel(r'Ground State Population $|\langle 00...0 | \psi \rangle|^2$')
ax1.set_title(f'Many-Body Ising Evolution: Global Phase Noise Impact ({N}-atom chain)')
ax1.set_ylim(-0.05, 1.15)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Bottom panel: noise-induced deviation
deviation = ensemble_mean - probs_ref
ax2.plot(t_us, deviation, color='tab:orange', linewidth=1.5, label='Mean deviation from noiseless')
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax2.fill_between(t_us, deviation, 0, color='tab:orange', alpha=0.2)
ax2.set_xlabel(r'Time ($\mu$s)')
ax2.set_ylabel(r'$\langle P_{00} \rangle - P_{00}^{\mathrm{clean}}$')
ax2.set_title('Noise-Induced Deviation')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('many_body_ising_ensemble.png', dpi=150)
plt.show()
