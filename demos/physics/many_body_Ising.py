import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import expm_multiply
import noiphi
from main_Rydbuild import buildRydHamil

# -- 1. Physical Parameters --
N = 4                    # Number of atoms in the chain
Omega = 2 * np.pi * 2.0    # Rabi frequency (MHz)
Delta = 2 * np.pi * 0.0    # Detuning (MHz)
C6 = 2 * np.pi * 800.0     # Interaction strength (Van der Waals)
dt = 1e-8                # 10ns time step (to resolve ~1MHz servo bump)
n_steps = 500            # Duration of simulation

# -- 2. Build many-body Hamiltonian components --
# buildRydHamil returns (Hx_upper, Hx_lower, H_detuning, H_interaction)
# Hx_plus/minus represent the collective raising/lowering operators
Hx_plus, Hx_minus, H_delta, H_int = buildRydHamil(N, C=C6)

# -- 3. Generate Noise Trajectory with noiphi --
# Loading the real-world 'blueENHANCED' 950nm laser profile
data = np.genfromtxt('../data/950nm_freqNoise_blueENHANCED.csv', delimiter=',')
f, s_freq = data[:, 0], data[:, 1]

# Convert frequency noise to phase PSD and simulate
s_phase = noiphi.conversion_tools.frequency_to_phase_psd(f, s_freq)
sim = noiphi.core.NoiseSimulator(f, s_phase, dt=dt, n_samples=n_steps)
time, phi = sim.generateNoise()

# -- 4. Time Evolution with Global Noise --
state = np.zeros(2**N, dtype=complex)
state[0] = 1.0  # Start with all atoms in ground state |00...0>
probs_ground = []

print(f"Starting evolution for {N} atoms...")
for p in phi:
    # Construct the Hamiltonian at this time step
    # The phase noise phi(t) is applied globally to the laser drive
    H = (Omega / 2.0) * (np.exp(1j * p) * Hx_plus + np.exp(-1j * p) * Hx_minus) \
        - Delta * H_delta + H_int
    
    # Evolve the many-body state by dt
    state = expm_multiply(-1j * H * dt, state)
    
    # Record the many-body ground state population
    probs_ground.append(np.abs(state[0])**2)

# -- 5. Visualization --
plt.figure(figsize=(10, 6))
plt.plot(time * 1e6, probs_ground, label=f'{N}-atom Chain (Noisy Evolution)')
plt.xlabel('Time (μs)')
plt.ylabel(r'Ground State Population $|\langle 00...0 | \psi(t) \rangle|^2$')
plt.title('Many-Body Ising Evolution: Global Phase Noise Impact')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
#plt.savefig('many_body_ising_demo.png')
#print("Demo complete. Result saved to many_body_ising_demo.png")
