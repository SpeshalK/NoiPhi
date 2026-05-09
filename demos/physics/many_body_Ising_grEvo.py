import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import expm_multiply
import noiphi
from main_Rydbuild import buildRydHamil

# -- 1. Physical Parameters --
N = 4                      # Number of atoms in the chain
V_int = 2 * np.pi * 0.0 * 1e6 # MHz - Scaling factor for interactions
Omega = 2 * np.pi * 2.0 * 1e6  # Rabi frequency (MHz)
Delta = 2 * np.pi * 0.0 * 1e6    # Detuning (MHz)
dt = 1e-8                  # 10ns time step (to resolve ~1MHz servo bump)
n_steps = 500              # Duration of simulation
time = np.arange(n_steps)*dt

# -- 2. Build many-body Hamiltonian components --
Hx_plus, Hx_minus, H_delta, H_int = buildRydHamil(N, C=3)

# -- 3. Generate Noise Trajectory with noiphi --
# Loading the real-world 'blueENHANCED' 950nm laser profile
data = np.genfromtxt('../data/950nm_freqNoise_blueENHANCED.csv', delimiter=',')
f, s_freq = data[:, 0], data[:, 1]

# Convert frequency noise to phase PSD and simulate
s_phase = noiphi.conversion_tools.frequency_to_phase_psd(f, s_freq)
sim = noiphi.core.NoiseSimulator(f, s_phase, dt=dt, n_samples=n_steps)
fulltime, phi = sim.generateNoise()
print (phi)

# -- 4. Time Evolution with Global Noise --
state_clean = np.zeros(2**N, dtype=complex)
state_noisy = np.zeros(2**N, dtype=complex)
state_clean[0],state_noisy[0] = 1.0,1.0  # Initial ground state
p_gr_clean,p_gr_noisy = [],[]

print(f"Starting evolution for {N} atoms...")
H_static= - (Delta * H_delta) +  (V_int * H_int)
H_dynamic_clean = (Omega / 2.0) * (Hx_plus +  Hx_minus) 

for p in phi:
    # Construct the Hamiltonian
    # The phase noise phi(t) is applied globally to the laser drive
    H_dynamic_noisy = (Omega / 2.0) * (np.exp(1j * p) * Hx_plus + np.exp(-1j * p) * Hx_minus) 

    H_clean=H_static+H_dynamic_clean
    H_noisy=H_static+H_dynamic_noisy
            
    # Evolve the many-body state by dt
    state_clean = expm_multiply(-1j * H_clean * dt, state_clean)
    state_noisy = expm_multiply(-1j * H_noisy * dt, state_noisy)

    # Record the many-body ground state population
    p_gr_clean.append(np.abs(state_clean[0])**2)
    p_gr_noisy.append(np.abs(state_noisy[0])**2)


# -- 5. Visualization --
plt.figure(figsize=(10, 6))
plt.plot(time * 1e6, p_gr_clean, label=f'{N}-atom Chain')
plt.plot(time * 1e6, p_gr_noisy, label=f'{N}-atom Chain (Noisy Evolution)')
plt.xlabel('Time (μs)')
plt.ylabel(r'Ground State Population $|\langle 00...0 | \psi(t) \rangle|^2$')
plt.title('Many-Body Ising Evolution: Global Phase Noise Impact')
plt.legend()
plt.ylim(0,1.1)
plt.grid(True, alpha=0.3)
plt.show()
#plt.savefig('many_body_ising_demo.png')
#print("Demo complete. Result saved to many_body_ising_demo.png")
