import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import noiphi

# -- 1. Load Enhanced Noise and Generate Trajectory --
# We use the enhanced 950 (blue) laser which has a significant servo bump
data = np.genfromtxt('../data/950nm_freqNoise_blueENHANCED.csv', delimiter=',')
f, s_freq = data[:, 0], data[:, 1]
s_phase = noiphi.conversion_tools.frequency_to_phase_psd(f, s_freq)

# Setup simulator: 10ns resolution to capture the 1MHz servo bump
sim = noiphi.core.NoiseSimulator(f, s_phase, dt=1e-8, n_samples=1000)
time, phi = sim.generateNoise()

# -- 2. Physics Setup --
Omega = 2 * np.pi * 1.0e6
state = np.array([1, 0], dtype=complex) # Initial ground state
probs_e = []

# Raising operator (Ground to Excited)
sig_plus = np.array([[0, 1], 
                     [0, 0]], dtype=complex)

# Lowering operator (Excited to Ground)
sig_minus = np.array([[0, 0], 
                      [1, 0]], dtype=complex)

# -- 3. Time Evolution --
dt = sim.dt
for p in phi:

    H = (Omega / 2.0) * (np.exp(1j * p) * sig_plus + np.exp(-1j * p) * sig_minus)
    
    # Evolve one small step
    U = expm(-1j * H * dt)
    state = U @ state
    probs_e.append(np.abs(state[1])**2)

# -- 4. Visualization --
plt.figure(figsize=(10, 4))
plt.plot(time * 1e6, probs_e, label="Noisy Rabi")
plt.title("Impact of strong laser phase noise on single-atom rabi oscillations")
plt.xlabel("Time (μs)")
plt.ylabel("Excited State Population")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
