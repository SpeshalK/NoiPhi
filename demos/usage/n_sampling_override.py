"""
    Generates a phase noise trajectory, optionally overriding class defaults.

    Parameters
    ----------
    n_samples : int, optional
        Number of points to generate. Defaults to self.n_samples.
    dt : float, optional
        Time step in seconds. Defaults to self.dt.

    Returns
    -------
    t : ndarray
        Time axis (seconds).
    phi : ndarray
        Phase noise trajectory (radians).
"""

import numpy as np
import matplotlib.pyplot as plt
import noiphi

# 1. Setup with experimental data 
data = np.genfromtxt('../data/950nm_freqNoise_blueENHANCED.csv', delimiter=',', skip_header=8)
f, s_freq = data[:, 0], data[:, 1]
s_phase = noiphi.conversion_tools.frequency_to_phase_psd(f, s_freq)

# Initialize ONCE with default settings
sim = noiphi.core.NoiseSimulator(f, s_phase)

# 2. OVERRIDE: The "Rabi" View (High resolution, short time)
# We want to see the MHz servo bump clearly.
t1, phi1 = sim.generateNoise(n_samples=2**12, dt=1e-8) 

# 3. OVERRIDE: The long time View (Lower resolution)
# We want to see how the phase drifts over a full millisecond.
t2, phi2 = sim.generateNoise(n_samples=2**18, dt=1e-6)

# 4. Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(t1 * 1e6, phi1, color='tab:blue')
ax1.set_title(f"Rabi Scale: {len(t1)} samples @ {sim.dt*1e9}ns dt")
ax1.set_ylabel("Phase (rad)")
ax1.set_xlabel("Time (μs)")

ax2.plot(t2 * 1e3, phi2, color='tab:orange')
ax2.set_title(f"Stability Scale: {len(t2)} samples @ 1μs dt (Override)")
ax2.set_ylabel("Phase (rad)")
ax2.set_xlabel("Time (ms)")

plt.tight_layout()
plt.show()
