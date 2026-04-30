"""
demo_freqPSDtoPhaseNoise.py

Demonstrates the workflow of converting an experimental frequency noise 
PSD (Hz^2/Hz) into a time-domain phase noise trajectory (radians) 
using the NoiPhi toolkit.
"""

import noiphi
import numpy as np
import matplotlib.pyplot as plt

from noiphi.convert import frequency_to_phase_psd

# -- Input data --

#Import data from a CSV file
laserData = np.genfromtxt('../data/795nm_freqNoise_red.csv',dtype="f4,f4",delimiter=',',skip_header=8)
frequencies=laserData['f0']
s_freq=laserData['f1']

# -- NoiPhi --

#Convert s_freq to s_phase
s_phase= frequency_to_phase_psd(frequencies,s_freq)

#Generate Noise simulation (if not specified : time-step is dt=1e-6, length of noise signal is n_samples=100000)
laser_NoiseSim = noiphi.NoiseSimulator(frequencies, s_phase)

#Generate unqiue noise trajectories
time,phi1=laser_NoiseSim.generateNoise()
_,phi2=laser_NoiseSim.generateNoise()
_,phi3=laser_NoiseSim.generateNoise()

print (phi1)

# -- Plotting --
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Time Domain Plot
ax1.plot(time * 1e6, phi1, label='Trajectory 1', alpha=0.8, color='tab:blue')
ax1.plot(time * 1e6, phi2, label='Trajectory 2', alpha=0.8, color='tab:orange')
ax1.set_title(f"Generated Phase Noise Trajectories ($dt={laser_NoiseSim.dt*1e9:.0f}$ ns)")
ax1.set_xlabel(r"Time ($\mu$s)")
ax1.set_ylabel(r"Phase $\phi(t)$ (rad)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Frequency Domain Verification
ax2.loglog(frequencies, s_phase, label='Target PSD (Input)', color='black', linestyle='--')
ax2.set_title("Target Phase Noise Spectrum")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel(r"$S_{\phi}(f)$ ($rad^2/Hz$)")
ax2.legend()
ax2.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.show()
