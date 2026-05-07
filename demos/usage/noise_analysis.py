"""
Rydberg Noise Analysis Demo: From Spectral Density to Experimental Stability

OVERVIEW:
This script demonstrates the workflow for characterizing laser phase noise in 
Rydberg atom experiments. Based on the analysis in de Léséleuc et al. (2018) 
and Kozlej (2026), we aim to quantify how technical laser imperfections 
translate into the dephasing and damping observed in Rabi oscillations 
.

OBJECTIVES:
1. Stochastic Simulation: Convert experimental Frequency Noise PSD (Hz^2/Hz) 
   into a time-domain phase trajectory phi(t) using a 'floor' extrapolation 
   to handle DC and Nyquist boundaries.
2. Coherence Characterization: Use the Wiener-Khinchin theorem to compute the 
   autocorrelation function, revealing the characteristic coherence time 
   of the laser system.
3. Frequency Stability: Calculate the Allan Deviation (ADEV) to identify 
   the observation times (tau) where the laser is most stable, helping 
   optimize gate and pulse durations.
4. Jitter Quantification: Compute the Integrated Phase Noise (IPN) to find 
   the total RMS phase jitter. High-frequency 'servo bumps' identified in 
   the IPN are primary contributors to Rydberg state dephasing.

METHODOLOGY:
- Extrapolation: Uses 'floor' mode to extend band-limited data to the 
  full simulation bandwidth.
- Correlation: O(N log N) FFT-based autocorrelation for efficiency.
- Integration: Trapezoidal integration of the PSD from f_max downwards 
  to visualize cumulative jitter.
"""

import numpy as np
import matplotlib.pyplot as plt
import noiphi

# -- 1. Load Experimental Data & Simulate Noise --
laserData = np.genfromtxt('../data/950nm_freqNoise_blueENHANCED.csv', delimiter=',', skip_header=8)
frequencies = laserData[:, 0]
s_freq = laserData[:, 1]

s_phase = noiphi.conversion_tools.frequency_to_phase_psd(frequencies, s_freq)
sim = noiphi.core.NoiseSimulator(frequencies, s_phase, n_samples=2**17 ,extrapolation_mode='floor')
time, phi = sim.generateNoise()
dt = sim.dt

# -- 2. Perform Stability Analysis --
# FFT-based autocorrelation (Wiener-Khinchin) to find coherence time
r_tau = noiphi.analysis_tools.autocorrWK(phi)

# Allan Deviation to characterize frequency stability in the time domain
taus, adev = noiphi.analysis_tools.AllanDev(phi, dt)

# Cumulative Integrated Phase Noise for RMS jitter estimation
#ipn_rms = noiphi.analysis_tools.integrated_phase_noise(sim.f_linear_full, sim.psd_linear_full)

# 1. Calculate the slice index (half of the total samples)
half_n = len(sim.f_linear_full) // 2

# 2. Extract only the positive frequencies and corresponding PSD
# We skip index 0 because log(0) is undefined
f_pos = sim.f_linear_full[1:half_n]
psd_pos = sim.psd_linear_full[1:half_n]

# 3. Recalculate IPN using only the positive side
ipn_rms= noiphi.analysis_tools.integrated_phase_noise(f_pos, psd_pos)

# -- 3. Plotting --
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot A: Coherence Decay
axes[0].plot(time[:2000]*1e6, r_tau[:2000])
axes[0].set_title("Autocorrelation $R(\tau)$")
axes[0].set_xlabel(r"Lag ($\mu$s)")
axes[0].set_ylabel("Normalized Correlation")
axes[0].grid(True, alpha=0.3)

# Plot B: Allan Deviation
axes[1].loglog(taus, adev, 'o-', color='tab:orange')
axes[1].set_title(r"Allan Deviation $\sigma_y(\tau)$")
axes[1].set_xlabel("Observation Time (s)")
axes[1].grid(True, which="both", alpha=0.3)

# Plot C: Integrated Jitter
axes[2].loglog(f_pos, ipn_rms, color='tab:green')
axes[2].set_title("Integrated Phase Noise (RMS Jitter)")
axes[2].set_xlabel("Frequency (Hz)")
axes[2].set_ylabel("RMS Phase Error (rad)")

plt.tight_layout()
plt.show()
