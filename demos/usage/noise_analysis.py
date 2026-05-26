"""
noise_analysis.py

Demonstrates the noise analysis toolkit of NoiPhi, characterising laser
phase noise in the context of Rydberg atom experiments. Based on the
analysis framework of de Léséleuc et al. (2018) and Kozlej (2026), this
script quantifies how technical laser imperfections translate into the
dephasing and damping observed in Rabi oscillations.

OBJECTIVES:
1. Stochastic Simulation: Convert experimental frequency noise PSD (Hz^2/Hz)
   into a time-domain phase trajectory phi(t) using 'floor' extrapolation
   to handle DC and Nyquist boundaries.

2. Coherence Characterisation: Use the Wiener-Khinchin theorem to compute
   the autocorrelation function, revealing the characteristic coherence
   time of the laser system.

3. Frequency Stability: Calculate the Allan Deviation (ADEV) to identify
   the observation times (tau) where the laser is most stable, helping
   optimise gate and pulse durations.

4. Jitter Quantification: Compute the Integrated Phase Noise (IPN) to find
   the total RMS phase jitter. High-frequency servo bumps identified in
   the IPN are primary contributors to Rydberg state dephasing.

METHODOLOGY:
- Extrapolation: 'floor' mode extends band-limited data to the full
  simulation bandwidth by holding the boundary PSD values constant.
- Autocorrelation: O(N log N) FFT-based computation via Wiener-Khinchin.
- Allan Deviation: Overlapping estimator for robust stability statistics.
- IPN: Trapezoidal integration of the PSD from f_max downward, using only
  positive frequencies (DC bin excluded as log(0) is undefined).

REFERENCES:
    de Léséleuc et al. (2018). Phys. Rev. A, 97.
    Kozlej (2026). PhD diss., University of Strathclyde.
"""

import numpy as np
import matplotlib.pyplot as plt
import noiphi

# -- 1. Load Experimental Data and Simulate Noise --
laserData = np.genfromtxt('../data/950nm_freqNoise_blueENHANCED.csv', delimiter=',')
frequencies = laserData[:, 0]
s_freq      = laserData[:, 1]

s_phase = noiphi.conversion_tools.frequency_to_phase_psd(frequencies, s_freq)

# PhaseNoiseSimulator pre-computes the interpolated grid at construction time.
# f_linear_full and psd_linear_full are immediately available as instance
# attributes without needing to call generateNoise() first.
sim = noiphi.core.PhaseNoiseSimulator(frequencies, s_phase,
                                       n_samples=2**17,
                                       extrapolation_mode='floor')
time, phi = sim.generateNoise()
dt = sim.dt

# Check phi is consistant with Parseval Thm.
df= 1.0 / (sim.dt * sim.n_samples)
print ('Variance of noise:', np.var(phi))
print ('sum of PSD * df (integral):', np.sum(sim.psd_linear[:sim.n_samples]) * df)

# -- 2. Stability Analysis --

# Autocorrelation via Wiener-Khinchin theorem
r_tau = noiphi.analysis_tools.autocorrWK(phi)

# Allan Deviation for time-domain frequency stability
taus, adev = noiphi.analysis_tools.AllanDev(phi, dt)

# Integrated Phase Noise — positive frequencies only.
# DC bin (index 0) is excluded because log(0) is undefined during
# interpolation. The negative-frequency half is discarded as the PSD
# is symmetric; only the positive side carries physical information.
half_n  = len(sim.f_linear) // 2
f_pos   = sim.f_linear[1:half_n]
psd_pos = sim.psd_linear[1:half_n]
ipn_rms = noiphi.analysis_tools.integrated_phase_noise(f_pos, psd_pos)


# -- 3. Plotting --
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel A: Coherence Decay
axes[0].plot(time[:2000] * 1e6, r_tau[:2000])
axes[0].set_title(r"Autocorrelation $R(\tau)$")
axes[0].set_xlabel(r"Lag ($\mu$s)")
axes[0].set_ylabel("Normalized Correlation")
axes[0].grid(True, alpha=0.3)

# Panel B: Allan Deviation
axes[1].loglog(taus, adev, 'o-', color='tab:orange')
axes[1].set_title(r"Allan Deviation $\sigma_y(\tau)$")
axes[1].set_xlabel("Observation Time (s)")
axes[1].set_ylabel(r"$\sigma_y(\tau)$")
axes[1].grid(True, which='both', alpha=0.3)

# Panel C: Integrated Phase Noise
axes[2].loglog(f_pos, ipn_rms, color='tab:green')
axes[2].set_title("Integrated Phase Noise (RMS Jitter)")
axes[2].set_xlabel("Frequency (Hz)")
axes[2].set_ylabel("RMS Phase Error (rad)")
axes[2].grid(True, which='both', alpha=0.3)

plt.suptitle('Laser Phase Noise Analysis (950nm blueENHANCED)', fontsize=13)
plt.tight_layout()
plt.show()
