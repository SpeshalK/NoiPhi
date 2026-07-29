"""
amplitude_noise_analysis.py

Demonstrates the noise analysis toolkit of NoiPhi applied to laser
amplitude (intensity) noise. Where noise_analysis.py characterises phase
noise for Rydberg experiments, this script quantifies the intensity
fluctuations that drive Rabi-frequency noise and AC-Stark-shift dephasing.

OBJECTIVES:
1. Stochastic Simulation: Convert an experimental RIN spectrum (dBc/Hz)
   into a linear amplitude noise PSD (W/sqrt(Hz)) and generate a
   time-domain power-fluctuation trajectory delta_P(t).

2. Coherence Characterisation: Use the Wiener-Khinchin theorem to compute
   the autocorrelation function of the intensity fluctuations, revealing
   the correlation timescale set by the relaxation-oscillation peak.

3. Jitter Quantification: Compute the Integrated RIN to find the cumulative
   RMS fractional intensity noise. The relaxation-oscillation peak is the
   dominant contributor, directly setting the achievable intensity stability.

METHODOLOGY:
- Extrapolation: 'floor' mode extends band-limited data to the full
  simulation bandwidth by holding the boundary PSD values constant.
- Autocorrelation: O(N log N) FFT-based computation via Wiener-Khinchin.
- Integrated RIN: Trapezoidal integration of the linear RIN PSD from
  f_max downward, using only positive frequencies (DC bin excluded).

NOTE: Allan Deviation is omitted here — it characterises frequency
stability and is meaningful for phase/frequency noise, not for the
stationary intensity fluctuations analysed in this script.

DATA:
    Nd_YAG_1064nm_RIN_dBc.csv — synthetic 1064nm Nd:YAG RIN spectrum.
    Physical parameters anchored to the RP Photonics RIN example
    (https://www.rp-photonics.com/relative_intensity_noise.html).
"""

import numpy as np
import matplotlib.pyplot as plt
import noiphi

# -- Lab parameter --
P0 = 100e-3   # Mean optical power (W)

# -- 1. Load experimental data and simulate noise --
laserData   = np.genfromtxt('../data/Amplitude/Nd_YAG_1064nm_RIN_dBc.csv', delimiter=',', skip_header=8)
frequencies = laserData[:, 0]
rin_dBc     = laserData[:, 1]

# dBc/Hz -> linear RIN (1/Hz) -> amplitude noise PSD (W/sqrt(Hz))
rin_linear = noiphi.conversion_tools.dBc_to_linear(rin_dBc)
s_amp      = noiphi.conversion_tools.rin_to_amplitude_psd(rin_linear, P0)

# NoiseSimulator pre-computes the interpolated grid at construction time.
sim = noiphi.core.NoiseSimulator(frequencies, s_amp,
                                 n_samples=2**17,
                                 extrapolation_mode='floor')
time, phi = sim.generateNoise()
dt = sim.dt

# Check phi is consistent with Parseval Thm. (convergence improves with ensemble)
df = 1.0 / (sim.dt * sim.n_samples)
print('Variance of noise:', np.var(phi))
print('sum of PSD * df (integral):', np.sum(sim.psd_linear[:sim.n_samples]) * df)

# -- 2. Analysis --

# Autocorrelation via Wiener-Khinchin theorem
r_tau = noiphi.analysis_tools.autocorrWK(phi)

# Integrated RIN — positive frequencies only. DC bin (index 0) excluded.
# The linear RIN spectrum (not the amplitude PSD) is integrated so the
# result is the dimensionless cumulative RMS fractional intensity noise.
half_n     = len(sim.f_linear) // 2
f_pos      = sim.f_linear[1:half_n]
rin_pos    = noiphi.conversion_tools.dBc_to_linear(
                 np.interp(f_pos, frequencies, rin_dBc))
irin_rms   = noiphi.analysis_tools.integrated_rin(f_pos, rin_pos)

# -- 3. Plotting --
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel A: Coherence decay
axes[0].plot(time[:2000] * 1e6, r_tau[:2000])
axes[0].set_title(r"Autocorrelation $R(\tau)$")
axes[0].set_xlabel(r"Lag ($\mu$s)")
axes[0].set_ylabel("Normalized Correlation")
axes[0].grid(True, alpha=0.3)

# Panel B: Time-domain intensity fluctuations
axes[1].plot(time[:5000] * 1e6, phi[:5000], color='tab:blue', alpha=0.8)
axes[1].set_title(r"Intensity Fluctuations $\delta P(t)$")
axes[1].set_xlabel(r"Time ($\mu$s)")
axes[1].set_ylabel("Power fluctuation (W)")
axes[1].grid(True, alpha=0.3)

# Panel C: Integrated RIN
axes[2].loglog(f_pos, irin_rms * 100, color='tab:green')
axes[2].set_title("Integrated RIN (Cumulative RMS)")
axes[2].set_xlabel("Frequency (Hz)")
axes[2].set_ylabel("RMS intensity noise (%)")
axes[2].grid(True, which='both', alpha=0.3)

plt.suptitle('Laser Amplitude Noise Analysis (1064nm Nd:YAG)', fontsize=13)
plt.tight_layout()
plt.show()
