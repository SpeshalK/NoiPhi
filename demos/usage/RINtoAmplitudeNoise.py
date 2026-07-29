"""
RINtoAmplitudeNoise.py

Demonstrates the workflow of converting an experimental relative intensity
noise (RIN) spectrum in dBc/Hz into a time-domain amplitude noise trajectory
using the NoiPhi toolkit.

Unlike phase noise, amplitude noise is a stationary fluctuation about the
mean optical power. The same TK95 generator is used — only the unit
conversion path and physical interpretation differ.

KEY WORKFLOW FEATURES:
1. dBc -> LINEAR: RIN measured on a spectrum analyser is logarithmic
   (dBc/Hz). dBc_to_linear() recovers the linear RIN spectrum (1/Hz).

2. RIN -> AMPLITUDE PSD: rin_to_amplitude_psd() maps RIN (1/Hz) to an
   absolute amplitude noise PSD (W/sqrt(Hz)) via sqrt(RIN)*P0, which is
   what NoiseSimulator samples to produce a power-fluctuation trajectory.

3. STATIONARY TRAJECTORY: amplitude noise does not diffuse, so the noise
   fluctuates about zero mean (DC bin zeroed) rather than random-walking
   like phase noise.

VERIFICATION:
Plot 4 compares the PSD of the generated trajectory (ensemble-averaged
Welch estimate) against the sampled input PSD to confirm the generator
reproduces the target spectrum.

DATA:
    Nd_YAG_1064nm_RIN_dBc.csv — synthetic 1064nm Nd:YAG RIN spectrum
    (pump excess + relaxation-oscillation peak + shot-noise floor).
    Physical parameters anchored to the RP Photonics RIN example
    (https://www.rp-photonics.com/relative_intensity_noise.html).
"""

import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

# Globally set larger font sizes for all plots in this script
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
})

import noiphi

# -- Lab parameter --
P0 = 100e-3   # Mean optical power (W) — 100 mW average output

# -- Input data --
laserData = np.genfromtxt('../data/Amplitude/Nd_YAG_1064nm_RIN_dBc.csv', delimiter=',', skip_header=8)
frequencies = laserData[:, 0]
rin_dBc     = laserData[:, 1]

# -- NoiPhi conversions --

# dBc/Hz -> linear RIN (1/Hz)
rin_linear = noiphi.conversion_tools.dBc_to_linear(rin_dBc)

# linear RIN -> amplitude noise PSD (W/sqrt(Hz))
s_amp = noiphi.conversion_tools.rin_to_amplitude_psd(rin_linear, P0)

# Generate noise simulation (defaults: dt=1e-8, n_samples=100000)
laser_NoiseSim = noiphi.core.NoiseSimulator(frequencies, s_amp)

# Generate unique amplitude noise trajectories
time, phi1 = laser_NoiseSim.generateNoise()
_,    phi2 = laser_NoiseSim.generateNoise()
_,    phi3 = laser_NoiseSim.generateNoise()

phi_ensemble = np.array([phi1, phi2, phi3])

# Get linear freq grid and PSD that was used for sampling
psd_sampled = laser_NoiseSim.psd_linear
f_sampled   = laser_NoiseSim.f_linear
mask = f_sampled > 0

# -- Verification logic --
fs = 1.0 / laser_NoiseSim.dt
f_welch, s_ensemble = welch(phi_ensemble, fs=fs, nperseg=laser_NoiseSim.n_samples // 8)
s_welch = np.mean(s_ensemble, axis=0)

# -- Plotting (2x2 grid) --
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
((ax1, ax2), (ax3, ax4)) = axs

# Plot 1: Raw experimental RIN spectrum (dBc/Hz)
ax1.semilogx(frequencies, rin_dBc, color='tab:purple', label='Input RIN')
ax1.set_title("1. Input: experimental RIN")
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("RIN (dBc/Hz)")
ax1.set_xlim(frequencies[0], frequencies[-1])
ax1.grid(True, which='both', alpha=0.3)
ax1.legend()

# Plot 2: Amplitude noise PSD + standardized sampling grid
ax2.loglog(frequencies, s_amp, label='Amplitude PSD (Input)', color='red', linestyle='--')
ax2.loglog(f_sampled[mask], psd_sampled[mask], label='Sampled PSD (standardized df + extrapolation)', color='black', linestyle='--')
ax2.set_title("2. Target amplitude noise spectrum")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel(r"$S_A(f)$ (W/$\sqrt{Hz}$)")
ax2.set_xlim(f_sampled[1], f_sampled[-1])
ax2.legend(loc='lower left')
ax2.grid(True, which='both', alpha=0.3)

# Plot 3: Time domain amplitude noise trajectories
ax3.plot(time * 1e6, phi1, label='Trajectory 1', alpha=0.8, color='tab:blue')
ax3.plot(time * 1e6, phi2, label='Trajectory 2', alpha=0.8, color='tab:orange')
ax3.set_title(r"3. Amplitude noise trajectories (first 100 $\mu$s, " + f"$dt={laser_NoiseSim.dt*1e9:.0f}$ ns)")
ax3.set_xlabel(r"Time ($\mu$s)")
ax3.set_ylabel(r"Power fluctuation $\delta P(t)$ (W)")
ax3.set_xlim(0, 100)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Periodogram verification
ax4.loglog(f_sampled[mask], psd_sampled[mask], 'k--', label='Sampled PSD', alpha=0.8)
ax4.loglog(f_welch, s_welch, color='tab:red', label='Noise periodogram (Welch)', alpha=0.7)
ax4.set_title("4. PSD verification")
ax4.set_xlabel("Frequency (Hz)")
ax4.set_ylabel(r"W/$\sqrt{Hz}$")
ax4.legend()
ax4.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.show()
