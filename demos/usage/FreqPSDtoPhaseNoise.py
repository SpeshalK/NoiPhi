"""
demo_freqPSDtoPhaseNoise.py

Demonstrates the workflow of converting an experimental frequency noise 
PSD (Hz^2/Hz) into a time-domain phase noise trajectory (radians) 
using the NoiPhi toolkit.

Low frequency artifact removal: Data below 30 kHz is cropped to remove the measurement 
system's white phase noise floor (+2 slope in freq-domain), ensuring 
the simulation reflects true laser dynamics.

VERIFICATION AND VALIDATION:
To ensure physical consistency, Plot 4 compares the PSD of the generated 
time-domain signal against the original input.

1. Ensemble Averaging: We generate multiple independent noise trajectories 
   and average their Power Spectral Densities (PSDs). This reduces the 
   stochastic "noise" in the Welch estimate, providing a smoother line that 
   converges to the theoretical target.

2. Normalization (Factor of 4):
    - One-sided Folding: scipy.signal.welch returns a single-sided PSD, folding 
    negative frequency power into positive bins (+3 dB).
    - TK95 Normalization: The internal generator (noise.py) uses a sqrt(2*df) 
    amplitude scaling to maintain variance during the IFFT (+3 dB).
    The resulting 6 dB (4x) offset is corrected to align with IEEE Std 1139 
    physical definitions.

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

# -- Input data --

laserData = np.genfromtxt('../data/795nm_freqNoise_red.csv',dtype="f4,f4",delimiter=',',skip_header=8)
freq_raw=laserData['f0']
s_freq_raw=laserData['f1']

# -- Crop low freq artifacts --

mask_30k = freq_raw >= 30000
frequencies = freq_raw[mask_30k]
s_freq = s_freq_raw[mask_30k]

# -- NoiPhi --

#Convert s_freq to s_phase
s_phase= noiphi.conversion_tools.frequency_to_phase_psd(frequencies,s_freq)

#Generate noise simulation (if not specified : time-step is dt=1e-6, length of noise signal is n_samples=100000)
laser_NoiseSim = noiphi.core.PhaseNoiseSimulator(frequencies, s_phase)

#Generate unqiue noise trajectories
time,phi1=laser_NoiseSim.generateNoise()
_,phi2=laser_NoiseSim.generateNoise()
_,phi3=laser_NoiseSim.generateNoise()

phi_ensemble= np.array([phi1,phi2,phi3])

#Get Linear freq grid and PSD that was used for sampling
psd_sampled=laser_NoiseSim.psd_linear_full
f_sampled=laser_NoiseSim.f_linear_full
mask=f_sampled>0

# -- Verification Logic --
fs = 1.0 / laser_NoiseSim.dt
f_welch, s_ensemble = welch(phi_ensemble, fs=fs, nperseg=laser_NoiseSim.n_samples // 8)
s_welch=np.mean(s_ensemble,axis=0)

# Normalize s_welch by a factor of 4:
# 1. A factor of 2 accounts for Welch's single-sided folding (summing +/- freq power)
# 2. A factor of 2 accounts for the internal variance normalization (sqrt(2*df)) in TK95
s_welch/=2*2

# -- Plotting (2x2 Grid) --
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
((ax1, ax2), (ax3, ax4)) = axs

# Plot 1: Raw Experimental Frequency Noise PSD
ax1.loglog(frequencies, s_freq, color='tab:green', label=r'Input $S_{\nu}(f)$')
ax1.set_title("1. Input: experimental frequency noise")
ax1.set_ylabel(r"$S_{\nu}(f)$ ($Hz^2/Hz$)")
ax1.set_xlim(frequencies[0],frequencies[-1])
ax1.grid(True, which='both', alpha=0.3)
ax1.legend()

# Plot 2: Phase noise PSD  + standard grid sampling PSD with extrapolation modes
ax2.loglog(frequencies, s_phase, label='Original PSD (Input)', color='red', linestyle='--')
ax2.loglog(f_sampled[mask], psd_sampled[mask], label='Sampled PSD (standardized df + extrapolation)', color='black', linestyle='--')
ax2.set_title("2. Target phase noise spectrum")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel(r"$S_{\phi}(f)$ ($rad^2/Hz$)")
ax2.set_xlim(f_sampled[0],f_sampled[-1])
ax2.legend(loc='lower left')
ax2.grid(True, which='both', alpha=0.3)

# Plot 3: Time domain plot of phase noise φ(t) trajectories
ax3.plot(time * 1e6, phi1, label='Trajectory 1', alpha=0.8, color='tab:blue')
ax3.plot(time * 1e6, phi2, label='Trajectory 2', alpha=0.8, color='tab:orange')
ax3.set_title(r"3. Phase noise Trajectories (first 100 $\mu$s,"+f"$dt={laser_NoiseSim.dt*1e9:.0f}$ ns)")
ax3.set_xlabel(r"Time ($\mu$s)")
ax3.set_ylabel(r"Phase $\phi(t)$ (rad)")
ax3.set_xlim(0,100)
ax3.set_ylim(-0.5,0.5)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Periodogram Verification
ax4.loglog(f_sampled[mask], psd_sampled[mask], 'k--', label='Sampled PSD', alpha=0.8)
ax4.loglog(f_welch, s_welch, color='tab:red', label='Noise periodogram (Welch)', alpha=0.7)
ax4.set_title("4. PSD verification")
ax4.set_xlabel("Frequency (Hz)")
ax4.set_ylabel(r"$rad^2/Hz$")
ax4.legend()
ax4.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.show()
