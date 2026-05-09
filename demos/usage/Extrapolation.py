"""
demo_extrapolationModes.py

Demonstrates the three extrapolation modes available in the NoiPhi toolkit:
1. 'floor': (Default) Fills out-of-band frequencies with a constant values of the first and last data points.
2. 'zero':  No extrapolation; power is zero outside the input range.
3. 'decay': Applies exponential Kohlraush decay function with user specified decay constant (default is β=2.0) 

This is critical for ensuring the IFFT has a complete frequency grid from DC 
to Nyquist, even when experimental data is band-limited.
"""

import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
import noiphi

# Global plot styling
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 16, 'axes.labelsize': 14})

# -- 1. Load and Clean Data --
laserData = np.genfromtxt('../data/795nm_freqNoise_red.csv', dtype="f4,f4", delimiter=',', skip_header=8)
# Apply 30kHz crop to remove instrumentation floor artifact
mask = laserData['f0'] >= 30000
frequencies = laserData['f0'][mask]
s_freq = laserData['f1'][mask]

# Convert to Phase Noise PSD
s_phase = noiphi.conversion_tools.frequency_to_phase_psd(frequencies, s_freq)

# -- 2. Generate 3 Independent Simulations with Different Modes --
# We use a smaller dt/larger n_samples to emphasize the extrapolation regions
modes = [ 'floor', 'decay','zero']
colors = ['tab:blue', 'tab:orange', 'tab:green']
sims = {}

for mode in modes:
    # Initialize simulator with specific extrapolation mode
    sim = noiphi.core.NoiseSimulator(frequencies, s_phase, extrapolation_mode=mode)
    time, phi = sim.generateNoise()
    
    # Perform Welch verification (using ensemble averaging logic for smoothness)
    fs = 1.0 / sim.dt
    f_w, s_w = welch(phi, fs=fs, nperseg=sim.n_samples // 8)
    
    sims[mode] = {
        'time': time, 
        'phi': phi, 
        'f_welch': f_w, 
        's_welch': s_w / 4, # Physical normalization factor
        'f_sampled': sim.f_linear_full,
        'psd_sampled': sim.psd_linear_full
    }

# -- 3. Plotting --
fig, axs = plt.subplots(2, 2, figsize=(15, 11))
((ax1, ax2), (ax3, ax4)) = axs

# Plot 1 & 2: Frequency Domain Comparisons
for i, mode in enumerate(modes):
    data = sims[mode]
    mask_f = data['f_sampled'] > 0
    
    # Plot 1: The Sampled PSDs (Linear Interpolation Grids)
    ax1.loglog(data['f_sampled'][mask_f], data['psd_sampled'][mask_f], 
               label=f'Mode: {mode}', color=colors[i], alpha=0.8)
    
    # Plot 2: The Resulting Noise PSDs (Welch)
    ax2.loglog(data['f_welch'], data['s_welch'], 
               label=f'Welch ({mode})', color=colors[i], alpha=0.6)

# Formatting Frequency Plots
ax1.set_title("1. Internal Sampled Grids (Extrapolated)")
ax2.set_title("2. Resulting Noise Spectra")
for ax in [ax1, ax2]:
    ax.loglog(frequencies, s_phase, 'k--', alpha=0.5, label='Original Data')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$S_{\phi}(f)$ ($rad^2/Hz$)")
    ax.set_ylim(bottom=1e-13)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

# Plot 3: Time Domain Trajectories
for i, mode in enumerate(modes):
    ax3.plot(sims[mode]['time'] * 1e6, sims[mode]['phi'], 
             label=mode, color=colors[i], alpha=0.7)
ax3.set_title("3. Time Domain Impact")
ax3.set_xlabel(r"Time ($\mu$s)")
ax3.set_ylabel(r"Phase $\phi(t)$ (rad)")
ax3.set_xlim(0, 100)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Zoom on Low Frequency (DC Gap)
for i, mode in enumerate(modes):
    data = sims[mode]
    mask_low = (data['f_welch'] > 0) & (data['f_welch'] < 1e5)
    ax4.loglog(data['f_welch'][mask_low], data['s_welch'][mask_low], color=colors[i])
ax4.set_title("4. Close-up: Low-Freq Behavior")
ax4.set_xlabel("Frequency (Hz)")
ax4.set_ylabel(r"$rad^2/Hz$")
ax4.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.show()
