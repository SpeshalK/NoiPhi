"""
VoltagePSDtoPhaseNoise.py

Analyzes and simulates phase noise for the 918nm Diode laser system. 
This script serves as the primary template for processing raw Spectrum 
Analyzer data into a calibrated time-domain noise trajectory.

KEY WORKFLOW FEATURES:
1. UNIT NORMALIZATION: Converts logarithmic dBm data to linear Voltage PSD (V^2/Hz). 
   This step correctly accounts for the Resolution Bandwidth (RBW) to ensure 
   the noise floor reflects physical density rather than instrument settings.

2. PDH DISCRIMINATOR MODELING: Converts Voltage PSD to Phase Noise PSD (rad^2/Hz) 
   by accounting for the frequency-dependent slope (K) of the Pound-Drever-Hall 
   error signal and the cavity linewidth (FWHM) [Schmid, F. et al. (2021) arXiv:2102.07422]

3. ARTIFACT-AWARE STITCHING: Merges low and high-frequency spans. 
   - Uses [2:] indexing on high-freq data to discard initial calibration spikes  
   - Employs a 'hard-cut' transition to prevent double-counting noise in 
     the overlap region.

4. SIMULATION STABILITY: Uses 'zero_offset' logic in the PhaseNoiseSimulator to 
   force the phase trajectory to start at 0 rad, preventing non-physical 
   initial phase jump.
"""

import numpy as np
import matplotlib.pyplot as plt

import noiphi

# --- Lab Parameters (918nm Diode Setup)---
K0 = 2.3e-4          # DC discriminator slope (V/Hz)
DfFWHM918 = 75e3     # Cavity linewidth (Hz)
OHM = 50.0           # System impedance (Ohms)
RBW_LOW = 10.0       # Resolution Bandwidth for low-freq file (Hz)
RBW_HIGH = 3000.0    # Resolution Bandwidth for high-freq file (Hz)

# 1. Load Data (User-handled generalization)
low_data = np.genfromtxt("../data/Diode918nm_VoltageNoise_lowfreq.csv", delimiter=',', skip_header=45)
high_data = np.genfromtxt("../data/Diode918nm_VoltageNoise_highfreq.csv", delimiter=',', skip_header=45)

f_l, dbm_l = low_data[:, 0], low_data[:, 1]
f_h, dbm_h = high_data[:, 0], high_data[:, 1]

# 2. Scale Voltage PSD from dBm to linear (V^2/Hz)
sv_l = noiphi.conversion_tools.dBm_to_Voltage_psd(dbm_l, rbw=RBW_LOW, impedance=OHM)
sv_h = noiphi.conversion_tools.dBm_to_Voltage_psd(dbm_h, rbw=RBW_HIGH, impedance=OHM)

# 3. Convert to Phase Noise PSD (rad^2/Hz)
# Accounts for frequency-dependent PDH discriminator slope roll-off
s_phi_l = noiphi.conversion_tools.voltage_to_phase_psd(f_l, sv_l, K0, DfFWHM918)
s_phi_h = noiphi.conversion_tools.voltage_to_phase_psd(f_h, sv_h, K0, DfFWHM918)

# 4. Stitch Spectra
f_final, s_phi_final = noiphi.conversion_tools.stitch_psds(f_l, s_phi_l, f_h[2:], s_phi_h[2:], transition_freq=1e4)

# 5. Generate Noise Trajectory
sim = noiphi.core.PhaseNoiseSimulator(f_final, s_phi_final, extrapolation_mode='decay')
t, phi = sim.generateNoise()

# --- Visualization ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Frequency Domain: Stitched PSD
ax1.loglog(f_l, s_phi_l, label='Low-Freq (RBW=10Hz)', alpha=0.6)
ax1.loglog(f_h, s_phi_h, label='High-Freq (RBW=3kHz)', alpha=0.6)
ax1.loglog(f_final, s_phi_final, 'k--', label='Stitched Result', linewidth=1)
ax1.set_title("918nm Diode: Stitched Phase Noise PSD")
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel(r"$S_{\phi}$ ($rad^2/Hz$)")
ax1.legend()
ax1.grid(True, which="both", alpha=0.3)

# Time Domain: Generated Phase Noise
ax2.plot(t[:2000] * 1e6, phi[:2000])
ax2.set_title("Generated Phase Noise Trajectory (Snippet)")
ax2.set_xlabel(r"Time ($\mu$s)")
ax2.set_ylabel("Phase (rad)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
