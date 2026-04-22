"""
demo_freqPSDtoPhaseNoise.py

Demonstrates the workflow of converting an experimental frequency noise 
PSD (Hz^2/Hz) into a time-domain phase noise trajectory (radians) 
using the NoiPhi toolkit.
"""

import noiphi
import numpy as np
import matplotlib as plt

from noiphi.convert import frequency_to_phase_psd

#Import data from a CSV file
redLaserData = np.genfromtxt('./data/795nm_freqNoise_red.csv',dtype="f4,f4",delimiter=',',skip_header=8)
frequencies=redLaserData[:,0]
s_freq=redLaserData[:,1]

#Convert s_freq to s_phase
s_phase= frequency_to_phase_psd(frequencies,s_freq)

#Generate Noise simulation
sim = noiphi.NoiseSimulator(frequencies, psd, dt, n_samples)
