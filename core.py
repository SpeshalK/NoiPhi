from noise import generate_tk95_noise

import numpy as np
from noise import generate_tk95_noise

class PhaseNoiseSimulator:
    """
    High-level orchestrator class for generating laser phase noise trajectories.

    This class handles the interpolation of experimental PSD data onto a 
    linear frequency grid and manages the execution of the TK95 algorithm.
    """

    def __init__(self, frequencies, psd, dt, n_samples):
        """
        Parameters
        ----------
        frequencies : array_like
            Input frequency axis (Hz) from experimental data.
        psd : array_like
            Input Power Spectral Density (rad^2/Hz).
        dt : float
            Desired time step for the output trajectory (seconds).
        n_samples : int
            Total number of samples to generate in the time domain.
        """
        self.frequencies = np.asarray(frequencies)
        self.psd = np.asarray(psd)
        self.dt = dt
        self.n_samples = n_samples

        # Derived parameters for the linear FFT grid
        self.fs_nyq = 1 / (2 * dt)
        self.df = 1 / (dt * n_samples)

    def _get_linear_grid(self):
        """
        Interpolates the input PSD onto a linear grid spanning [-f_Nyq, f_Nyq].
        """
        # Create linear positive frequency axis
        f_linear_pos = np.arange(0, self.fs_nyq, self.df)
        
        # Interpolate PSD (typically done in log-log space for accuracy)
        psd_interp = np.exp(np.interp(
            np.log(f_linear_pos[1:]), 
            np.log(self.frequencies), 
            np.log(self.psd),
            left=-np.inf, right=-np.inf
        ))
        
        # Prepend DC bin (set to 0)
        psd_linear_pos = np.insert(psd_interp, 0, 0)
        
        # Construct full symmetric frequency axis for TK95 as expected by FFT modules
        # fs = [0, df, ..., f_nyq, -f_nyq+df, ..., -df]
        f_linear_full = np.fft.fftfreq(self.n_samples, d=self.dt)
        psd_linear_full = np.interp(np.abs(f_linear_full), f_linear_pos, psd_linear_pos)

        return f_linear_full, psd_linear_full

    def generate(self):
        """
        Generates the time-domain phase noise trajectory.

        Returns
        -------
        t : ndarray
            Time axis (seconds).
        phi : ndarray
            Phase noise trajectory (radians).
        """
        t = np.arange(self.n_samples) * self.dt
        fs, psd_mapped = self._get_linear_grid()
        
        phi = generate_tk95_noise(fs, psd_mapped)
        
        return t, phi

def phasenoise_maker(frequencies, psd, dt=1e-6, n_samples=1000):
    """
    Functional wrapper for quick noise generation.
    """
    sim = PhaseNoiseSimulator(frequencies, psd, dt, n_samples)
    return sim.generate()
