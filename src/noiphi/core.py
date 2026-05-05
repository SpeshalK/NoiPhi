from .noise import generate_tk95_noise

import numpy as np
from .noise import generate_tk95_noise

# Default to 10 ns (100 MHz sampling) to capture high-frequency phase jitters up to 50MHz (NS sampling theorem)
DEFAULT_DT = 1e-8  
# Default to 100,000 samples (~1 ms of total time) 
# Enough to see low-freq drift while keeping FFTs fast
DEFAULT_N_SAMPLES = 100_000

class NoiseSimulator:
    """
    High-level orchestrator class for generating laser noise trajectories.

    This class handles the interpolation of experimental PSD data onto a 
    linear frequency grid and manages the execution of the TK95 algorithm.

    The generateNoise() method returns a unique time-domain phase noise trajectory
    in units of radians.

    NOTE: Method is agnostic about units dataType of PSD. (Phase,Voltage,frequency PSD return their corresponding noise arrays)
    """

    def __init__(self, frequencies, psd, dt=DEFAULT_DT, n_samples=DEFAULT_N_SAMPLES,extrapolation_mode='floor',beta=2.0,zero_offset=True):
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
        extrapolation_mode : str
            Keyword for handling frequencies outside of data range: 'zero','floor', or 'decay'(Kohlrausch)
        beta : float
            Decay exponent (Kohlrausch) if extrapolation_mode set to 'decay' (preset to 2.0 for 1/f^2 decay)
        """

        if len(frequencies) != len(psd):
            raise ValueError("Frequencies and PSD arrays must have the same length.")

        self.frequencies = np.asarray(frequencies)
        self.psd = np.asarray(psd)
        self.dt = dt
        self.n_samples = n_samples

        self.extrapolation_mode = extrapolation_mode.lower()
        self.beta = beta

        self.fs_nyq = 1 / (2 * dt)
        self.df = 1 / (dt * n_samples)

        self.zero_offset = zero_offset

    def _interpolate_log_psd(self, f_target):
        """
        Maps the experimental PSD to target frequencies using log-log interpolation
        and the specified extrapolation mode.

        f_target: array of positive frequencies (excluding DC) for user
                  frequency and PSD data to be interpolated onto.
        """

        log_f_target = np.log(f_target)

        log_f_data = np.log(self.frequencies)
        log_psd_data = np.log(self.psd)

        # Set boundary values for np.interp based on mode
        if self.extrapolation_mode == 'zero':
            l_val, r_val = -np.inf, -np.inf
        elif self.extrapolation_mode == 'decay':
            l_val, r_val = log_psd_data[0], -np.inf # Decay handled manually below
        else: # Default to 'floor'
            l_val, r_val = log_psd_data[0], log_psd_data[-1]

        log_psd_interp = np.interp(
            log_f_target, log_f_data, log_psd_data, 
            left=l_val, right=r_val
        )

        # Apply Kohlrausch/Power-law decay for the 'right' side if mode is 'decay'
        if self.extrapolation_mode == 'decay':
            high_freq_mask = f_target > self.frequencies[-1]
            log_f_last = log_f_data[-1]
            log_psd_last = log_psd_data[-1]
            
            # S(f) = S_last * (f/f_last)^-beta  => log(S) = log(S_last) - beta * log(f/f_last)
            log_psd_interp[high_freq_mask] = log_psd_last - self.beta * (
                log_f_target[high_freq_mask] - log_f_last
            )
    

        return np.exp(log_psd_interp)

    def _get_linear_grid(self):
        """
        Interpolates the input PSD onto a linear grid spanning [-f_Nyq, f_Nyq].
        """
        # Create linear positive frequency axis (excluding DC)
        f_linear_pos = np.arange(self.df, self.fs_nyq, self.df)
        
        # Interpolate
        psd_interp = self._interpolate_log_psd(f_linear_pos)

        # Prepend DC bin (set to 0)
        psd_linear_pos = np.insert(psd_interp, 0, 0)
        f_linear_pos_with_dc = np.insert(f_linear_pos, 0, 0)
        
        # Construct full symmetric frequency axis for TK95 as expected by FFT modules
        # fs = [0, df, ..., f_nyq, -f_nyq+df, ..., -df]
        f_linear_full = np.fft.fftfreq(self.n_samples, d=self.dt)
        psd_linear_full = np.interp(np.abs(f_linear_full), f_linear_pos_with_dc, psd_linear_pos)

        # Store linear f and psd as attributes
        self.f_linear_full = f_linear_full
        self.psd_linear_full = psd_linear_full

        return f_linear_full, psd_linear_full
    
    def generateNoise(self):
        """
        Generates the time-domain phase noise trajectory
        using TK95 algorithm.

        Applies centering of initial value of trajectory to 
        0 if: zero_offset=True.

        Returns
        -------
        t : ndarray
            Time axis (seconds).
        phi : ndarray
            Noise trajectory (radians).
        """
        t = np.arange(self.n_samples) * self.dt
        fs, psd_mapped = self._get_linear_grid()
        
        phi = generate_tk95_noise(fs, psd_mapped)
    
        if self.zero_offset:
            phi-=phi[0]

        return t, phi

def phasenoise_maker(frequencies, psd, dt=1e-6, n_samples=1000, **kwargs):
    """
    Functional wrapper for quick noise generation.
    """
    # Now this forwards extrapolation_mode and beta to the class
    sim = NoiseSimulator(frequencies, psd, dt=dt, n_samples=n_samples, **kwargs)
    return sim.generateNoise()
