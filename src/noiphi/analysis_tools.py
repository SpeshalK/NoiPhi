import numpy as np

def autocorrWK(phi):
    """
    Computes autocorrelation using the Wiener-Khinchin theorem (FFT-based).

    Parameters
    ----------
    phi : array_like
        The input time-domain signal (e.g., phase noise trajectory in radians).
        If the signal length is N, the FFT is performed on a zero-padded 
        length of 2N to avoid cyclic correlation artifacts.

    Returns
    -------
    res : ndarray
        The normalized autocorrelation function R(tau) for lags 0 to N-1.
        The output is normalized such that R(0) = 1.0, representing the 
        coherence of the signal relative to its variance.

    Notes
    -----
    1. Performance: This function operates in O(N log N) time, making it 
       exponentially faster than O(N^2) loop-based methods for large 
       datasets (e.g., N > 100,000).
    2. Detrending: The function automatically removes the mean (DC component) 
       to ensure the correlation reflects the stochastic fluctuations around 
        the steady state.
    3. Normalization: 
       R(tau) = Real(IFFT(|FFT(phi)|^2)) / Var(phi)
    """

    phi_centered = phi - np.mean(phi)
    
    # Get power spectrum
    f_phiPow = np.fft.fft(phi_centered, n=2*len(phi))
    psd = np.real(f_phiPow * np.conj(f_phiPow))
    
    # Inverse FFT to get autocorrelation
    r = np.fft.ifft(psd)
    
    # Slice and normalize
    result = np.real(r[:len(phi)])
    return result / result[0]


def AllanDev(phi,dt,taus=None):
    """
    Computes the Allan Deviation (ADEV) from a phase noise trajectory.
    
    ADEV is the square root of the Allan Variance, used to characterize the 
    stability of frequency sources in the time domain.

    Parameters
    ----------
    phi : array_like
        Phase noise trajectory in radians.
    dt : float
        Time step between samples in seconds.
    taus : array_like, optional
        Observation times (integration periods) to compute. If None, 
        uses an octave-spaced grid (1, 2, 4, 8... samples)

    Returns
    -------
    taus : ndarray
        The observation times (seconds).
    adev : ndarray
        The Allan Deviation sigma_y(tau).
    """

    # 1. Convert phase (rad) to fractional frequency fluctuations (y)
    # y = [phi(t + dt) - phi(t)] / (2 * pi * f_center * dt)
    # Note: Using normalized phase rate (rad/s) here for general tool
    y = np.diff(phi) / (2 * np.pi * dt)
    N = len(y)

    if taus is None:
        # Default to octave spacing up to ~1/4 of the total signal length
        max_m = int(np.floor(np.log2(N / 4)))
        ms = 2**np.arange(max_m)
        taus = ms * dt
    else:
        ms = (np.asarray(taus) / dt).astype(int)
        ms = ms[ms > 0]

    adev = []
    actual_taus = []

    for m in ms:
        # Overlapping Allan Variance formula
        # sigma^2(tau) = 1 / (2 * (N - 2m + 1) * m^2) * sum(...)
        # For speed, we use block averaging of y:
        # Calculate average frequency over blocks of size m
        y_bar = np.convolve(y, np.ones(m)/m, mode='valid')
        
        # Difference between blocks separated by m
        diff = y_bar[m:] - y_bar[:-m]
        
        # Allan Variance is half the mean square of these differences
        avar = 0.5 * np.mean(diff**2)
        adev.append(np.sqrt(avar))
        actual_taus.append(m * dt)

    return np.array(actual_taus), np.array(adev)


def integrated_phase_noise(frequencies, psd):
    """
    Computes the Cumulative Integrated Phase Noise (IPN) in radians.
    
    The IPN represents the total RMS phase jitter accumulated within a 
    specific bandwidth. It is calculated by integrating the PSD from 
    the highest frequency down to each point on the frequency axis.

    Parameters
    ----------
    frequencies : array_like
        Frequency axis in Hz (positive values only)
    psd : array_like
        Power Spectral Density in rad^2/Hz

    Returns
    -------
    ipn_rms : ndarray
        The RMS phase jitter (radians) integrated from f_max down to f.
    """
    f = np.asarray(frequencies)
    s = np.asarray(psd)
    
    df = np.diff(f)
    
    # Trapezoidal integration: (S1 + S2)/2 * df
    segment_weights = (s[:-1] + s[1:]) / 2.0
    area_segments = segment_weights * df
    
    # Integrate from the highest frequency downwards (reverse cumulative sum).
    integrated_variance = np.zeros_like(f)
    integrated_variance[:-1] = np.cumsum(area_segments[::-1])[::-1]
    
    # Return RMS value (sqrt of variance)
    return np.sqrt(integrated_variance)
