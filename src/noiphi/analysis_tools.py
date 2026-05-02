import numpy as np

def autocorrWK(sig):
    """
    Computes autocorrelation using the Wiener-Khinchin theorem (FFT-based).

    Parameters
    ----------
   sig : array_like
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
       R(tau) = Real(IFFT(|FFT(sig)|^2)) / Var(sig)
    """

    sig_centered = sig - np.mean(sig)
    
    # Get power spectrum
    f_sigPow = np.fft.fft(sig_centered, n=2*len(sig))
    psd = np.real(f_sigPow * np.conj(f_sigPow))
    
    # Inverse FFT to get autocorrelation
    r = np.fft.ifft(psd)
    
    # Slice and normalize
    res = np.real(r[:len(sig)])
    return res / res[0]
