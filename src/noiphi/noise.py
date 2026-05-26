import numpy as np
import scipy.fftpack as sc


def generate_tk95_noise(df,psd,rng):
    """
    Low-level implementation of the TK95 noise generation algorithm.

    Draws complex Gaussian amplitudes scaled to the PSD at each frequency bin,
    enforces Hermitian symmetry for real-valued output, and transforms to the
    time domain via an inverse FFT.

    This function performs no input validation. It is intended to be called
    from `core.py`, which handles validation and orchestration.

    Parameters
    ----------
    fs : ndarray
        Uniformly spaced frequency array (Hz) spanning [-f_Nyq, f_Nyq].
    psd : ndarray
        Power Spectral Density evaluated at each element of `fs`.
        Must be the same length as `fs`.
    rng : numpy.random.Generator
        Random number generator instance. Pass np.random.default_rng(seed)
        for reproducible output, or np.random.default_rng() for random.

    Returns
    -------
    phi : ndarray
        Real-valued noise time-series of length `len(fs)`.

    References
    ----------
    Timmer, J. & König, M. (1995). On generating power law noise.
    Astronomy and Astrophysics, 300, 707-710.
    """
    n_half = len(psd)     
    n_full = 2 * n_half   
    half   = n_half - 1   

    real_parts = rng.normal(0.0, 1.0, half) * np.sqrt(psd[1:half+1] / 2)
    imag_parts = rng.normal(0.0, 1.0, half) * np.sqrt(psd[1:half+1] / 2)

    spectrum = np.zeros(n_full, dtype=complex)
    spectrum[1:half+1]  = real_parts + 1j * imag_parts
    spectrum[-(half):]  = np.conjugate(spectrum[1:half+1])[::-1]

    phi = sc.ifft(spectrum) * n_full * np.sqrt(df / 2)
    return phi.real
