import numpy as np
import scipy.fftpack as sc


def generate_tk95_noise(fs, psd):
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

    Returns
    -------
    phi : ndarray
        Real-valued noise time-series of length `len(fs)`.

    References
    ----------
    Timmer, J. & König, M. (1995). On generating power law noise.
    Astronomy and Astrophysics, 300, 707-710.
    """
    df = fs[-1] - fs[-2]

    spectrum = np.zeros(len(fs), dtype=complex)

    # Sampling the spectrum with Gaussian random variables
    for i in range(1, int(len(fs) / 2 - 1)):
        # Generate complex Gaussian noise scaled by the PSD
        spectrum[i] = np.random.normal(0.0) * np.sqrt(psd[i] / 2) + \
                      1j * np.random.normal(0.0) * np.sqrt(psd[i] / 2)
        spectrum[-i] = np.conjugate(spectrum[i])

    phi = np.fft.fftshift(sc.ifft(spectrum)) * len(fs) * np.sqrt(2 * df).real

    return phi.real
