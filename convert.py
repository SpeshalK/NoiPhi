import numpy as np

def pdh_discriminator_slope(frequencies, k0, delta_f_fwhm):
    """
    Computes the frequency-dependent PDH discriminator slope k(f).

    At Fourier frequencies above the cavity linewidth, the field stored in
    the cavity can no longer follow the incident field fluctuations. The
    discriminator slope therefore rolls off with frequency according to
    equation (2) of Schmid et al. (2021):

        k(f) = k0 / sqrt(1 + 4 * (f / delta_f_fwhm)^2)

    Parameters
    ----------
    frequencies : ndarray
        Fourier frequency array (Hz). Must be positive.
    k0 : float
        DC discriminator slope (V/Hz), measured at low Fourier frequencies.
    delta_f_fwhm : float
        FWHM linewidth of the reference cavity (Hz).

    Returns
    -------
    k : ndarray
        Frequency-dependent discriminator slope (V/Hz), same shape as
        `frequencies`.

    References
    ----------
    Schmid, F. et al. (2021). Simple phase noise measurement scheme for
    cavity-stabilized laser systems. Optics Letters.
    https://doi.org/10.1364/OL.44.002709
    """
    frequencies = np.asarray(frequencies, dtype=float)
    return k0 / np.sqrt(1 + 4 * (frequencies / delta_f_fwhm) ** 2)


def frequency_to_phase_psd(frequencies, S_nu):
    """
    Converts a frequency noise PSD to a phase noise PSD.

    Uses the relation:

        S_phi(f) = S_nu(f) / f^2

    Parameters
    ----------
    frequencies : ndarray
        Fourier frequency array (Hz). Must be positive and non-zero.
    S_nu : ndarray
        Single-sided frequency noise PSD (Hz^2/Hz), same length as
        `frequencies`.

    Returns
    -------
    S_phi : ndarray
        Single-sided phase noise PSD (rad^2/Hz), same shape as `frequencies`.

    Notes
    -----
    Division by zero will occur if `frequencies` contains zero. The DC bin
    should be excluded before calling this function.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    S_nu = np.asarray(S_nu, dtype=float)
    return S_nu / frequencies ** 2


def phase_to_frequency_psd(frequencies, S_phi):
    """
    Converts a phase noise PSD to a frequency noise PSD.

    Uses the relation:

        S_nu(f) = S_phi(f) * f^2

    Parameters
    ----------
    frequencies : ndarray
        Fourier frequency array (Hz).
    S_phi : ndarray
        Single-sided phase noise PSD (rad^2/Hz), same length as `frequencies`.

    Returns
    -------
    S_nu : ndarray
        Single-sided frequency noise PSD (Hz^2/Hz), same shape as
        `frequencies`.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    S_phi = np.asarray(S_phi, dtype=float)
    return S_phi * frequencies ** 2


def voltage_to_phase_psd(frequencies, S_V, k0, delta_f_fwhm):
    """
    Converts a voltage PSD from a PDH error signal to a phase noise PSD.

    Uses the full transfer function from equation (3) of Schmid et al. (2021):

        S_phi(f) = S_V(f) / (f^2 * k^2(f))

    where k(f) is the frequency-dependent PDH discriminator slope computed
    by `pdh_discriminator_slope`.

    Parameters
    ----------
    frequencies : ndarray
        Fourier frequency array (Hz). Must be positive and non-zero.
    S_V : ndarray
        Single-sided voltage PSD (V^2/Hz) of the PDH error signal, same
        length as `frequencies`.
    k0 : float
        DC discriminator slope (V/Hz).
    delta_f_fwhm : float
        FWHM linewidth of the reference cavity (Hz).

    Returns
    -------
    S_phi : ndarray
        Single-sided phase noise PSD (rad^2/Hz), same shape as `frequencies`.

    Notes
    -----
    Division by zero will occur if `frequencies` contains zero. The DC bin
    should be excluded before calling this function.

    References
    ----------
    Schmid, F. et al. (2021). Simple phase noise measurement scheme for
    cavity-stabilized laser systems. Optics Letters.
    https://doi.org/10.1364/OL.44.002709
    """
    frequencies = np.asarray(frequencies, dtype=float)
    S_V = np.asarray(S_V, dtype=float)
    k = pdh_discriminator_slope(frequencies, k0, delta_f_fwhm)
    return S_V / (frequencies ** 2 * k ** 2)


def dBc_to_linear(L_dBc):
    """
    Converts a single sideband phase noise level from dBc/Hz to linear units.

    The single sideband phase noise L(f) in dBc/Hz is defined as the noise
    power spectral density relative to the carrier power. The conversion to
    linear (rad^2/Hz) is:

        L_linear = 10^(L_dBc / 10)

    Parameters
    ----------
    L_dBc : array-like or float
        Single sideband phase noise (dBc/Hz).

    Returns
    -------
    L_linear : ndarray or float
        Phase noise in linear units (rad^2/Hz).

    Notes
    -----
    For small phase noise (L_dBc << 0 dBc/Hz), L_linear ≈ S_phi(f) / 2.
    """
    return 10 ** (np.asarray(L_dBc, dtype=float) / 10)
