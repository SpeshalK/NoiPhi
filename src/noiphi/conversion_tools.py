import numpy as np

#--------------------- PHASE NOISE ---------------------------------

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


#--------------------- AMPLITUDE NOISE ---------------------------------

def rin_to_power_psd(rin, P0):
    """
    Converts a linear RIN spectrum to an absolute power noise PSD.
 
    Uses the relation:
 
        S_P(f) = RIN(f) * P0^2
 
    Parameters
    ----------
    rin : array-like
        Linear relative intensity noise (1/Hz).
    P0 : float
        Mean optical power (W).
 
    Returns
    -------
    S_P : ndarray
        Single-sided power noise PSD (W^2/Hz).
    """
    rin = np.asarray(rin, dtype=float)
    return rin * P0 ** 2


def rin_to_amplitude_psd(rin, P0):
    """
    Converts a linear RIN spectrum to an absolute amplitude noise PSD.
 
    Uses the relation:
 
        S_A(f) = sqrt(RIN(f)) * P0
 
    Parameters
    ----------
    rin : array-like
        Linear relative intensity noise (1/Hz).
    P0 : float
        Mean optical power (W).
 
    Returns
    -------
    S_A : ndarray
        Single-sided amplitude noise PSD (W/sqrt(Hz)).
    """
    rin = np.asarray(rin, dtype=float)
    return np.sqrt(rin) * P0


#--------------------- AMPLITUDE + PHASE NOISE ---------------------------------

def dBc_to_linear(L_dBc):
    """
    Converts a noise level from dBc/Hz to linear units.
 
    Uses the standard dB-to-linear relation:
 
        L_linear = 10^(L_dBc / 10)
 
    Parameters
    ----------
    L_dBc : array-like or float
        Noise level in dBc/Hz.
 
    Returns
    -------
    L_linear : ndarray or float
        Linear noise level. Units depend on the measured quantity:
        rad^2/Hz for single-sideband phase noise, 1/Hz for RIN.
 
    Notes
    -----
    For small phase noise (L_dBc << 0 dBc/Hz), L_linear ≈ S_phi(f) / 2.
    """
    return 10 ** (np.asarray(L_dBc, dtype=float) / 10)

#--------------------- FUNCTIONALITY ---------------------------------

def dBm_to_Voltage_psd(psd_dbm, rbw, impedance=50.0):
    """
    Scales raw Spectrum Analyzer power levels to a linear Voltage PSD (V^2/Hz).

    This function implements the standard conversion used to normalize instrument
    traces based on Resolution Bandwidth (RBW) and system impedance.

    Parameters
    ----------
    psd_dbm : array_like
        The raw power spectral density measured by the instrument (dBm or dBm/Hz).
    rbw : float
        The Resolution Bandwidth (Hz) setting used during the measurement.
    impedance : float, optional
        The system impedance in Ohms. Default is 50.0.

    Returns
    -------
    s_v : ndarray
        The linear Voltage Power Spectral Density in V^2/Hz.

    Notes
    -----
    The formula used is:
        S_V = (Impedance * 10^(P_dBm / 10)) / (1000 * RBW)
    """
    psd_dbm = np.asarray(psd_dbm, dtype=float)
    
    # Convert dBm to linear scale and normalize by RBW and Impedance
    # The factor of 1000 accounts for the mW to W conversion (dBm)
    p_linear = 10**(psd_dbm / 10.0)
    s_v = (impedance * p_linear) / (1000.0 * rbw)
    
    return s_v



def stitch_psds(f_low, s_low, f_high, s_high, transition_freq):
    """
    Stitches two PSD segments at a specific transition frequency to 
    eliminate overlap and minimize discontinuities.
    
    Parameters
    ----------
    f_low, s_low : ndarray
        Frequency and PSD values for the low-frequency segment.
    f_high, s_high : ndarray
        Frequency and PSD values for the high-frequency segment.
    transition_freq : float
        The frequency (Hz) where the script should switch from low to high.
    """
    # Filter low segment to stay below transition
    mask_low = f_low < transition_freq
    f_low_cut = f_low[mask_low]
    s_low_cut = s_low[mask_low]
    
    # Filter high segment to stay above transition
    mask_high = f_high >= transition_freq
    f_high_cut = f_high[mask_high]
    s_high_cut = s_high[mask_high]
    
    f_final = np.concatenate([f_low_cut, f_high_cut])
    s_final = np.concatenate([s_low_cut, s_high_cut])
    
    return f_final, s_final
