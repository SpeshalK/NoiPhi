import pytest
import numpy as np
from noiphi.conversion_tools import (
    frequency_to_phase_psd,
    phase_to_frequency_psd,
    dBc_to_linear,
    stitch_psds,
    dBm_to_Voltage_psd,
    pdh_discriminator_slope,
    voltage_to_phase_psd,
    rin_to_power_psd,
    rin_to_amplitude_psd
)
from noiphi.core import NoiseSimulator

def test_frequency_and_phase_inversion():
    """
    Ensure that translating frequency noise to phase noise and back 
    is perfectly invertible across the spectrum.
    """
    frequencies = np.array([10.0, 100.0, 1000.0, 10000.0])
    mock_S_nu = np.array([1.0, 0.5, 0.1, 0.01])
    
    # S_phi = S_nu / f^2
    S_phi = frequency_to_phase_psd(frequencies, mock_S_nu)
    # S_nu_recovered = S_phi * f^2
    S_nu_recovered = phase_to_frequency_psd(frequencies, S_phi)
    
    np.testing.assert_allclose(S_nu_recovered, mock_S_nu, rtol=1e-12)

def test_dBc_to_linear_scaling():
    """Verify standard logarithmic power ratio scaling to linear units."""
    # -30 dBc should convert to 10**(-3) = 0.001
    assert dBc_to_linear(-30.0) == pytest.approx(0.001, rel=1e-9)
    # 0 dBc should convert to 10**(0) = 1.0
    assert dBc_to_linear(0.0) == pytest.approx(1.0, rel=1e-9)

def test_stitch_psds_boundaries():
    """Verify that stitching masks eliminate overlaps without losing data boundaries."""
    f_low = np.array([1, 2, 3, 4, 5])
    s_low = np.ones(5) * 10
    
    f_high = np.array([4, 5, 6, 7, 8])
    s_high = np.ones(5) * 100
    
    # Transition at 5.0 Hz
    f_final, s_final = stitch_psds(f_low, s_low, f_high, s_high, transition_freq=5.0)
    
    # Low array should cut off strict under 5.0 -> [1, 2, 3, 4]
    # High array should keep values >= 5.0 -> [5, 6, 7, 8]
    expected_freqs = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    expected_psds = np.array([10., 10., 10., 10., 100., 100., 100., 100.]) # Note: s_high elements
    
    np.testing.assert_array_equal(f_final, expected_freqs)
    np.testing.assert_array_equal(s_final, expected_psds)
    assert s_final[4] == 100  # Verify inclusion edge condition at transition


def test_dbm_to_voltage_psd_known_baseline():
    """
    Instrument Check: Verify 0 dBm across 50 Ohms with a 1 Hz RBW 
    evaluates exactly to 0.05 V^2/Hz (1 mW across 50 Ohms).
    """
    # 0 dBm = 1 milliwatt = 1e-3 Watts
    raw_power_dbm = 0.0
    rbw = 1.0
    impedance = 50.0
    
    # Expected: (50 * 10^(0/10)) / (1000 * 1) = 50 / 1000 = 0.05 V^2/Hz
    expected_v_psd = 0.05
    
    actual_v_psd = dBm_to_Voltage_psd(raw_power_dbm, rbw, impedance=impedance)
    
    np.testing.assert_allclose(actual_v_psd, expected_v_psd, rtol=1e-7)

def test_pdh_discriminator_slope_boundaries():
    """
    Physics Check: Verify PDH slope equals k0 at DC, and rolls off 
    by exactly 1/sqrt(2) at half the cavity FWHM linewidth.
    """
    k0 = 2.5  # V/Hz discriminator slope
    cavity_fwhm = 10000.0  # 10 kHz cavity linewidth
    
    # Evaluate at DC (0 Hz) and half-linewidth (5 kHz)
    freqs = np.array([0.0, cavity_fwhm / 2.0])
    
    expected_slopes = np.array([
        k0,               # At 0 Hz: k(f) == k0
        k0 / np.sqrt(2.0) # At f == FWHM/2: k(f) == k0 / sqrt(2)
    ])
    
    actual_slopes = pdh_discriminator_slope(freqs, k0, cavity_fwhm)
    
    np.testing.assert_allclose(actual_slopes, expected_slopes, rtol=1e-7)

def test_voltage_to_phase_psd_transfer_function():
    """
    Physics Check: Validate full Schmid (2021) voltage-to-phase transfer function
    by manually calculating a distinct point.
    """
    f = 5000.0          # 5 kHz Fourier frequency
    s_v = 1e-6          # 1 uV^2/Hz error signal noise power
    k0 = 2.0            # 2 V/Hz DC slope
    cavity_fwhm = 10000.0 # 10 kHz linewidth
    
    # Manual execution step-by-step:
    # 1. k(5kHz) = 2.0 / sqrt(1 + 4*(5000/10000)^2) = 2.0 / sqrt(2)
    # 2. k(5kHz)^2 = 4.0 / 2 = 2.0
    # 3. S_phi = S_V / (f^2 * k^2) = 1e-6 / (25e6 * 2.0) = 1e-6 / 50e6 = 2e-14
    expected_s_phi = 2.0e-14
    
    actual_s_phi = voltage_to_phase_psd(
        np.array([f]), s_v, k0, cavity_fwhm
    )
    
    np.testing.assert_allclose(actual_s_phi, np.array([expected_s_phi]), rtol=1e-7)


# --------------------------- RIN conversions ---------------------------

def _mock_rin():
    """Flicker + white RIN spectrum (1/Hz), ~0.57% integrated RMS."""
    frequencies = np.logspace(1, 5.5, 400)
    rin = 1e-10 * (1 + 1e3 / frequencies)
    return frequencies, rin


def test_rin_to_power_psd_scaling():
    """S_P = RIN * P0^2 (W^2/Hz)."""
    rin = np.array([1e-12, 1e-10])
    np.testing.assert_allclose(rin_to_power_psd(rin, 0.5), rin * 0.25, rtol=1e-12)


def test_rin_to_amplitude_psd_scaling():
    """S_A = RIN * P0 / 4 (W/Hz), from A = sqrt(P)."""
    rin = np.array([1e-12, 1e-10])
    np.testing.assert_allclose(rin_to_amplitude_psd(rin, 0.5), rin * 0.125, rtol=1e-12)


def test_amplitude_and_power_psd_consistency():
    """dP = 2*A0*dA, so S_P must equal 4 * P0 * S_A."""
    _, rin = _mock_rin()
    P0 = 2.0
    np.testing.assert_allclose(rin_to_power_psd(rin, P0),
                               4 * P0 * rin_to_amplitude_psd(rin, P0), rtol=1e-12)


@pytest.mark.parametrize("converter, mean_value, rms_factor", [
    (rin_to_power_psd,     lambda P0: P0,          1.0),   # std(dP)/P0 = sqrt(int RIN)
    (rin_to_amplitude_psd, lambda P0: np.sqrt(P0), 0.5),   # std(dA)/A0 = sqrt(int RIN)/2
])
def test_rin_trajectory_fractional_rms(converter, mean_value, rms_factor):
    """
    End-to-end check against a quantity that does not depend on the
    converter: the fractional RMS of a generated trajectory must match
    the integrated RIN. A Welch round-trip cannot catch a units error
    here, because the simulator reproduces whatever PSD it is given.
    """
    frequencies, rin = _mock_rin()
    P0 = 2.0
    integrated_rin = np.sum(0.5 * (rin[1:] + rin[:-1]) * np.diff(frequencies))
    expected = rms_factor * np.sqrt(integrated_rin)

    sim = NoiseSimulator(frequencies, converter(rin, P0), dt=1e-6, n_samples=2**18,
                         extrapolation_mode='zero', zero_offset=False, seed=1)
    _, x = sim.generateNoise()

    assert np.std(x) / mean_value(P0) == pytest.approx(expected, rel=0.03)
