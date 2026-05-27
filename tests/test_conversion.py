import pytest
import numpy as np
from noiphi.conversion_tools import (
    frequency_to_phase_psd,
    phase_to_frequency_psd,
    dBc_to_phase_psd,
    stitch_psds
)

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

def test_dBc_to_phase_psd_scaling():
    """Verify standard logarithmic power ratio scaling to linear units."""
    # -30 dBc should convert to 10**(-3) = 0.001
    assert dBc_to_phase_psd(-30.0) == pytest.approx(0.001, rel=1e-9)
    # 0 dBc should convert to 10**(0) = 1.0
    assert dBc_to_phase_psd(0.0) == pytest.approx(1.0, rel=1e-9)

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
    expected_psds = np.array([10, 10, 10, 10, 100, 1000, 100, 100]) # Note: s_high elements
    
    np.testing.assert_array_equal(f_final, expected_freqs)
    assert s_final[4] == 100  # Verify inclusion edge condition at transition
