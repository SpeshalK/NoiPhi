import pytest
import numpy as np
from noiphi.analysis_tools import autocorrWK, AllanDev, integrated_phase_noise


# ── autocorrWK ────────────────────────────────────────────────────────────────

class TestAutocorrWK:
    def test_normalization_at_zero_lag(self):
        """R(0) must equal 1.0 by definition of a normalized autocorrelation."""
        rng = np.random.default_rng(0)
        phi = rng.standard_normal(1000)
        result = autocorrWK(phi)
        assert result[0] == pytest.approx(1.0, rel=1e-9)

    def test_output_length_matches_input(self):
        """Output array length must equal the input signal length."""
        rng = np.random.default_rng(1)
        phi = rng.standard_normal(512)
        result = autocorrWK(phi)
        assert len(result) == len(phi)

    def test_pure_sinusoid_autocorrelation(self):
        """
        Physics Check: The normalized autocorrelation of A*sin(2π*f0*t) is
        cos(2π*f0*τ). We verify R(0) = 1, R(T/4) ≈ 0, and R(T/2) ≈ -1
        for one full period of the signal's frequency.
        """
        N = 4096
        dt = 1e-3
        f0 = 10.0  # Hz
        t = np.arange(N) * dt
        phi = np.sin(2 * np.pi * f0 * t)

        result = autocorrWK(phi)

        period_samples = int(round(1.0 / (f0 * dt)))   # samples per period
        quarter = period_samples // 4

        assert result[0]             == pytest.approx(1.0,  abs=1e-6)
        assert result[quarter]       == pytest.approx(0.0,  abs=0.05)
        assert result[period_samples // 2] == pytest.approx(-1.0, abs=0.05)

    def test_mean_removal_dc_offset(self):
        """
        A signal with a large DC offset must give the same autocorrelation
        as the zero-mean version, confirming mean subtraction is applied.
        """
        rng = np.random.default_rng(2)
        noise = rng.standard_normal(500)
        phi_dc = noise + 1000.0     # large DC component

        result_centered = autocorrWK(noise)
        result_dc       = autocorrWK(phi_dc)

        np.testing.assert_allclose(result_dc, result_centered, rtol=1e-9)


# ── AllanDev ──────────────────────────────────────────────────────────────────

class TestAllanDev:
    def test_output_arrays_same_length(self):
        """The returned taus and adev arrays must have matching lengths."""
        rng = np.random.default_rng(3)
        phi = np.cumsum(rng.standard_normal(1024))
        taus, adev = AllanDev(phi, dt=1e-3)
        assert len(taus) == len(adev)

    def test_all_positive_values(self):
        """Allan deviation must be strictly positive for any non-trivial signal."""
        rng = np.random.default_rng(4)
        phi = np.cumsum(rng.standard_normal(512))
        _, adev = AllanDev(phi, dt=1e-3)
        assert np.all(adev > 0)

    def test_custom_taus_respected(self):
        """
        When taus are supplied explicitly, the returned tau axis must
        reflect those values (rounded to the nearest sample period).
        """
        rng = np.random.default_rng(5)
        phi = np.cumsum(rng.standard_normal(2048))
        dt = 1e-3
        # All three are exact multiples of dt so no rounding error
        custom_taus = np.array([0.010, 0.020, 0.040])
        taus_out, adev = AllanDev(phi, dt=dt, taus=custom_taus)
        np.testing.assert_allclose(taus_out, custom_taus, rtol=1e-9)
        assert len(adev) == len(custom_taus)

    def test_white_frequency_noise_scaling(self):
        """
        Physics Check: For white frequency noise the overlapping Allan deviation
        scales as σ_y(τ) ∝ 1/√τ. Doubling the averaging time τ therefore
        halves the variance, giving adev(τ) / adev(2τ) ≈ √2.
        """
        rng = np.random.default_rng(42)
        N  = 32768
        dt = 1e-4
        # A random walk in phase produces white frequency noise increments
        phi = np.cumsum(rng.standard_normal(N)) * dt

        # Use two consecutive octave averaging times
        tau1, tau2 = 64 * dt, 128 * dt
        taus, adev = AllanDev(phi, dt=dt, taus=np.array([tau1, tau2]))

        ratio = adev[0] / adev[1]
        # Expected ratio is √2 ≈ 1.414; 15% tolerance covers finite-sample noise
        assert ratio == pytest.approx(np.sqrt(2), rel=0.15)


# ── integrated_phase_noise ────────────────────────────────────────────────────

class TestIntegratedPhaseNoise:
    def test_output_shape_matches_input(self):
        """IPN output array must have the same shape as the input arrays."""
        f = np.linspace(1, 1000, 200)
        s = np.ones(200) * 1e-6
        ipn = integrated_phase_noise(f, s)
        assert ipn.shape == f.shape

    def test_monotonically_non_increasing(self):
        """
        IPN integrates from f_max downward, so every element must be
        greater than or equal to the next — more bandwidth, more noise.
        """
        f = np.linspace(1, 1000, 500)
        s = 1e-6 / f     # 1/f noise — always positive
        ipn = integrated_phase_noise(f, s)
        assert np.all(np.diff(ipn) <= 0)

    def test_flat_psd_total_power(self):
        """
        Physics Check: For a flat PSD of level S0 over bandwidth B,
        the total RMS phase noise equals √(S0 · B). The first element
        of the IPN array (integration from f_max back to f_min) must
        match this analytic result to within 1%.
        """
        f  = np.linspace(10.0, 1010.0, 1000)
        S0 = 1e-6                                  # rad²/Hz
        s  = np.ones(1000) * S0

        ipn = integrated_phase_noise(f, s)

        bandwidth     = f[-1] - f[0]               # ≈ 1000 Hz
        expected_rms  = np.sqrt(S0 * bandwidth)    # √(1e-3) ≈ 0.03162 rad

        assert ipn[0] == pytest.approx(expected_rms, rel=0.01)

    def test_zero_psd_gives_zero_ipn(self):
        """A zero PSD contributes no noise power — IPN must be zero everywhere."""
        f = np.linspace(1, 100, 50)
        s = np.zeros(50)
        ipn = integrated_phase_noise(f, s)
        np.testing.assert_array_equal(ipn, 0.0)

    def test_last_element_is_zero(self):
        """
        The highest-frequency bin contributes no integrated noise above itself,
        so ipn[-1] must always be zero regardless of the PSD shape.
        """
        f = np.linspace(100, 10000, 300)
        s = 1e-8 / f ** 2
        ipn = integrated_phase_noise(f, s)
        assert ipn[-1] == 0.0
