import pytest
import numpy as np
from noiphi.core import NoiseSimulator

@pytest.fixture
def flat_mock_data():
    """Reusable fixture providing a simple flat power spectrum configuration."""
    frequencies = np.linspace(10, 1000, 100)
    psd = np.ones(100) * 1e-5  # Flat floor profile
    return frequencies, psd

# ── Core Functionality ───────────────────────────────────────────────────────

def test_reproducibility_via_seed(flat_mock_data):
    """Ensure passing an integer seed produces identical stochastic trajectories."""
    freqs, psd = flat_mock_data
    
    sim1 = NoiseSimulator(freqs, psd, dt=1e-4, n_samples=500, seed=42)
    sim2 = NoiseSimulator(freqs, psd, dt=1e-4, n_samples=500, seed=42)
    
    _, phi1 = sim1.generateNoise()
    _, phi2 = sim2.generateNoise()
    
    np.testing.assert_array_equal(phi1, phi2, err_msg="Seeded trajectories diverged.")

def test_zero_offset_behavior(flat_mock_data):
    """Verify the default constraint forces phi[0] == 0 tracking."""
    freqs, psd = flat_mock_data
    sim = NoiseSimulator(freqs, psd, zero_offset=True)
    _, phi = sim.generateNoise()
    
    assert phi[0] == 0.0, "Trajectory did not anchor to origin point."

def test_mismatched_array_lengths():
    """Assert initializing with faulty shape geometries raises a descriptive ValueError."""
    freqs = np.array([1, 2, 3])
    psd = np.array([1, 2])  # Length mismatch
    
    with pytest.raises(ValueError, match="arrays must have the same length"):
        NoiseSimulator(freqs, psd)

# ── Noise ───────────────────────────────────────────────────────

def test_double_and_discard_length_preservation(flat_mock_data):
    """
    Verify the underlying double-and-discard slice logic matches the user's 
    requested sample count exactly, despite creating double samples internally.
    """
    freqs, psd = flat_mock_data
    requested_samples = 1234
    
    sim = NoiseSimulator(freqs, psd, n_samples=requested_samples)
    t, phi = sim.generateNoise()
    
    assert len(t) == requested_samples
    assert len(phi) == requested_samples

def test_parsevals_theorem_variance_convergence(flat_mock_data):
    """
    Physics Check: Verify Parseval's theorem. The ensemble average variance 
    of generated trajectories must converge to the integrated power of the linear PSD.
    """
    freqs, psd = flat_mock_data
    sampleNum=1024
    trajNum=200

    sim=NoiseSimulator(freqs,psd,dt=1e-4,n_samples=sampleNum,seed=42,zero_offset=False)


    # 1. Theoretical variance
    expected_variance=np.sum(sim.psd_linear)*sim.df

    # 2. Generated ensemble variance
    generated_variances=np.zeros(trajNum)
    
    for p in range(trajNum):
        _,phi=sim.generateNoise()
        generated_variances[p]=np.var(phi)

    ensemble_average_variance=np.mean(generated_variances)

    # 4. Assert convergence using np.testing.assert_allclose
    # With 200 trajectories, the statistical fluctuations clear up beautifully,
    # allowing us to pass a tight relative tolerance (rtol) of 5%.
    np.testing.assert_allclose(
        ensemble_average_variance, 
        expected_variance, 
        rtol=0.05,
        err_msg=f"Parseval variance mismatch! Expected {expected_variance}, got {ensemble_average_variance}"
    )

# ── Extrapolation modes ───────────────────────────────────────────────────────
 
@pytest.fixture
def extrapolation_setup():
    """
    Data that intentionally leaves gaps at both ends of the linear grid,
    forcing every extrapolation mode to act on real out-of-range bins.
 
    Grid parameters
    ---------------
    dt=1e-4  →  fs_nyq = 5 000 Hz,  df = 10 Hz  (n_samples=1 000)
 
    Data range: 100–1 000 Hz  (log-spaced, flat PSD = 1e-6 rad²/Hz)
 
    Below-data bins : 10, 20, …, 90 Hz   (9 grid points below 100 Hz)
    Above-data bins : 1 010, …, 4 990 Hz (399 grid points above 1 000 Hz)
    """
    data_f   = np.logspace(2, 3, 50)      # 100 Hz → 1 000 Hz
    data_psd = np.ones(50) * 1e-6         # flat floor PSD
    dt       = 1e-4
    n        = 1000
    return data_f, data_psd, dt, n
 
 
def test_zero_mode_below_data(extrapolation_setup):
    """'zero' mode: every bin below the data range must have PSD = 0."""
    data_f, data_psd, dt, n = extrapolation_setup
    sim = NoiseSimulator(data_f, data_psd, dt=dt, n_samples=n,
                              extrapolation_mode='zero')
 
    below = (sim.f_linear > 0) & (sim.f_linear < data_f[0])
    assert np.any(below), "Fixture produced no below-data grid points."
    assert np.all(sim.psd_linear[below] == 0.0)
 
 
def test_zero_mode_above_data(extrapolation_setup):
    """'zero' mode: every bin above the data range must have PSD = 0."""
    data_f, data_psd, dt, n = extrapolation_setup
    sim = NoiseSimulator(data_f, data_psd, dt=dt, n_samples=n,
                              extrapolation_mode='zero')
 
    above = sim.f_linear > data_f[-1]
    assert np.any(above), "Fixture produced no above-data grid points."
    assert np.all(sim.psd_linear[above] == 0.0)
 
 
def test_floor_mode_holds_low_boundary(extrapolation_setup):
    """'floor' mode: below-data bins must equal the first PSD value."""
    data_f, data_psd, dt, n = extrapolation_setup
    sim = NoiseSimulator(data_f, data_psd, dt=dt, n_samples=n,
                              extrapolation_mode='floor')
 
    below = (sim.f_linear > 0) & (sim.f_linear < data_f[0])
    np.testing.assert_allclose(sim.psd_linear[below], data_psd[0], rtol=1e-9)
 
 
def test_floor_mode_holds_high_boundary(extrapolation_setup):
    """'floor' mode: above-data bins must equal the last PSD value."""
    data_f, data_psd, dt, n = extrapolation_setup
    sim = NoiseSimulator(data_f, data_psd, dt=dt, n_samples=n,
                              extrapolation_mode='floor')
 
    above = sim.f_linear > data_f[-1]
    np.testing.assert_allclose(sim.psd_linear[above], data_psd[-1], rtol=1e-9)
 
 
def test_decay_mode_low_freq_is_floor(extrapolation_setup):
    """'decay' mode: the low-frequency side is still a floor, not a decay."""
    data_f, data_psd, dt, n = extrapolation_setup
    sim = NoiseSimulator(data_f, data_psd, dt=dt, n_samples=n,
                              extrapolation_mode='decay')
 
    below = (sim.f_linear > 0) & (sim.f_linear < data_f[0])
    np.testing.assert_allclose(sim.psd_linear[below], data_psd[0], rtol=1e-9)
 
 
def test_decay_mode_power_law_above_data(extrapolation_setup):
    """
    Physics Check: above the data range the PSD must follow
    S(f) = S(f_max) * (f / f_max)^(-beta).
 
    With beta=2 and flat input PSD = 1e-6, the ratio of any two
    above-data bins must equal (f2/f1)^2 to within 0.01%.
    """
    data_f, data_psd, dt, n = extrapolation_setup
    beta = 2.0
    sim = NoiseSimulator(data_f, data_psd, dt=dt, n_samples=n,
                              extrapolation_mode='decay', beta=beta)
 
    # Grid is 10-Hz spaced; 2 000 and 4 000 Hz are exact grid points
    f_grid = sim.f_linear
    idx_2k = np.argmin(np.abs(f_grid - 2000.0))
    idx_4k = np.argmin(np.abs(f_grid - 4000.0))
 
    psd_2k = sim.psd_linear[idx_2k]
    psd_4k = sim.psd_linear[idx_4k]
 
    # Expected: S(4000) / S(2000) = (4000/2000)^(-2) = 0.25
    expected_ratio = (f_grid[idx_4k] / f_grid[idx_2k]) ** (-beta)
    np.testing.assert_allclose(psd_2k / psd_4k, 1.0 / expected_ratio, rtol=1e-4)
 
 
def test_all_modes_agree_within_data_range(extrapolation_setup):
    """
    Within the data range, all three extrapolation modes must produce
    identical interpolated PSD values — mode only affects out-of-range bins.
    """
    data_f, data_psd, dt, n = extrapolation_setup
 
    sims = {
        mode: NoiseSimulator(data_f, data_psd, dt=dt, n_samples=n,
                                  extrapolation_mode=mode)
        for mode in ('zero', 'floor', 'decay')
    }
 
    in_range = (sims['zero'].f_linear >= data_f[0]) & \
               (sims['zero'].f_linear <= data_f[-1])
 
    ref = sims['zero'].psd_linear[in_range]
    for mode in ('floor', 'decay'):
        np.testing.assert_allclose(
            sims[mode].psd_linear[in_range], ref, rtol=1e-9,
            err_msg=f"'{mode}' diverged from 'zero' inside the data range."
        )

# ── generateNoise overrides ───────────────────────────────────────────────────
 
def test_n_samples_override_changes_output_length(flat_mock_data):
    """Passing n_samples to generateNoise must change output length, not the instance default."""
    freqs, psd = flat_mock_data
    sim = NoiseSimulator(freqs, psd, dt=1e-4, n_samples=500)
 
    t, phi = sim.generateNoise(n_samples=250)
 
    assert len(t)   == 250
    assert len(phi) == 250
 
 
def test_dt_override_changes_time_step(flat_mock_data):
    """Passing dt to generateNoise must use that step size in the returned time axis."""
    freqs, psd = flat_mock_data
    sim = NoiseSimulator(freqs, psd, dt=1e-4, n_samples=500)
 
    override_dt = 5e-5
    t, _ = sim.generateNoise(dt=override_dt)
 
    # Uniform spacing should equal the override dt throughout
    np.testing.assert_allclose(np.diff(t), override_dt, rtol=1e-12)
 
 
def test_combined_override_n_samples_and_dt(flat_mock_data):
    """Both overrides together must each take effect independently."""
    freqs, psd = flat_mock_data
    sim = NoiseSimulator(freqs, psd, dt=1e-4, n_samples=500)
 
    override_n  = 300
    override_dt = 2e-5
    t, phi = sim.generateNoise(n_samples=override_n, dt=override_dt)
 
    assert len(t)   == override_n
    assert len(phi) == override_n
    np.testing.assert_allclose(np.diff(t), override_dt, rtol=1e-12)
 
 
def test_overrides_do_not_mutate_instance_grid(flat_mock_data):
    """
    An override call must not modify f_linear or psd_linear on the instance.
    The docstring explicitly guarantees: 'the default grid stored on the
    instance is not modified'.
    """
    freqs, psd = flat_mock_data
    sim = NoiseSimulator(freqs, psd, dt=1e-4, n_samples=500)
 
    f_before   = sim.f_linear.copy()
    psd_before = sim.psd_linear.copy()
 
    sim.generateNoise(n_samples=999, dt=1e-5)
 
    np.testing.assert_array_equal(sim.f_linear,   f_before)
    np.testing.assert_array_equal(sim.psd_linear, psd_before)
 
 
# ── Time axis ─────────────────────────────────────────────────────────────────
 
def test_time_axis_starts_at_zero(flat_mock_data):
    """Time axis must always begin at t = 0."""
    freqs, psd = flat_mock_data
    sim = NoiseSimulator(freqs, psd, dt=1e-4, n_samples=500)
    t, _ = sim.generateNoise()
    assert t[0] == 0.0
 
 
def test_time_axis_is_uniformly_spaced(flat_mock_data):
    """All time steps must equal dt to floating-point precision."""
    freqs, psd = flat_mock_data
    dt = 1e-4
    sim = NoiseSimulator(freqs, psd, dt=dt, n_samples=500)
    t, _ = sim.generateNoise()
    np.testing.assert_allclose(np.diff(t), dt, rtol=1e-12)
 
 
def test_time_axis_end_value(flat_mock_data):
    """Last time value must equal (n_samples - 1) * dt, not n_samples * dt."""
    freqs, psd = flat_mock_data
    dt = 1e-4
    n  = 500
    sim = NoiseSimulator(freqs, psd, dt=dt, n_samples=n)
    t, _ = sim.generateNoise()
    assert t[-1] == pytest.approx((n - 1) * dt, rel=1e-12)


# ── Input Validation ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("frequencies, psd, message", [
    ([100.0, 10.0, 1000.0],     [1.0, 1.0, 1.0],     "strictly increasing"),   # unsorted
    ([10.0, 10.0, 1000.0],      [1.0, 1.0, 1.0],     "strictly increasing"),   # repeated point
    ([0.0, 10.0, 100.0],        [1.0, 1.0, 1.0],     "strictly positive"),     # DC bin included
    ([-10.0, 10.0, 100.0],      [1.0, 1.0, 1.0],     "strictly positive"),     # negative frequency
    ([10.0, 100.0, 1000.0],     [1.0, 0.0, 1.0],     "strictly positive"),     # zero PSD
    ([10.0, 100.0, 1000.0],     [1.0, -1.0, 1.0],    "strictly positive"),     # negative PSD
    ([10.0, np.nan, 1000.0],    [1.0, 1.0, 1.0],     "NaN or infinite"),
    ([10.0, 100.0, 1000.0],     [1.0, np.inf, 1.0],  "NaN or infinite"),
    ([10.0],                    [1.0],               "at least two points"),
    ([[10.0, 100.0]],           [[1.0, 1.0]],        "one-dimensional"),
])
def test_invalid_spectrum_raises(frequencies, psd, message):
    """Malformed frequency/PSD data must be rejected rather than silently mis-sampled."""
    with pytest.raises(ValueError, match=message):
        NoiseSimulator(frequencies, psd)


def test_non_numeric_spectrum_raises():
    """Arrays that cannot be converted to floats are a type error."""
    with pytest.raises(TypeError, match="numeric"):
        NoiseSimulator(["a", "b", "c"], [1.0, 1.0, 1.0])


def test_unknown_extrapolation_mode_raises(flat_mock_data):
    """A misspelt mode must not fall through to 'floor'."""
    freqs, psd = flat_mock_data
    with pytest.raises(ValueError, match="extrapolation_mode"):
        NoiseSimulator(freqs, psd, extrapolation_mode='flor')


def test_extrapolation_mode_is_case_insensitive(flat_mock_data):
    """Mode names are normalised to lower case, as before."""
    freqs, psd = flat_mock_data
    sim = NoiseSimulator(freqs, psd, extrapolation_mode='DECAY')
    assert sim.extrapolation_mode == 'decay'


@pytest.mark.parametrize("beta", [0.0, -2.0, np.nan, np.inf])
def test_invalid_beta_raises_in_decay_mode(flat_mock_data, beta):
    """A non-positive or non-finite exponent would not describe a decay."""
    freqs, psd = flat_mock_data
    with pytest.raises(ValueError, match="beta"):
        NoiseSimulator(freqs, psd, extrapolation_mode='decay', beta=beta)


def test_beta_is_ignored_outside_decay_mode(flat_mock_data):
    """beta is unused by 'floor' and 'zero', so it is not validated there."""
    freqs, psd = flat_mock_data
    NoiseSimulator(freqs, psd, extrapolation_mode='floor', beta=-2.0)


@pytest.mark.parametrize("dt", [0.0, -1e-6, np.nan, np.inf])
def test_invalid_dt_raises(flat_mock_data, dt):
    """dt must be positive and finite, at construction and as an override."""
    freqs, psd = flat_mock_data
    with pytest.raises(ValueError, match="dt"):
        NoiseSimulator(freqs, psd, dt=dt)

    sim = NoiseSimulator(freqs, psd, dt=1e-4, n_samples=500)
    with pytest.raises(ValueError, match="dt"):
        sim.generateNoise(dt=dt)


@pytest.mark.parametrize("n_samples, error", [
    (0,     ValueError),
    (1,     ValueError),
    (-10,   ValueError),
    (500.0, TypeError),    # floats are rejected, even when integral
    (True,  TypeError),
    ("500", TypeError),
])
def test_invalid_n_samples_raises(flat_mock_data, n_samples, error):
    """n_samples must be an integer >= 2, at construction and as an override."""
    freqs, psd = flat_mock_data
    with pytest.raises(error, match="n_samples"):
        NoiseSimulator(freqs, psd, n_samples=n_samples)

    sim = NoiseSimulator(freqs, psd, dt=1e-4, n_samples=500)
    with pytest.raises(error, match="n_samples"):
        sim.generateNoise(n_samples=n_samples)


def test_non_numeric_dt_raises(flat_mock_data):
    freqs, psd = flat_mock_data
    with pytest.raises(TypeError, match="dt"):
        # Deliberately the wrong type, so the type checker is silenced on this line.
        NoiseSimulator(freqs, psd, dt="1e-6")  # pyright: ignore[reportArgumentType]


def test_valid_input_types_are_accepted():
    """Lists, integer arrays and NumPy integer n_samples are all legitimate input."""
    # NumPy integers are accepted at runtime; the type checker only infers plain int.
    sim = NoiseSimulator([10, 100, 1000], [1, 1, 1], dt=1e-4, n_samples=np.int64(64))  # pyright: ignore[reportArgumentType]
    _, phi = sim.generateNoise()
    assert len(phi) == 64
    assert np.all(np.isfinite(phi))
