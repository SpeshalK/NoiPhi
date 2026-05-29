import pytest
import numpy as np
from noiphi.core import PhaseNoiseSimulator

@pytest.fixture
def flat_mock_data():
    """Reusable fixture providing a simple flat power spectrum configuration."""
    frequencies = np.linspace(10, 1000, 100)
    psd = np.ones(100) * 1e-5  # Flat floor profile
    return frequencies, psd

def test_reproducibility_via_seed(flat_mock_data):
    """Ensure passing an integer seed produces identical stochastic trajectories."""
    freqs, psd = flat_mock_data
    
    sim1 = PhaseNoiseSimulator(freqs, psd, dt=1e-4, n_samples=500, seed=42)
    sim2 = PhaseNoiseSimulator(freqs, psd, dt=1e-4, n_samples=500, seed=42)
    
    _, phi1 = sim1.generateNoise()
    _, phi2 = sim2.generateNoise()
    
    np.testing.assert_array_equal(phi1, phi2, err_msg="Seeded trajectories diverged.")

def test_zero_offset_behavior(flat_mock_data):
    """Verify the default constraint forces phi[0] == 0 tracking."""
    freqs, psd = flat_mock_data
    sim = PhaseNoiseSimulator(freqs, psd, zero_offset=True)
    _, phi = sim.generateNoise()
    
    assert phi[0] == 0.0, "Trajectory did not anchor to origin point."

def test_mismatched_array_lengths():
    """Assert initializing with faulty shape geometries raises a descriptive ValueError."""
    freqs = np.array([1, 2, 3])
    psd = np.array([1, 2])  # Length mismatch
    
    with pytest.raises(ValueError, match="arrays must have the same length"):
        PhaseNoiseSimulator(freqs, psd)

def test_double_and_discard_length_preservation(flat_mock_data):
    """
    Verify the underlying double-and-discard slice logic matches the user's 
    requested sample count exactly, despite creating double samples internally.
    """
    freqs, psd = flat_mock_data
    requested_samples = 1234
    
    sim = PhaseNoiseSimulator(freqs, psd, n_samples=requested_samples)
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

    sim=PhaseNoiseSimulator(freqs,psd,dt=1e-4,n_samples=sampleNum,seed=42,zero_offset=False)


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
    



