# Changelog

All notable changes to NoiPhi will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-05-09

### Added

**Core**
- `PhaseNoiseSimulator` class implementing the TK95 (Timmer & König, 1995) algorithm for generating stochastic time-domain noise trajectories from experimental PSDs
- Interpolated frequency grid (`f_linear_full`, `psd_linear_full`) pre-computed at construction time and available as instance attributes without requiring a `generateNoise()` call
- `generateNoise()` method with optional `n_samples` and `dt` overrides for multi-scale sampling; default grid preserved on instance after override calls
- `seed` parameter for reproducible noise generation via `numpy.random.default_rng`
- Vectorised TK95 Gaussian sampling replacing per-bin loop for improved performance
- Three extrapolation modes for handling out-of-band frequencies:
  - `floor` — constant extrapolation at boundary PSD values (default)
  - `zero` — no power outside measured range
  - `decay` — Kohlrausch power-law roll-off with configurable exponent `beta` (default `β=2.0`)
- `zero_offset` option to anchor phase trajectory to `phi[0] = 0`
- `phasenoise_maker()` functional wrapper for quick single-call noise generation

**Conversion Tools**
- `frequency_to_phase_psd()` — converts frequency noise PSD (Hz²/Hz) to phase noise PSD (rad²/Hz)
- `voltage_to_phase_psd()` — converts voltage PSD (V²/Hz) to phase noise PSD via PDH discriminator slope
- `dBm_to_Voltage_psd()` — converts logarithmic dBm spectrum analyser data to linear voltage PSD (V²/Hz), accounting for resolution bandwidth
- `stitch_psds()` — merges low and high-frequency PSD spans with configurable hard-cut transition frequency

**Analysis Tools**
- `autocorrWK()` — FFT-based autocorrelation via the Wiener-Khinchin theorem for coherence time characterisation
- `AllanDev()` — overlapping Allan Deviation for time-domain frequency stability analysis
- `integrated_phase_noise()` — cumulative integrated phase noise (IPN), integrating from `f_max` downward to give RMS phase jitter as a function of frequency

**Demonstrations** (`usage/`)
- `FreqPSDtoPhaseNoise.py` — full workflow from experimental frequency noise CSV to phase noise trajectory, with PSD round-trip verification via Welch periodogram
- `VoltagePSDtoPhaseNoise.py` — voltage PSD pipeline including dBm conversion, PDH discriminator modelling, and multi-span stitching (918nm diode laser example)
- `Extrapolation.py` — side-by-side comparison of all three extrapolation modes in frequency and time domains
- `n_sampling_override.py` — demonstrates multi-scale sampling from a single simulator instance
- `noise_analysis.py` — autocorrelation, Allan deviation, and IPN analysis workflow for Rydberg experiment characterisation

**Demonstrations** (`physics/`)
- `Rabi_Oscill_Phase_Noisy.py` — single-atom Rabi oscillations under laser phase noise, ensemble mean and ±1σ band vs noiseless reference
- `many_body_Ising_grEvo.py` — N-atom Rydberg chain ground state evolution under global phase noise, ensemble average with deviation panel
- `Ramsey_fringe_demo.py` — Ramsey fringe contrast decay vs free precession time, with empirical `σ_φ(T)` analytic overlay connecting directly to IPN analysis tools

---

## [0.1.1] - 2026-05-20
 
### Bug Fix
 
**Noise**
- `noise.py` — Fixing scaling of output noise signal φ replacing scalefactor np.sqrt(df*2) --> np.sqrt(df/2)
             — Removing superfluous ifftshift
            
- `core.py`  — Implementing double and discard (frequency space) functionality to remove Fourier periodicity.
             — Bug fixing double and discard functionality to ensure user facing frequency grid consistancy.
---

## [0.1.2] - 2026-05-30
 
### Added
 
**CI**
- `ci.yml` — GitHub Actions workflow that automatically runs the full pytest suite on every push and pull request to `main`, using `actions/checkout@v5` and `actions/setup-python@v6` (Node.js 24 compatible)
 
**Tests**
- `tests/test_analysis.py` — full pytest suite for `analysis_tools`, covering `autocorrWK`, `AllanDev`, and `integrated_phase_noise` with normalisation, output shape, physics, and edge-case checks
- `tests/test_core.py` — extended with coverage for all three extrapolation modes (`zero`, `floor`, `decay`), `generateNoise()` override parameters (`n_samples`, `dt`, combined), override immutability of the instance grid, and time axis correctness (origin, uniform spacing, end value)
### Fixed
 
**CI / packaging**
- Corrected `pyproject.toml` package discovery from `where = ["."]` to `where = ["src"]` to match the `src/` layout, resolving `ModuleNotFoundError: No module named 'noiphi'` in clean CI environments
- Added `[tool.pytest.ini_options] pythonpath = ["src"]` so pytest resolves the package without requiring an editable install
- Updated GitHub Actions to Node.js 24 compatible versions: `actions/checkout@v4` → `v5`, `actions/setup-python@v5` → `v6`
---

## [0.2.0] - YYYY-MM-DD

### Added

**Amplitude noise**
- `NoiseSimulator` is now agnostic to the noise type: any one-sided PSD (phase, frequency, voltage, power, amplitude) produces a trajectory in the square root of the PSD units
- `NoiseSimulator` now validates its inputs and raises `ValueError`/`TypeError` for malformed spectra, unknown extrapolation modes, and invalid `dt`, `n_samples` or `beta`
- `rin_to_power_psd()` — converts linear RIN (1/Hz) to a power noise PSD (W²/Hz) via `RIN * P0²`, for power-fluctuation trajectories δP(t)
- `rin_to_amplitude_psd()` — converts linear RIN (1/Hz) to a field-amplitude noise PSD (W/Hz) via `RIN * P0 / 4`, with A = √P
- `integrated_rin()` — cumulative integrated RIN, giving RMS fractional intensity noise as a function of bandwidth

**Demonstrations** (`usage/`)
- `RINtoAmplitudeNoise.py` — full workflow from an experimental RIN spectrum (dBc/Hz) to a power-fluctuation trajectory, with PSD round-trip verification via Welch periodogram
- `amplitude_noise_analysis.py` — autocorrelation and integrated RIN analysis of laser intensity noise
- `demos/data/Amplitude/Nd_YAG_1064nm_RIN_dBc.csv` — synthetic 1064nm Nd:YAG RIN spectrum

**Tests**
- `tests/test_demos.py` — auto-discovers every demo script and smoke-tests it as a subprocess in a sandboxed copy of the demos tree
- `pytest.ini` — `demo` marker; demo tests are excluded by default and run with `pytest -m demo`
- `tests/test_conversion.py` — RIN conversion tests, including an end-to-end check of trajectory fractional RMS against the integrated RIN

**Project**
- `CONTRIBUTING.md`, `SECURITY.md`, and GitHub issue templates

### Changed

**Breaking**
- `PhaseNoiseSimulator` renamed to `NoiseSimulator`
- `phasenoise_maker()` renamed to `noise_maker()`
- `dBc_to_phase_psd()` renamed to `dBc_to_linear()`

**Demonstrations**
- Demo data split into `demos/data/Phase/` and `demos/data/Amplitude/`; all demo scripts updated to the new paths
- `noise_analysis.py` renamed to `phase_noise_analysis.py`

---


## [Unreleased]

### Planned
- Utilities for fitting PSDs to experimental data
- Support for non-Gaussian sampling methods
- Generalised noise sources (magnetic field fluctuations, intensity noise, etc.)
