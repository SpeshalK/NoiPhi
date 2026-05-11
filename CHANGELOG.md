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

## [Unreleased]

### Planned
- Amplitude noise simulation
- Utilities for fitting PSDs to experimental data
- Support for non-Gaussian sampling methods
- Generalised noise sources (magnetic field fluctuations, intensity noise, etc.)
