# NoiPhi

**Stochastic noise simulation for quantum and photonic systems.**

NoiPhi is a Python package for generating physically realistic noise trajectories from experimental power spectral densities (PSDs), based on the Timmer-König (TK95) algorithm. It was originally developed to simulate laser phase noise in quantum many-body systems, and is designed to grow into a general-purpose noise simulation toolkit.

---

## Features

- Gaussian noise generation from arbitrary experimental PSDs using the TK95 algorithm
- Laser phase noise simulation for quantum many-body and photonic systems
- Clean, composable API suitable for integration into larger simulation pipelines
- Designed for extensibility — amplitude noise and generalised noise sources planned

---

## Background

The TK95 algorithm (Timmer & König, 1995) generates stochastic time series whose statistical properties exactly match a given PSD. This makes it particularly well-suited to simulating coloured noise — such as laser phase noise — where the spectral shape is known from experiment but the temporal realisation is random.

NoiPhi applies this approach to quantum simulation contexts, where accurate noise modelling is critical for predicting decoherence, heating rates, and other environmentally driven dynamics.

---

## Installation

```bash
pip install noiphi
```

Or from source:

```bash
git clone https://github.com/your-username/noiphi.git
cd noiphi
pip install -e .
```

---

## Quick Start

```python
import numpy as np
from noiphi import PhaseNoiseSimulator

# Define a frequency axis and a PSD (e.g. from experiment)
frequencies = np.logspace(1, 6, 1000)        # 10 Hz to 1 MHz
psd = 1e-10 / frequencies                    # 1/f noise example

# Generate a phase noise trajectory
sim = PhaseNoiseSimulator(frequencies, psd, dt=1e-6, n_samples=10_000)
t, phi = sim.generate()
```

---

## How It Works

1. **Input a PSD** — provide a measured or analytical power spectral density over a frequency grid.
2. **Gaussian sampling** — the TK95 algorithm draws complex Gaussian amplitudes scaled to the PSD at each frequency bin.
3. **Inverse FFT** — the frequency-domain representation is transformed to produce a stationary time series with the correct spectral statistics.
4. **Output** — a time-domain noise trajectory ready for use in simulation.

---

## Roadmap

- [x] Laser phase noise via TK95 Gaussian sampling
- [ ] Amplitude noise simulation
- [ ] Support for non-Gaussian sampling methods
- [ ] Generalised noise sources (magnetic field fluctuations, intensity noise, etc.)
- [ ] Utilities for fitting PSDs to experimental data

---

## Citation

If you use NoiPhi in academic work, please cite the original TK95 algorithm:

> Timmer, J. & König, M. (1995). *On generating power law noise.* Astronomy and Astrophysics, 300, 707–710.

A dedicated NoiPhi citation entry will be added upon first release.

---

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
