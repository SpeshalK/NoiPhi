# NoiPhi

**Stochastic noise simulation kit for quantum and photonic systems.**

NoiPhi is a Python package for generating physically realistic noise trajectories from experimental power spectral densities (PSDs), based on the Timmer-König (TK95) algorithm. It was originally developed to simulate laser phase noise in quantum many-body systems, and is designed to grow into a general-purpose noise simulation toolkit.


![Many-Body Ising Dynamics and Ramsey Contrast Decay](/home/slopo/Work/Projects/NoiPhi/demos/physics/plotsManyBodyIsing+Ramsey.jpg)
---
## Why NoiPhi?

While many libraries focus on theoretical power-law noise (e.g., $1/f^α$), NoiPhi is built for the experimentalist who needs to transform a specific PSD from a spectrum analyzer into a realistic noise trajectory:

- **Lightweight & Composable:** Optimised for photonics and quantum optics labs — generate, reproduce, and analyse noise trajectories from a single pipeline with minimal dependencies.

- **Realistic Noise for Theorists:** Provides a direct bridge between experimental laser characterisation and numerical simulation. Move beyond idealised power-law models and stress-test your systems against true noise profile of real hardware.

- **Data-Driven Accuracy**: Preserves the unique "knees" and noise floors of your specific hardware using log-log interpolation of experimental data.  

- **Physics-Aware Extrapolation**: Allows user to choose how to handle high-frequency roll-off beyond your measurement range (e.g., Kohlrausch decay), preventing unphysical artifacts.  


## Features

- Gaussian noise generation from arbitrary experimental PSDs using the TK95 algorithm
- Tools for conversion and standardization of experimental laser noise PSD data in Voltage, frequency, and phase. 
- Laser phase noise simulation for quantum many-body and photonic systems
- Clean, composable API suitable for integration into larger simulation pipelines
- Designed for extensibility (amplitude noise and generalised noise sources planned)
- Relevant demonstrations to show standard use cases

---

## Background

The TK95 algorithm (Timmer & König, 1995) generates stochastic time series whose statistical properties exactly match a given PSD. This makes it particularly well-suited to simulating coloured noise, such as laser phase noise, where the spectral shape is known from experiment but the temporal realisation is random.

NoiPhi applies this approach to quantum simulation contexts, where accurate noise modelling is critical for predicting decoherence, heating rates, and other environmentally driven dynamics.

---

## Installation

From source:

```bash
git clone https://github.com/SpeshalK/NoiPhi.git
cd noiphi
pip install -e .
```

---
## Dependencies

NoiPhi is designed to be lightweight, relying on the standard Python scientific stack to ensure easy integration into existing lab environments.

| Library | Purpose | Min. Version | Required |
| :--- | :--- | :--- | :--- |
| **Python** | Base Language | `3.8+` | Core |
| **NumPy** | FFT operations and array manipulation | `1.20+` | Core |
| **SciPy** | Log-log interpolation of experimental PSDs | `1.7+` | Core |
| **Matplotlib** | Noise visualization and demo plotting | `3.4+` | Demos |

> **Note:** For a complete list of specific sub-dependencies, please see the [pyproject.toml](pyproject.toml) file.

---

## Quick Start

```python
import numpy as np
import noiphi

frequencies = np.logspace(1, 6, 1000)
psd = 1e-10 / frequencies

sim = noiphi.core.PhaseNoiseSimulator(frequencies, psd, dt=1e-6, n_samples=10_000)
t, phi = sim.generateNoise()
```

---

## How It Works

1. **Input a PSD** — provide a measured or analytical power spectral density over a frequency grid.
2. **Gaussian sampling** — the TK95 algorithm draws complex Gaussian amplitudes scaled to the PSD at each frequency bin.
3. **Inverse FFT** — the frequency-domain representation is transformed to produce a stationary time series with the correct spectral statistics.
4. **Output** — a time-domain noise trajectory ready for use in simulation.

---

## Demonstration repo

The demo repository is organized to support a full showcase of the NoiPhi toolkit:

- data/: Example CSV files of experimental laser noise data in different units.

- usage/: API examples showing how to parse experimental data into NoiPhi, set sampling parameters, deploying noise analysis tools, and configure extrapolation modes.

- physics/: Applied examples, such as simulating laser linewidth or phase-driven decoherence in quantum systems.  
---

## Roadmap

- [x] Laser phase noise via TK95 Gaussian sampling
- [x] Edge case handling (set frequencies out of data-range to zero/constant/decay)
- [x] Noise analysis tools (Autocorrelation, Allan Deviation, Cumulative Integrated Phase noise)
- [x] Detailed demonstrations (usage,analysis,pysics)
- [ ] Amplitude noise simulation
- [ ] Utilities for fitting PSDs to experimental data
- [ ] Support for non-Gaussian sampling methods
- [ ] Generalised noise sources (magnetic field fluctuations, intensity noise, etc.)

---

## Citations

If you use NoiPhi in academic work, please also cite the original TK95 algorithm:

> Timmer, J. & König, M. (1995). *On generating power law noise.* Astronomy and Astrophysics, 300, 707–710.

Other work which aided the creation of this software package includes:

> Schmid, F. & Weitenberg, J. & Hänsch, T. W. & Udem, T. & Ozawa, A. (2019) *Simple phase noise measurement scheme for cavity-stabilized laser systems.* Optics Letters, Vol. 44, Issue 11.

> de Léséleuc, S. & Barredo, D. & Lienhard, V. & Browaeys, A. & Lahaye, T (2018). *Analysis of imperfections in the coherent optical excitation of single atoms to Rydberg states.* Phys. Rev. A, Vol. 97, Issue 5.

The noise code in this software has been directly implemented in the following articles:

> Kozlej, T & Pelegri, G & Pritchard, J.D & Daley, A.J. (2026) *Adiabatic state preparation and thermalization of simulated phase noise in a Rydberg spin Hamiltonian.* arXiv:2505.04595 

> Dr. Tomas Kozlej (2026) *Laser Phase noise in Rydberg Atom Arrays.* PhD diss., University of Strathclyde.

---

## Testing

NoiPhi uses [pytest](https://docs.pytest.org). To run the full test suite after installing from source:

```bash
pip install pytest
pytest
```

The suite covers the core simulator (`PhaseNoiseSimulator`), all conversion tools, and the noise analysis toolkit, including physics-based checks (Parseval's theorem, Allan deviation scaling, PDH discriminator roll-off).

---

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

---

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
