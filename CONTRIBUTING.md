# Contributing to NoiPhi

First off, thank you for taking the time to contribute! 🎉

NoiPhi is an open-source project aimed at bridging the gap between experimental noise data and quantum dynamics simulations. Contributions from both experimentalists and theorists (whether fixing a bug, improving documentation, or adding a new noise model) are highly welcome.

The following guidelines outline how to contribute effectively to this repository.

---

## How Can I Contribute?

### 1. Reporting Bugs & Feature Requests
If you find a bug, an issue with physical units, or have an idea for a feature:
1. Check the [Issues tab](https://github.com/SpeshalK/NoiPhi/issues) to make sure it hasn't already been reported.
2. If it hasn't, open a new Issue. Please include:
   * A clear, descriptive title.
   * Steps to reproduce the bug (if applicable), including minimal sample code or spectrum analyzer data layouts.
   * The expected vs. actual behavior.

### 2. Improving Documentation & Demos
Clear explanations and reliable examples are vital for scientific software. If you notice a typo in a docstring, want to clarify the physics in the README, or want to contribute a new simulation case study (e.g., matching a specific laser setup), feel free to open a Pull Request (PR).

### 3. Submitting Code Contributions
If you want to contribute code to fix a bug or implement a feature:
1. **Fork the repository** and create your branch from `main` (e.g., `git checkout -b feature/amplitude-noise`).
2. Make your changes, ensuring you follow standard Python style conventions (PEP 8).
3. **Write tests:** If you add new core functionality or data pipeline steps, please add corresponding unit tests to the `tests/` directory.

---

## Development Workflow & Code Quality

To ensure the physics engines remain accurate and stable, this repository enforces a few quality control steps using automated CI via GitHub Actions.

### Setting Up Your Environment
Clone your fork and install the package locally in editable mode along with development dependencies:

```bash
git clone [https://github.com/](https://github.com/)<your-username>/NoiPhi.git
cd NoiPhi
pip install -e .[dev]
```

## Running the tests suite

Before opening a Pull Request, verify that all existing functionality works perfectly by running the pytest suite locally (add -v for verbose output):

```bash
pytest -v
```
Please ensure that your contribution does not break existing tests. If your pull request modifies core behaviors or interpolation grids, make sure the test cases are updated accordingly.

## Code of Conduct & Academic Integrity

We expect all contributors to maintain a respectful, collaborative, and inclusive environment that has the sole intent of improving the toolkit. Because this software is used for academic research, we ask that any physical formulas, stochastic derivations, or numerical interpolation methods integrated into the core package are accurately referenced or documented.
