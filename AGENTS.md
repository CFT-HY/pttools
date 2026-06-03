# AGENTS.md

## Commands
- Install dependencies: `./install_requirements.sh`
- Run tests: `pytest`
  - The full test suite can take up to 20 min to run. For faster results, you can run only some tests.
- Lint: `pylint`
- Type checking: `mypy`
- Build documentation: `cd docs && make all`
  - Building the documentation will run the examples and can therefore take up to 35 min to run.

## Code style
- Use Python 3.12 type hints where possible
- JIT compile heavy computations with Numba

## General instructions
- When working on code that has physics equations, ensure that the equations are not changed.

## Description of PTtools
PTtools is a library for computing the gravitational wave spectra of first-order cosmological phase transitions.
It is based on the Sound Shell Model, which is introduced in the article "Gravitational waves from first order cosmological phase transitions in the Sound Shell Model" by Hindmarsh & Hijazi (2019).
The article is available here: https://ar5iv.labs.arxiv.org/html/1909.10040

Modules:
- `analysis`: plotting and data analysis tools.
- `bubble`: bubble fluid profile solver
- `models`: equations of state as subclasses of `Model`
- `omgw0`: conversion from the time of GW formation to observable gravitational wave spectrum today, provides `Spectrum` class
- `speedup`: utilities for Numba compilation and parallelism
- `ssm`: Sound Shell Model, provides the `SSMSpectrum` class
- `utils`: generic utilities

The primary use case is that the user provides the input parameters by creating instances of the `Model`, `Bubble` and `Spectrum` classes,
and plots the resulting fluid velocity profile from `Bubble` and the gravitational wave power spectrum from `Spectrum`.
