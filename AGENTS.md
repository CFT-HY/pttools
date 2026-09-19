# AGENTS.md

## Commands
- Install dependencies: `./install_requirements.sh`
- Run tests: `pytest`
  - The full test suite can take up to 20 min to run. For faster results, you can run only some tests.
  - Do not disable the addopts of pyproject.toml with `-o addopts=""`, as this would drop `--dist=loadgroup`.
- Lint: `ruff check`
- Type checking: `pyrefly check`
- Build documentation with examples: `cd docs && make all`
  - This will run the examples and can therefore take up to 35 min to run.
- Build documentation without examples: `cd docs && make all-noplot`

## Code style
- Use Python 3.12+ type hints where possible.
- JIT compile heavy computations with Numba.

## Docstring conventions
- Use the Sphinx docstring format.
- Use `:param:`, `:return:` and `:raises:`, where appropriate.
  The descriptions of physics variables should begin with the form `$symbol$, name`, where appropriate.
- For functions that return a physics variable,
  the first line of the docstring should be of the form `$symbol$, name.`, where appropriate.
- If a function contains physics equations, add them as LaTeX in its docstring.
- When using equations from articles, cite the article, including the number of the equation, if possible.
- Use Sphinx extlinks for references, as configured in `./pttools/docs/links.py`.

## General instructions
- Before editing code that has physics equations, ensure that there are unit tests that verify the results of that code.
  If there are no such unit tests yet, create them. Use the existing output of the code as a reference,
  and also reference values from the literature, if there are any.
- If you change any of the physics, inform the user explicitly and exactly what has been changed and why.
- PTtools is used by PTPlot, which may be available at `../PTPlot`.

## Description of PTtools
PTtools is a library for computing the gravitational wave spectra of first-order cosmological phase transitions.
It is based on the Sound Shell Model, which is introduced in the article
"Gravitational waves from first order cosmological phase transitions in the Sound Shell Model"
by Hindmarsh & Hijazi (2019), arXiv:1909.10040.
Links to other relevant articles are in `pttools.docs.links.EXTLINKS_STATIC`.

Modules:
- `analysis`: plotting and data analysis tools
- `bubble`: bubble fluid profile solver
- `docs`: documentation utilities
- `models`: equations of state as subclasses of `Model`
- `omgw0`: conversion from the time of GW formation to observable gravitational wave spectrum today, provides `Spectrum` class
- `speedup`: utilities for Numba compilation and parallelism
- `ssm`: Sound Shell Model, provides the `SSMSpectrum` class
- `utils`: generic utilities

Examples are available in `examples`.
The primary use case is that the user provides the input parameters by creating instances of the `Model`, `Bubble` and `Spectrum` classes,
and plots the resulting fluid velocity profile from `Bubble` and the gravitational wave power spectrum from `Spectrum`.
For an example of this primary use case, see `examples/basic/basic.py`.
