# AGENTS.md

## Commands
- The project uses [uv](https://docs.astral.sh/uv/) for managing the Python version, the virtual environment and the dependencies.
- Install dependencies: `uv sync --all-extras`
  - This creates the virtual environment `.venv` with the Python version of `.python-version` and the exact package versions of `uv.lock`.
  - Add a dependency with `uv add PACKAGE` (or `uv add --group dev PACKAGE` for development dependencies), which also updates `uv.lock`.
  - Upgrade the locked dependency versions with `uv lock --upgrade`.
- Run tests: `uv run pytest`
  - The full test suite can take up to 20 min to run. For faster results, you can run only some tests.
  - Do not disable the addopts of pyproject.toml with `-o addopts=""`, as this would drop `--dist=loadgroup`.
- Lint: `uv run ruff check`
- Type checking: `uv run pyrefly check`
- Run all lints and type checks: `./lint.sh`
  - This runs `pyrefly check`, `pyrefly coverage check`, `ruff check` and `python -m pttools.docs.lint`.
    All checks are run even if some of them fail. The exit code is 0 if all checks pass,
    the exit code of the failed check if exactly one check fails, and 100 if multiple checks fail.
  - Fast lint: `./lint.sh --fast` skips the significantly slower `python -m pttools.docs.lint`.
  - After changes that create or modify docstrings or files in `./docs/`, run the full lint `./lint.sh` (~2 min).
    After other changes, run the fast lint `./lint.sh --fast`.
- Build documentation with examples: `cd docs && uv run make all`
  - This will run the examples and can therefore take up to 35 min.
- Build documentation without examples: `cd docs && uv run make all-noplot`
- Build the package: `uv build`

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
- After changing equations in docstrings, run `uv run python -m pttools.docs.lint`.
  It builds the documentation without running the examples (`make latexpdf-noplot`), prints the Sphinx errors and warnings
  and the LaTeX errors, and saves the Sphinx output to `./logs/sphinx_TIMESTAMP.log`. Its exit code is that of `make`.
  Fix all reported errors, as the documentation is built with `--fail-on-warning`.

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
