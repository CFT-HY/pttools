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
  - Git pre-commit hooks run the checks of the fast lint in parallel on the staged changes, and block the commit if any of them fails.
    They also run the built-in hooks of prek, which fix trailing whitespace, missing newlines at the end of files and mixed line endings,
    check the syntax of JSON, TOML and YAML files, and check for large files and for the consistency of shebangs and executable bits.
    The hooks are managed with [prek](https://github.com/j178/prek), configured in `prek.toml`,
    and installed with `uv run prek install`. When changing the checks of `./lint.sh --fast`, update the hooks accordingly.
  - After changes that create or modify any reST or math markup in docstrings or `.rst` files, or files in `./docs/`, run the full lint `./lint.sh` (~2 min).
    After other changes, run the fast lint `./lint.sh --fast`.
- Build documentation with examples: `uv run make -C docs all`
  - This will run the examples and can therefore take up to 35 min.
- Build documentation without examples: `uv run make -C docs all-noplot`
- Build the package: `uv build`

## CI
- The CI workflow `.github/workflows/main.yml` runs all the CI jobs on every push.
  Most jobs start only after the `lint`, `typecheck` and `build` jobs have passed,
  and the Windows and macOS tests after `test-numba`.
  The `docs` job lints the documentation with `python -m pttools.docs.lint`, checks the links,
  and then builds the full documentation. Broken links are reported as warnings instead of failing the job,
  see `--broken-links-as-warnings` of `pttools.docs.lint`.
  The Docker image is published for every commit that passes the Linux tests,
  and the PyPI package and the Software Heritage archive only for version tags, after all the other jobs have passed.
- The following are also used by other repositories, such as PTPlot, from the `dev` branch:
  - the reusable workflows `build.yml`, `deploy-shf.yml`, `docs.yml`, `graph.yml`, `lint.yml` and `typecheck.yml`,
  - the composite actions `.github/actions/deploy-docker`, `.github/actions/setup-env` and `.github/actions/test`.

  Keep their inputs backwards compatible.
  The reusable workflows must not use local actions (`./.github/actions/...`),
  as those would be resolved in the repository of the caller.
  Jobs that authenticate with OpenID Connect (PyPI trusted publishing, Docker Hub, attestations)
  must not be in reusable workflows, since the token would then identify the reusable workflow instead of the caller.

## Code style
- Use Python 3.12+ type hints where possible.
- JIT compile heavy computations with Numba.
- Prefer to organize modules alphabetically (ignoring the leading underscores)
  with constants first, then classes, and then functions.
  However, the exceptions are that `logger` should be right after the imports, and that `main` should be the last function in a module.
  You may deviate from these if required for e.g. type hints.

## Docstring conventions
- Use the Sphinx docstring format.
- Use `:param:`, `:return:` and `:raises:`, where appropriate.
  The descriptions of physics variables should begin with the form `$symbol$, name`, where appropriate.
- For functions that return a physics variable,
  the first line of the docstring should be of the form `$symbol$, name.`, where appropriate.
- If a function contains physics equations, add them as LaTeX in its docstring.
- When using equations from articles, cite the article, including the number of the equation, if possible.
- Use Sphinx extlinks for references, as configured in `./pttools/docs/links.py`.
- The attributes that are exportable fields (see `pttools.utils.fields`) are documented
  by the `description` of the field in the `export.py` module of the package, instead of a `#:` comment.
  `pttools.docs.fields` gives these descriptions to Sphinx autodoc as if they were `#:` comments.
  The fields of properties and methods have no `description`,
  as it's taken from the first line of their docstring, which should therefore be of the form `$symbol$, name.`
  As the descriptions are parsed as reStructuredText before `$...$` is converted to math,
  write e.g. `$c_s^2(T_{\text{gw}})$` instead of `$c_s^2(T_\text{gw})$`, since `T_` after `(` would be parsed as a link.
- After changing equations in docstrings, run `uv run python -m pttools.docs.lint`.
  It builds the documentation without running the examples or generating the figures of the `plot` directive (`make latexpdf-noplot`), prints the Sphinx errors and warnings
  and the LaTeX errors, and saves the Sphinx output to `./logs/sphinx_TIMESTAMP.log`. Its exit code is that of `make`.
  It builds in a new temporary directory `./docs/lint/lint_TIMESTAMP_RANDOM`, whose path it prints,
  so multiple lints can be run simultaneously. These directories are not removed automatically.
  Fix all reported errors, as the documentation is built with `--fail-on-warning`.
- In docstring equations, write `\ast` instead of `*` after an opening bracket, e.g. `r_{\ast,0}`,
  as reStructuredText reads `{*` as the start of italics.

## General instructions
- Before editing code that has physics equations, ensure that there are unit tests that verify the results of that code.
  If there are no such unit tests yet, create them. Use the existing output of the code as a reference,
  and also reference values from the literature, if there are any.
- If you change any of the physics, inform the user explicitly and exactly what has been changed and why.
- When changing the physics beyond bug fixes, add an argument for switching to the old behaviour, if feasible.

## Description of PTtools
PTtools is a library for computing the gravitational wave spectra of first-order cosmological phase transitions.
It is based on the Sound Shell Model, which is introduced in the article
"Gravitational waves from first order cosmological phase transitions in the Sound Shell Model"
by Hindmarsh & Hijazi (2019), arXiv:1909.10040.
Links to other relevant articles are in `pttools.docs.links.EXTLINKS_STATIC`.

Sub-packages:
- `analysis`: data analysis tools
- `bubble`: bubble fluid profile solver
- `docs`: documentation utilities
- `export`: exporting and importing models, bubbles and spectra as HDF5 files
- `lattice`: importing the data of hydrodynamic lattice simulations, e.g. those of Correia et al. (2025) (`lattice.chrw25`)
- `math`: generic mathematical and numerical utilities, e.g. integration, fits and assertions
- `models`: equations of state as subclasses of `Model`
- `omgw0`: conversion from the time of GW formation to observable gravitational wave spectrum today, provides `Spectrum` class
- `plot`: plotting
- `speedup`: utilities for Numba compilation and parallelism
- `ssm`: Sound Shell Model, provides the `SSMSpectrum` class
- `utils`: generic utilities

Examples are available in `examples`.
The primary use case is that the user provides the input parameters by creating instances of the `Model`, `Bubble` and `Spectrum` classes,
and plots the resulting fluid velocity profile from `Bubble` and the gravitational wave power spectrum from `Spectrum`.
For an example of this primary use case, see `examples/basic/basic.py`.

To avoid circular dependencies, each sub-package in `pttools` should depend only on the sub-packages before it in this order:
`utils`, `speedup`, `math`, `bubble`, `models`, `ssm`, `omgw0`, `lattice`, `export`, `analysis`, `plot`, `docs`.
Scripts in `./docs/fig/` and `./tests/test_examples.py` import from `examples`, and some scripts in `examples` import from `tests`.
This is not an issue, since the `__init__.py` files in `docs.fig`, `examples` and `tests` don't import the package contents.

## Related repositories
You may `git clone` these with SSH in the same parent directory as PTtools.
- [PTPlot](https://github.com/CFT-HY/PTPlot): `../PTPlot`, plotting software that uses PTtools. (public)
  - When updating the PTtools public API, check whether PTPlot needs to be updated as well.
    If there are no uncommitted changes in the relevant files or other active sessions editing the code,
    you may edit the cloned repository directly. PTPlot is maintained by the same authors as PTtools,
    and therefore backwards-incompatible changes in the PTtools public API are OK.
    However, PTPlot uses PTtools from a pinned commit.
    Therefore, any changes in PTPlot that would not pass its lint or tests without updating its PTtools
    should either be postponed until the pin has been updated first (as updating the pin requires approval from the user),
    or should be performed in a separate worktree with PTtools pointed to this version of the repository.
    In case of postponing the PTPlot changes, provide the user with a summary of what should or could be changed in PTPlot.
- [PhD research of Mika Mäki](https://github.com/AgenttiX/phd-research): `../phd-research` (private)
  - Relevant articles are available at `../phd-research/literature/`
- Legacy PTtools-related repositories by Mark Hindmarsh & Chloe Gowling (private)
  - [fisher_analysis](https://bitbucket.org/cgowling/fisher_analysis): `../fisher_analysis`
    - Gowling & Hindmarsh (2021), "Observational prospects for phase transitions at LISA: Fisher matrix analysis" (Gowling:2021gcy, 2106.05984)
  - [gw_data_tools](https://bitbucket.org/cgowling/gw_data_tools): `../gw_data_tools`
  - [MCMC SSM](https://bitbucket.org/cgowling/mcmc_ssm): `../MCMC_SSM`
  - [MCMC SSM 2.0](https://bitbucket.org/cgowling/mcmc_ssm_2_0): `../MCMC_SSM_2_0`
  - [PTtools Omgw0 addons](https://bitbucket.org/cgowling/pttools_omgw0_addons): `../pttools_omgw0_addons`
    - The precursor of the `pttools.omgw0` module
  - [sound-shell-model](https://bitbucket.org/hindmars/sound-shell-model): `../sound-shell-model`
    - The precursor of the `tests.paper` module
    - Hindmarsh (2018), "Sound Shell Model for Acoustic Gravitational Wave Production at a First-Order Phase Transition in the Early Universe" (Hindmarsh:2016lnk, 1608.04735)
    - Hindmarsh & Hijazi (2019), "Gravitational waves from first order cosmological phase transitions in the Sound Shell Model" (Hindmarsh:2019phv, 1909.10040)
- Correia et al. (2025), "Gravitational waves from strong first-order phase transitions" (Correia:2025qif, 2505.17824)
  - [Hydro_plot](https://github.com/jrcccc/Hydro_plot): `../Hydro_plot`, another version of the plotting code (private)
  - [UETC_paper_Hydro](https://github.com/jrcccc/UETC_paper_Hydro): `../UETC_paper_Hydro`, the code and data used in the article (private)
