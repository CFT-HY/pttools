For developers
==============

Developing a new feature
------------------------
Create a new feature branch in the repo.
If you don't have permissions to create a branch in the repo,
you can either request the permissions or create a fork.
Feature branches and forks can be merged without squashing.


Developing a hotfix
-------------------
Small bugfixes and improvements can be done in a separate hotfix branch.
This branch should be merged to main without squashing.


Creating a new release
----------------------
- Update the PTtools version number in
    - CITATION.cff (update also the release date)
    - codemeta.json (update also the release date in the dateModified field)
    - pyproject.toml
- Ensure that the unit tests pass and that the documentation is generated successfully
- Check these for warnings, errors and unnecessary log output and fix them if necessary
    - Unit test logs
    - Documentation logs
    - Ruff logs
    - Pyrefly logs


Updating Python version requirements
------------------------------------
When updating the Python version requirements,
update the version numbers in:

- .github/workflows/\*.yml
- .python-version
- .readthedocs.yaml
- Dockerfile
- environment.yml
- pyproject.toml

Then regenerate the lock file with ``uv lock`` and commit the updated ``uv.lock``.


Updating dependencies
---------------------
The dependencies are declared in ``pyproject.toml``,
and the exact versions used for development, testing and the documentation builds are pinned in ``uv.lock``.

- Add a dependency with ``uv add PACKAGE``,
  or ``uv add --group dev PACKAGE`` / ``uv add --group docs PACKAGE`` for the development and documentation tools.
- Upgrade all dependencies to the newest versions allowed by ``pyproject.toml`` with ``uv lock --upgrade``,
  or a single package with ``uv lock --upgrade-package PACKAGE``.
- Run the unit tests after upgrading, and commit the updated ``uv.lock``.
- When updating the versions in ``pyproject.toml``, also update ``environment.yml``.


Numba caching
-------------
Numba-jitted functions are cached on disk when ``NUMBA_ENABLE_CACHE`` is enabled.
The cache key of a compiled function consists of the Numba types of its arguments.
Anything in those types that differs between processes results in a different cache key on every run,
which prevents the cache from ever hitting and makes the cache files grow without bound.
Integers and floats are safe, but the following are not.

- Jitted functions should not be given other jitted functions as arguments,
  since the Numba type of such an argument is tied to the identity of the dispatcher object,
  which is created anew in every process.
  Give them the address of a cfunc instead, as is done with
  ``cs2_ptr`` (see ``pttools.bubble.cs2``) and ``df_dtau_ptr``.
- Pointers should not be used as default values of jitted functions,
  since Numba treats an omitted argument as a compile-time constant,
  and includes its value in the type of that argument.
- Pointers should not be read from module-level variables within jitted functions either.
  Their values are not a part of the cache key,
  and therefore the cached machine code would contain an address from a previous process.
- Cached functions that call ``parallel=True`` functions need the Numba threading layer to be launched
  before their machine code can be loaded from the cache.
  Numba tracks this with ``reload_init``, but loses it for callees that were themselves loaded from the cache.
  PTtools works around this in ``pttools.speedup.numba_fixes``.
  Without the workaround, a function compiled in a process that loaded a parallel callee from the cache
  crashes the next process that loads it from the cache: with a segmentation fault on Linux,
  and with ``Fatal Python error: Aborted`` from LLVM on macOS.

The $c_s^2$ functions of the models and the differential equations of the fluid profiles based on them
are created dynamically, and therefore they cannot be cached on disk:
their cache keys would include the Numba dispatcher objects they capture,
and those are identified by a random UUID that differs between processes.
Instead, they are compiled once per process and per model,
and ``ConstCSModel`` shares them between all models with the same sound speeds
(see ``pttools.models.const_cs.const_cs_funcs``).
Each worker process of a process pool therefore compiles them once for each distinct set of sound speeds,
and models that are pickled to the workers reuse the functions compiled there.
