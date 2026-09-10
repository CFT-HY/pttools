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
- .readthedocs.yaml
- Dockerfile
- pyproject.toml


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
