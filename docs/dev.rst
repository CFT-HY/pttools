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
    - Pylint logs
    - Mypy logs


Updating Python version requirements
------------------------------------
When updating the Python version requirements,
update the version numbers in:

- .github/workflows/\*.yml
- .readthedocs.yaml
- Dockerfile
- pyproject.toml
