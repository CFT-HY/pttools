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
    - codemeta.json (update also the release date)
    - pyproject.toml
- Update the dateModified field in codemeta.json.
- Check these for warnings, errors and unnecessary log output and fix them if necessary
    - Unit test logs
    - Documentation logs


Updating Python version requirements
------------------------------------
When updating the Python version requirements,
update the version numbers in:

- .github/workflows/\*.yml
- .readthedocs.yaml
- Dockerfile
- pyproject.toml
