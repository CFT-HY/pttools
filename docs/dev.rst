For developers
==============

Developing a new feature
------------------------
Create a new feature branch in the repo.
If you don't have permissions to create a branch in the repo,
you can either request the permissions or create a fork.
Feature branches and forks should be merged with quashing.

Developing a hotfix
-------------------
Small bugfixes and improvements can be done in a separate hotfix branch.
This branch should be merged to main without squashing.

Creating a new release
----------------------
Update the PTtools version number in:

- CITATION.cff
- codemeta.json
- pyproject.toml

Updating Python version requirements
------------------------------------
When updating the Python version requirements,
update the version numbers in:

- .github/workflows/\*.yml
- .readthedocs.yaml
- Dockerfile
- pyproject.toml
