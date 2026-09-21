"""Locating the documentation directory of the project being documented.

The documentation utilities of PTtools are used also by other projects, such as PTPlot,
in which PTtools is installed as a package in a virtual environment.
The paths of the documentation can therefore not be derived from the location of the PTtools package,
and are instead found relative to the virtual environment or the current working directory.
"""

import os
import sys

from pttools.utils.system import PTTOOLS_DIR

#: Files that a Sphinx documentation directory should contain
DOCS_DIR_FILES: tuple[str, ...] = ("conf.py", "Makefile")
#: Name of the documentation directory
DOCS_DIR_NAME: str = "docs"
#: Name of the log directory, which is alongside the documentation directory
LOG_DIR_NAME: str = "logs"


def is_docs_dir(path: str) -> bool:
    """Check whether the given directory is a Sphinx documentation directory that can be built with ``make``."""
    return all(os.path.isfile(os.path.join(path, name)) for name in DOCS_DIR_FILES)


def env_dir() -> str | None:
    """Path of the Python virtual environment in which Python is running.

    This detects the environments created by ``venv``, ``virtualenv`` and ``uv``,
    which all set :py:data:`sys.prefix` to the environment directory,
    whereas :py:data:`sys.base_prefix` remains the directory of the base Python installation.

    :return: path of the environment, or None if not running in a virtual environment
    """
    if sys.prefix != sys.base_prefix:
        return os.path.abspath(sys.prefix)
    return None


def find_docs_dir(cwd: str | None = None) -> str | None:
    """Find the documentation directory of the project.

    The following locations are checked in order, and the first one that is a documentation directory
    (see :py:func:`is_docs_dir`) is returned:

    1. The ``docs`` directory alongside the virtual environment (e.g. ``venv`` or ``.venv``)
       in which Python is running (see :py:func:`env_dir`).
       This is the case when PTtools is installed as a package in the environment of another project,
       such as PTPlot, and when PTtools itself is run with ``uv run``.
    2. The ``docs`` subdirectory of the current working directory.
    3. The current working directory itself.
    4. The ``docs`` directory of the PTtools repository, if PTtools is run from a source checkout.

    :param cwd: the directory to use as the current working directory, or None for the actual one
    :return: path of the documentation directory, or None if not found
    """
    if cwd is None:
        cwd = os.getcwd()
    cwd = os.path.abspath(cwd)
    candidates: list[str] = []
    if (env := env_dir()) is not None:
        candidates.append(os.path.join(os.path.dirname(env), DOCS_DIR_NAME))
    candidates.append(os.path.join(cwd, DOCS_DIR_NAME))
    candidates.append(cwd)
    candidates.append(os.path.join(os.path.dirname(PTTOOLS_DIR), DOCS_DIR_NAME))
    for candidate in candidates:
        if is_docs_dir(candidate):
            return candidate
    return None


def default_log_dir(docs_dir: str | None = None) -> str:
    """Default directory for the log files, which is ``logs`` alongside the documentation directory.

    :param docs_dir: the documentation directory, or None to find it with :py:func:`find_docs_dir`.
        If the documentation directory is not found, the ``logs`` subdirectory
        of the current working directory is used.
    """
    if docs_dir is None:
        docs_dir = find_docs_dir()
    if docs_dir is None:
        return os.path.join(os.getcwd(), LOG_DIR_NAME)
    return os.path.join(os.path.dirname(os.path.abspath(docs_dir)), LOG_DIR_NAME)
