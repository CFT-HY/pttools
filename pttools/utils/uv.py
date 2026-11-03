"""Utilities for the `uv <https://docs.astral.sh/uv/>`_ package manager."""

import os
from pathlib import Path
import shutil
import sys

__all__ = ["env_without_venv", "find_uv"]


def env_without_venv() -> dict[str, str]:
    """Get the environment variables of this process without the virtualenv in which it is run.

    When this process is run in a virtualenv, e.g. with ``uv run``,
    ``$VIRTUAL_ENV`` and the ``bin`` directory of the virtualenv in ``$PATH`` are removed,
    so that the subprocesses that are run with ``uv run --project`` in another project
    see only the virtualenv of that project.

    :return: the environment variables
    """
    env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    if sys.prefix != sys.base_prefix:
        venv_bin = (Path(sys.prefix) / "bin").resolve()
        env["PATH"] = os.pathsep.join(
            directory
            for directory in env.get("PATH", "").split(os.pathsep)
            if not directory or Path(directory).resolve() != venv_bin
        )
    return env


def find_uv() -> str | None:
    """Find uv, which may not be in the PATH, e.g. in cron.

    Cron runs with a minimal PATH, which does not include the user-specific directories
    where the standalone installer of uv puts it.
    ``$UV_INSTALL_DIR`` and ``$XDG_BIN_HOME`` are the custom installation directories supported by the installer,
    ``~/.local/bin`` is its default, and ``~/.cargo/bin`` is the default of older uv versions.
    The existing directories are added to the PATH of this process and its subprocesses,
    unless they are already there.

    :return: the path of uv, or ``None`` if it was not found
    """
    home = Path.home()
    dirs = [home / ".cargo" / "bin", home / ".local" / "bin"]
    dirs += [Path(os.environ[name]) for name in ("XDG_BIN_HOME", "UV_INSTALL_DIR") if os.environ.get(name)]
    path = os.environ.get("PATH", "")
    for directory in dirs:
        if directory.is_dir() and str(directory) not in path.split(os.pathsep):
            path = f"{directory}{os.pathsep}{path}"
    os.environ["PATH"] = path
    return shutil.which("uv")
