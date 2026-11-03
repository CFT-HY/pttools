"""Git utilities."""

import os
import subprocess

__all__ = ["git"]


def git(path: str | os.PathLike[str], *args: str) -> str:
    """Run a Git command in a repository.

    :param path: the directory of the repository
    :param args: the arguments of the Git command
    :return: the stripped standard output of the command
    :raises subprocess.CalledProcessError: if the command fails
    """
    return subprocess.run(["git", *args], cwd=path, stdout=subprocess.PIPE, text=True, check=True).stdout.strip()
