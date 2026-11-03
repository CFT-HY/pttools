"""Git and GitHub utilities."""

import os
from pathlib import Path
import subprocess

__all__ = ["git", "git_ls_files", "git_toplevel", "github_warning", "tracked_files"]


def git(path: str | os.PathLike[str], *args: str) -> str:
    """Run a Git command in a repository.

    :param path: the directory of the repository
    :param args: the arguments of the Git command
    :return: the stripped standard output of the command
    :raises subprocess.CalledProcessError: if the command fails
    """
    return subprocess.run(["git", *args], cwd=path, stdout=subprocess.PIPE, text=True, check=True).stdout.strip()


def git_ls_files(path: str | os.PathLike[str], *args: str) -> list[str]:
    """List the files tracked by Git in the given directory.

    :param path: the directory
    :param args: additional arguments to ``git ls-files``
    :return: paths of the files relative to the given directory
    :raises subprocess.CalledProcessError: if the command fails
    """
    # With -z the paths are not quoted, even if they contain special characters.
    output = subprocess.run(
        ["git", "ls-files", "-z", "--cached", *args, "--", "."],
        cwd=path, check=True, stdout=subprocess.PIPE, text=True
    ).stdout
    return [file for file in output.split("\0") if file]


def git_toplevel(path: str | os.PathLike[str]) -> Path:
    """Get the root directory of the Git repository that contains the given directory.

    :param path: the directory
    :return: the root directory of the repository
    :raises subprocess.CalledProcessError: if the directory is not in a Git repository
    """
    return Path(git(path, "rev-parse", "--show-toplevel"))


def github_warning(title: str, message: str) -> str:
    """Create a warning annotation for GitHub Actions.

    The special characters of the message are escaped, as required by the workflow commands of GitHub Actions.

    :param title: the title of the annotation
    :param message: the message of the annotation
    :return: the workflow command, which creates the annotation when printed to stdout in a workflow
    """
    escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return f"::warning title={title}::{escaped}"


def tracked_files(path: str | os.PathLike[str], exclude_per_directory: str | None = None) -> list[str]:
    """Get the files that are tracked by Git, optionally excluding those matched by ignore files.

    The files that have been deleted from the working tree but not from the Git index are skipped,
    as are the Git submodules.

    :param path: the directory
    :param exclude_per_directory: the name of the ignore files, e.g. ``.clocignore``,
        which have the same syntax and semantics as ``.gitignore`` files.
        Git reads them from the given directory, its parent directories within the repository and its subdirectories.
    :return: paths of the files relative to the given directory
    :raises subprocess.CalledProcessError: if Git fails
    """
    ignored = set() if exclude_per_directory is None else set(
        git_ls_files(path, "--ignored", f"--exclude-per-directory={exclude_per_directory}")
    )
    return [file for file in git_ls_files(path) if file not in ignored and (Path(path) / file).is_file()]
