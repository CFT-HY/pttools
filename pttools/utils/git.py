"""Git and GitHub utilities."""

import os
from pathlib import Path
import subprocess

__all__ = ["git", "git_commit_id", "git_ls_files", "git_toplevel", "github_warning", "tracked_files"]


def git(path: str | os.PathLike[str], *args: str, strip: bool = True) -> str:
    """Run a Git command in a repository.

    :param path: the directory of the repository
    :param args: the arguments of the Git command
    :param strip: whether to strip the leading and trailing whitespace from the output
    :return: the standard output of the command
    :raises subprocess.CalledProcessError: if the command fails
    """
    output = subprocess.run(["git", *args], cwd=path, stdout=subprocess.PIPE, text=True, check=True).stdout
    return output.strip() if strip else output


def git_commit_id(path: str | os.PathLike[str], rev: str = "HEAD") -> str:
    """Get the full commit ID of a revision with ``git rev-parse``.

    :param path: the directory of the repository
    :param rev: the revision, e.g. ``HEAD``, a branch, a tag or an abbreviated commit ID
    :return: the full commit ID of the revision
    :raises subprocess.CalledProcessError: if the revision does not exist or does not point to a commit
    """
    # --verify ensures that the revision exists, and ^{commit} that annotated tags are resolved to their commits.
    return git(path, "rev-parse", "--verify", "--end-of-options", f"{rev}^{{commit}}")


def git_ls_files(path: str | os.PathLike[str], *args: str) -> list[str]:
    """List the files tracked by Git in the given directory.

    :param path: the directory
    :param args: additional arguments to ``git ls-files``
    :return: paths of the files relative to the given directory
    :raises subprocess.CalledProcessError: if the command fails
    """
    # With -z the paths are not quoted, even if they contain special characters.
    # The output is not stripped, as the paths may begin or end with whitespace.
    output = git(path, "ls-files", "-z", "--cached", *args, "--", ".", strip=False)
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
