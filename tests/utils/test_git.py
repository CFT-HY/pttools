"""Tests for the Git utilities."""

from pathlib import Path
import subprocess

import pytest

from pttools.utils.git import git, git_ls_files, git_toplevel, github_warning, tracked_files


def test_git(tmp_path: Path) -> None:
    """The stripped output of the command should be returned, and a failure should raise an error."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    assert git(tmp_path, "symbolic-ref", "--short", "HEAD") == "main"
    with pytest.raises(subprocess.CalledProcessError):
        git(tmp_path, "rev-parse", "HEAD")


def init_repo(path: Path) -> Path:
    """Create a Git repository with tracked, untracked, ignored and deleted files.

    :param path: the directory of the repository
    :return: the subdirectory ``sub`` of the repository
    """
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    sub = path / "sub"
    (sub / "deep").mkdir(parents=True)
    for file in ("a.py", "b.txt", "deep/c.py", "deleted.py", "with space.py"):
        (sub / file).write_text("x\n")
    (path / "top.py").write_text("x\n")
    (path / ".testignore").write_text("*.txt\n")
    (sub / "deep" / ".testignore").write_text("c.py\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    (sub / "untracked.py").write_text("x\n")
    (sub / "deleted.py").unlink()
    return sub


def test_git_ls_files(tmp_path: Path) -> None:
    """The files tracked by Git in the directory should be listed relative to it, including the deleted ones."""
    sub = init_repo(tmp_path)
    assert sorted(git_ls_files(sub)) == [
        "a.py", "b.txt", "deep/.testignore", "deep/c.py", "deleted.py", "with space.py"
    ]


def test_git_toplevel(tmp_path: Path) -> None:
    """The root of the repository should be found from a subdirectory."""
    sub = init_repo(tmp_path)
    assert git_toplevel(sub).resolve() == tmp_path.resolve()


def test_github_warning() -> None:
    """The special characters of the message should be escaped."""
    assert github_warning("Title", "50% done\nnext\rline") == "::warning title=Title::50%25 done%0Anext%0Dline"


def test_tracked_files(tmp_path: Path) -> None:
    """The deleted files should be skipped, and the ignore files of the parent directories and subdirectories used."""
    sub = init_repo(tmp_path)
    assert sorted(tracked_files(sub)) == ["a.py", "b.txt", "deep/.testignore", "deep/c.py", "with space.py"]
    assert sorted(tracked_files(sub, exclude_per_directory=".testignore")) == [
        "a.py", "deep/.testignore", "with space.py"
    ]
