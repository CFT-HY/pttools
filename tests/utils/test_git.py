"""Tests for the Git utilities."""

from pathlib import Path
import subprocess

import pytest

from pttools.utils.git import git


def test_git(tmp_path: Path) -> None:
    """The stripped output of the command should be returned, and a failure should raise an error."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    assert git(tmp_path, "symbolic-ref", "--short", "HEAD") == "main"
    with pytest.raises(subprocess.CalledProcessError):
        git(tmp_path, "rev-parse", "HEAD")
