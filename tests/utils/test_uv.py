"""Tests for the uv utilities."""

import os
from pathlib import Path
import sys

import pytest

from pttools.utils.uv import env_without_venv, find_uv


def test_env_without_venv(monkeypatch: pytest.MonkeyPatch) -> None:
    """The virtualenv of this process should be removed from the environment."""
    venv_bin = str(Path(sys.prefix) / "bin")
    monkeypatch.setenv("VIRTUAL_ENV", sys.prefix)
    monkeypatch.setenv("PATH", os.pathsep.join([venv_bin, "/usr/bin", "/bin"]))
    env = env_without_venv()
    assert "VIRTUAL_ENV" not in env
    expected = ["/usr/bin", "/bin"] if sys.prefix != sys.base_prefix else [venv_bin, "/usr/bin", "/bin"]
    assert env["PATH"] == os.pathsep.join(expected)


def test_find_uv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The uv executable should be found in $UV_INSTALL_DIR, which is added to the PATH only once."""
    fake_uv = tmp_path / "uv"
    fake_uv.write_text("#!/bin/sh\n")
    fake_uv.chmod(0o755)
    monkeypatch.setenv("UV_INSTALL_DIR", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin")
    assert find_uv() == str(fake_uv)
    assert find_uv() == str(fake_uv)
    assert os.environ["PATH"].split(os.pathsep).count(str(tmp_path)) == 1
