"""Tests for the file locks."""

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("fcntl", reason="The locks use fcntl, which is not available on Windows.")

from pttools.utils.lock import acquire_lock, read_lock, write_lock_info


def test_lock(tmp_path: Path) -> None:
    """The lock should be detected while it is held, and the information should be readable from the lock file."""
    lock_file = tmp_path / "test.lock"
    assert read_lock(lock_file) == (False, None)

    info = {"pid": 123, "started": "2026-10-04T12:25:06"}
    fd = acquire_lock(lock_file)
    assert fd is not None
    try:
        write_lock_info(fd, info)
        assert read_lock(lock_file) == (True, info)
        # Another holder should not get the lock.
        assert acquire_lock(lock_file) is None
        # The previous information is replaced.
        write_lock_info(fd, {"pid": 1})
        assert read_lock(lock_file) == (True, {"pid": 1})
    finally:
        os.close(fd)
    assert read_lock(lock_file) == (False, {"pid": 1})
    assert json.loads(lock_file.read_text()) == {"pid": 1}


def test_read_lock_invalid(tmp_path: Path) -> None:
    """A lock file without valid information should be read as having no information."""
    lock_file = tmp_path / "test.lock"
    for content in ("", "not json", "[1, 2]"):
        lock_file.write_text(content)
        assert read_lock(lock_file) == (False, None)
