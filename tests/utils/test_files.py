"""Tests for the file utilities."""

from pathlib import Path

import pytest

from pttools.utils.files import dir_size, format_size, split_lines, tail


def test_dir_size(tmp_path: Path) -> None:
    """The sizes of the files in the subdirectories should be included, but symbolic links should not be followed."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_bytes(b"a" * 10)
    (tmp_path / "sub" / "b.txt").write_bytes(b"b" * 5)
    (tmp_path / "link.txt").symlink_to(tmp_path / "a.txt")
    assert dir_size(tmp_path) == 15


def test_format_size() -> None:
    """Sizes should be formatted in megabytes with one decimal."""
    assert format_size(0) == "0.0 MB"
    assert format_size(12_345_678) == "12.3 MB"


def test_split_lines() -> None:
    """The lines should be split also at carriage returns, and their trailing whitespace removed."""
    assert split_lines("a  \r\nprogress 1\rprogress 2\nb\n") == ["a", "progress 1", "progress 2", "b", ""]


@pytest.mark.parametrize("n_lines", [0, 3, 10_000])
def test_tail(tmp_path: Path, n_lines: int) -> None:
    """The last lines should be returned also when they span multiple blocks."""
    path = tmp_path / "test.log"
    lines = [f"line {i} " + "x" * 100 for i in range(n_lines)]
    path.write_text("".join(f"{line}\n" for line in lines))
    assert tail(path, 5) == lines[-5:]
