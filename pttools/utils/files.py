"""File utilities."""

import os
from pathlib import Path
import re

__all__ = ["SIZE_UNITS", "dir_size", "format_size", "split_lines", "tail", "unique_path"]

#: Units of :py:func:`format_size` with decimal prefixes
SIZE_UNITS: tuple[str, ...] = ("B", "KB", "MB", "GB", "TB")


def dir_size(path: Path) -> int:
    """Get the total size of the files in a directory and its subdirectories, without following symbolic links.

    :param path: the directory
    :return: the total size in bytes
    """
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file() and not item.is_symlink())


def format_size(size: float) -> str:
    """Format a size in a human-readable format with decimal prefixes.

    The unit is the largest of B, KB, MB, GB and TB in which the size is at least one,
    e.g. ``512 B``, ``12.3 MB`` or ``1.5 TB``.

    :param size: the size in bytes
    :return: the formatted size, in bytes without decimals and in the larger units with one decimal
    """
    # The sizes are compared after rounding, so that e.g. 999 950 B is 1.0 MB instead of 1000.0 KB.
    if abs(round(size)) < 1000:  # noqa: PLR2004
        return f"{size:.0f} B"
    for unit in SIZE_UNITS[1:-1]:
        size /= 1000
        if abs(round(size, 1)) < 1000:  # noqa: PLR2004
            return f"{size:.1f} {unit}"
    return f"{size / 1000:.1f} {SIZE_UNITS[-1]}"


def split_lines(text: str) -> list[str]:
    """Split text, such as console output, into lines, including the progress lines separated by carriage returns.

    The trailing whitespace of each line is removed.

    :param text: the text
    :return: the lines
    """
    return [line.rstrip() for line in re.split(r"\r\n|\r|\n", text)]


def tail(path: Path, lines: int) -> list[str]:
    """Read the last lines of a file without reading the whole file.

    :param path: the path of the file
    :param lines: the number of lines
    :return: the last lines
    """
    block_size = 8192
    with path.open("rb") as file:
        position = file.seek(0, os.SEEK_END)
        data = b""
        while position > 0 and data.count(b"\n") <= lines:
            read_size = min(block_size, position)
            position -= read_size
            file.seek(position)
            data = file.read(read_size) + data
    return data.decode(errors="replace").splitlines()[-lines:]


def unique_path(path: Path, suffix: str | None = None) -> Path:
    """Get a path that does not exist yet, so that an existing file is not overwritten.

    If the path already exists, a suffix ``_2``, ``_3`` etc. is added to the filename before its extension.

    :param path: the preferred path
    :param suffix: the extension of the filename, e.g. ``.tar.xz``, by default the last suffix of the path
    :return: the path, or the first one with a number suffix that does not exist
    """
    if suffix is None:
        suffix = path.suffix
    stem = path.name.removesuffix(suffix)
    unique = path
    i = 2
    while unique.exists():
        unique = path.with_name(f"{stem}_{i}{suffix}")
        i += 1
    return unique
