"""File utilities."""

import os
from pathlib import Path
import re

__all__ = ["dir_size", "format_size", "split_lines", "tail"]


def dir_size(path: Path) -> int:
    """Get the total size of the files in a directory and its subdirectories, without following symbolic links.

    :param path: the directory
    :return: the total size in bytes
    """
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file() and not item.is_symlink())


def format_size(size: int) -> str:
    """Format a size in a human-readable format.

    :param size: the size in bytes
    :return: the size in megabytes, e.g. ``12.3 MB``
    """
    return f"{size / 1e6:.1f} MB"


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
