"""Advisory file locks with information on the process that holds the lock.

The locks are taken with :func:`fcntl.flock`, so this module is not available on Windows.
"""

import collections.abc
import fcntl
import json
import os
from pathlib import Path
from typing import Any

__all__ = ["acquire_lock", "read_lock", "write_lock_info"]


def acquire_lock(lock_file: Path) -> int | None:
    """Take an exclusive lock without blocking.

    The lock is released when the returned file descriptor is closed,
    which happens at the latest when this process exits.
    The file descriptor is not inherited by the subprocesses.

    :param lock_file: the lock file, which is created if it does not exist
    :return: the file descriptor of the lock file, or ``None`` if another process holds the lock
    """
    fd = os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def read_lock(lock_file: Path) -> tuple[bool, dict[str, Any] | None]:
    """Check whether a lock is held, and read the information written to it with :func:`write_lock_info`.

    The lock is checked by trying to take a shared lock without blocking.

    :param lock_file: the lock file
    :return: whether the lock is held, and the information of the current or latest holder of the lock, if available
    """
    try:
        file = lock_file.open()
    except FileNotFoundError:
        return False, None
    with file:
        try:
            fcntl.flock(file, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            locked = True
        else:
            locked = False
            fcntl.flock(file, fcntl.LOCK_UN)
        try:
            info = json.loads(file.read())
        except json.JSONDecodeError:
            info = None
    return locked, info if isinstance(info, dict) else None


def write_lock_info(fd: int, info: collections.abc.Mapping[str, object]) -> None:
    """Write information on the holder of a lock to the lock file as JSON, replacing the previous information.

    :param fd: the file descriptor of the lock file from :func:`acquire_lock`
    :param info: the information, e.g. the process ID
    """
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, (json.dumps(info, indent=2) + "\n").encode())
