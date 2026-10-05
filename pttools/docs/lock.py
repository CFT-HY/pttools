"""Locking the generation of files in the documentation source directory.

Some Sphinx extensions generate files in the documentation source directory instead of the build directory:
Sphinx-Gallery generates ``auto_examples``, ``gen_modules/backreferences`` and ``sg_api_usage.rst``,
and apidoc and autosummary generate ``gen_modules``.
Simultaneous builds of the same documentation, e.g. by :py:mod:`pttools.docs.lint`,
would therefore write the same files simultaneously, which can make the builds fail.

The files are generated in the event handlers of ``builder-inited`` and ``env-updated``,
which take only a small part of the build time.
:py:func:`setup_source_lock` holds a file lock during these events,
so that the builds wait for each other only while the files are generated,
and run concurrently for the rest of the time, e.g. while reading the sources and writing the output.
Handlers that raise an exception abort the build, and the lock is then released by the operating system
when the process exits.
"""

import os
from pathlib import Path
import time
import typing as tp

from sphinx.util import logging

from pttools.docs.paths import LINT_DIR_NAME

if tp.TYPE_CHECKING:
    from sphinx.application import Sphinx

logger: logging.SphinxLoggerAdapter = logging.getLogger(__name__)

#: Name of the lock file, which is in the ``lint`` directory of the documentation directory
LOCK_FILE_NAME: str = "source.lock"
#: The Sphinx events whose handlers generate files in the documentation source directory
LOCKED_EVENTS: tuple[str, ...] = ("builder-inited", "env-updated")
#: Priority of the event handler that acquires the lock, which runs before the other handlers of the event.
#: The default priority of Sphinx event handlers is 500, and the handlers with lower priorities run first.
ACQUIRE_PRIORITY: int = -10000
#: Priority of the event handler that releases the lock, which runs after the other handlers of the event
RELEASE_PRIORITY: int = 10000
#: If acquiring the lock takes longer than this, the waiting time is logged, in seconds
LOG_WAIT_THRESHOLD: float = 1


class SourceLock:
    """An exclusive lock on a file, which is shared between processes.

    The lock file is opened when acquiring the lock and closed when releasing it,
    so that the processes forked by parallel Sphinx builds don't inherit it.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        """Create the lock without acquiring it.

        :param path: the lock file, which is created when acquiring the lock
        """
        self.path: Path = Path(path)
        self._fd: int | None = None

    @property
    def locked(self) -> bool:
        """Whether this process holds the lock."""
        return self._fd is not None

    def acquire(self) -> float:
        """Acquire the lock, waiting until other processes have released it.

        :return: the time waited for the lock, in seconds
        """
        if self._fd is not None:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        start = time.perf_counter()
        try:
            _lock(fd)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        return time.perf_counter() - start

    def release(self) -> None:
        """Release the lock, if this process holds it."""
        if self._fd is None:
            return
        try:
            _unlock(self._fd)
        finally:
            os.close(self._fd)
            self._fd = None


if os.name == "nt":
    import msvcrt

    def _lock(fd: int) -> None:
        # LK_LOCK gives up after 10 attempts at one-second intervals, so it's retried until the lock is acquired.
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                return
            except OSError:
                pass

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


def source_lock_path(docs_dir: str | os.PathLike[str]) -> Path:
    """Path of the lock file for the given documentation directory."""
    return Path(docs_dir) / LINT_DIR_NAME / LOCK_FILE_NAME


def setup_source_lock(app: "Sphinx") -> SourceLock:
    """Hold a file lock during the Sphinx events that generate files in the documentation source directory.

    :param app: the Sphinx application
    :return: the lock
    """
    lock = SourceLock(source_lock_path(app.srcdir))

    def acquire(*args: tp.Any) -> None:
        logger.debug("Acquiring the documentation source lock %s", lock.path)
        waited = lock.acquire()
        if waited > LOG_WAIT_THRESHOLD:
            logger.info(
                f"Waited {waited:.0f} s for another documentation build to release the source lock {lock.path}")

    def release(*args: tp.Any) -> None:
        lock.release()

    for event in LOCKED_EVENTS:
        app.connect(event, acquire, priority=ACQUIRE_PRIORITY)
        app.connect(event, release, priority=RELEASE_PRIORITY)
    # In case the lock was not released by the events, e.g. due to an exception in a handler
    app.connect("build-finished", release)
    return lock
