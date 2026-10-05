"""Unit tests for the lock of the documentation source directory."""

from pathlib import Path
import subprocess
import sys
import tempfile
import time
import typing as tp
import unittest

from pttools.docs import lock
from pttools.docs.paths import LINT_DIR_NAME


class FakeApp:
    """A stand-in for the Sphinx application, which records the connected event handlers."""

    def __init__(self, srcdir: Path) -> None:
        """Create the application with the given source directory."""
        self.srcdir: Path = srcdir
        self.handlers: list[tuple[str, tp.Callable[..., tp.Any], int]] = []

    def connect(self, event: str, callback: tp.Callable[..., tp.Any], priority: int = 500) -> None:
        """Record the event handler."""
        self.handlers.append((event, callback, priority))


class SourceLockTest(unittest.TestCase):
    """Tests for the source lock."""

    @tp.override
    def setUp(self) -> None:
        self.tmp_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self.path: Path = Path(self.tmp_dir.name).resolve()
        self.lock_path: Path = lock.source_lock_path(self.path)

    @tp.override
    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_source_lock_path(self) -> None:
        """The lock file is in the lint directory."""
        assert self.lock_path == self.path / LINT_DIR_NAME / lock.LOCK_FILE_NAME

    def test_acquire_release(self) -> None:
        """The lock can be acquired and released repeatedly, and extra calls do nothing."""
        source_lock = lock.SourceLock(self.lock_path)
        for _ in range(2):
            source_lock.acquire()
            assert source_lock.locked
            source_lock.acquire()
            assert source_lock.locked
            source_lock.release()
            assert not source_lock.locked
            source_lock.release()
        assert self.lock_path.is_file()

    def test_other_process_waits(self) -> None:
        """Another process waits until the lock is released."""
        source_lock = lock.SourceLock(self.lock_path)
        source_lock.acquire()
        code = (
            "import sys\n"
            "from pttools.docs.lock import SourceLock\n"
            "SourceLock(sys.argv[1]).acquire()\n"
            "print('acquired')\n"
        )
        with subprocess.Popen(
                [sys.executable, "-c", code, str(self.lock_path)], stdout=subprocess.PIPE, text=True) as process:
            try:
                # Wait long enough for the process to import PTtools and to block on the lock.
                time.sleep(10)
                assert process.poll() is None
            finally:
                source_lock.release()
            stdout, _ = process.communicate(timeout=60)
        assert process.returncode == 0
        assert stdout.strip() == "acquired"

    def test_setup_source_lock(self) -> None:
        """The lock is held between the first and the last handler of the events that generate source files."""
        app = FakeApp(self.path)
        source_lock = lock.setup_source_lock(app)  # type: ignore[arg-type]
        for event in lock.LOCKED_EVENTS:
            handlers = sorted((handler for handler in app.handlers if handler[0] == event), key=lambda h: h[2])
            assert len(handlers) == 2
            assert handlers[0][2] < 0
            assert handlers[1][2] > 1000
            handlers[0][1](app)
            assert source_lock.locked
            handlers[1][1](app)
            assert not source_lock.locked
        finished = [handler for handler in app.handlers if handler[0] == "build-finished"]
        assert len(finished) == 1
        source_lock.acquire()
        finished[0][1](app, None)
        assert not source_lock.locked
