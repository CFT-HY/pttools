"""Tests for the process utilities."""

import os
import subprocess
import sys
import time

import pytest

from pttools.utils import process

pytestmark: pytest.MarkDecorator = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Finding the processes requires /proc of Linux."
)


def test_read_process() -> None:
    """The information and the command line of this process should be read."""
    info = process.read_process(os.getpid())
    assert info is not None
    assert info.pid == os.getpid()
    assert info.ppid == os.getppid()
    assert process.is_alive(info)
    assert process.read_cmdline(os.getpid())[0] == sys.orig_argv[0]
    assert str(os.getpid()) in process.format_process(info)


def test_stop() -> None:
    """A process tree should be stopped, including a child that ignores SIGTERM."""
    with subprocess.Popen(["sh", "-c", 'sh -c \'trap "" TERM; exec sleep 1000\' & sleep 1000 & wait']) as parent:
        try:
            deadline = time.monotonic() + 30
            while len(descendants := process.find_descendants([parent.pid])) < 2:
                assert time.monotonic() < deadline, "The child processes were not created."
                time.sleep(0.1)
            root = process.read_process(parent.pid)
            assert root is not None
            targets = {parent.pid: root, **descendants}
            assert process.stop(targets, timeout=1) == []
            assert not any(process.is_alive(target) for target in targets.values())
        finally:
            parent.kill()
