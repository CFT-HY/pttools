"""Tests for the system utilities."""

import os
from pathlib import Path
import platform
import tempfile
import typing as tp
import unittest

import pytest

from pttools.utils import system


class CPUArchTest(unittest.TestCase):
    """Tests for the detection of the CPU architecture."""

    @staticmethod
    def test_cpu_arch() -> None:
        """The architecture should be that of the platform, and at most one of the architecture flags should be set."""
        assert platform.machine() == system.CPU_ARCH
        assert not (system.IS_ARM64 and system.IS_X86_64)
        known = ("aarch64", "arm64", "ARM64", "x86_64", "AMD64")
        assert system.IS_ARM64 or system.IS_X86_64 or system.CPU_ARCH not in known


class ProjectDirTest(unittest.TestCase):
    """Tests for finding the root directory of the project repository."""

    @tp.override
    def setUp(self) -> None:
        self.tmp_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self.root: Path = Path(self.tmp_dir.name).resolve()

    @tp.override
    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def make_venv(self, *parts: str) -> Path:
        """Create a virtual environment directory with a pyvenv.cfg file under the temporary directory."""
        venv = self.root.joinpath(*parts)
        venv.mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("")
        return venv

    def test_source_checkout(self) -> None:
        """When run from a source checkout, the project is the PTtools repository."""
        assert system.project_dir(self.root / "pttools" / "pttools", is_package=False) == self.root / "pttools"

    def test_package(self) -> None:
        """When installed as a package, the project is the repository at whose root the virtual environment is."""
        venv = self.make_venv("project", ".venv")
        for site_packages in (venv / "lib" / "python3.14" / "site-packages", venv / "Lib" / "site-packages"):
            with self.subTest(site_packages=site_packages):
                assert system.project_dir(site_packages / "pttools", is_package=True) == self.root / "project"

    def test_package_other_venv_name(self) -> None:
        """The virtual environment is recognised by its pyvenv.cfg file regardless of its name."""
        venv = self.make_venv("project", "venv")
        pttools_dir = venv / "lib" / "python3.14" / "site-packages" / "pttools"
        assert system.project_dir(pttools_dir, is_package=True) == self.root / "project"

    def test_package_without_venv(self) -> None:
        """A package that is not in a virtual environment has no project directory."""
        pttools_dir = self.root / "usr" / "lib" / "python3.14" / "site-packages" / "pttools"
        assert system.project_dir(pttools_dir, is_package=True) is None

    def test_current(self) -> None:
        """PROJECT_DIR corresponds to the current installation of PTtools."""
        assert system.project_dir() == system.PROJECT_DIR
        if not system.IS_PACKAGE:
            assert system.PTTOOLS_DIR.parent == system.PROJECT_DIR



def test_physical_cpu_count() -> None:
    """The number of physical cores should be at least 1 and at most the number of logical CPUs."""
    assert 1 <= system.physical_cpu_count() <= (os.cpu_count() or 1)


@pytest.fixture
def no_sys_topology(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the CPU topology in /sys unavailable."""
    read_text = Path.read_text

    def fake_read_text(self: Path, *args: tp.Any, **kwargs: tp.Any) -> str:
        if str(self).startswith("/sys/devices/system/cpu/"):
            raise FileNotFoundError(self)
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)


@pytest.mark.usefixtures("no_sys_topology")
def test_physical_cpu_count_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the CPU topology in /sys, the physical cores should be counted with psutil."""
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(system, "psutil", psutil)
    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: 3 if not logical else 6)
    assert system.physical_cpu_count() == 3


@pytest.mark.usefixtures("no_sys_topology")
def test_physical_cpu_count_without_psutil(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without the CPU topology in /sys and psutil, the number of logical CPUs should be returned.

    An error should be logged.
    """
    monkeypatch.setattr(system, "psutil", None)
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    assert system.physical_cpu_count() == cpus
    output = capsys.readouterr().out
    assert "ERROR: The CPU topology is not available in /sys, and psutil is not installed" in output


if __name__ == "__main__":
    unittest.main()
