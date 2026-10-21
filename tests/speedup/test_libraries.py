"""Tests for loading the native libraries of Numba, i.e. SVML and TBB."""

import contextlib
import importlib
import importlib.metadata
import io
import os
import re
import subprocess
import sys
import typing as tp
import unittest
from unittest import mock

from pttools.speedup import libraries
from pttools.speedup.libraries import library, svml, tbb


def installed(package: str) -> bool:
    """Whether the given PyPI package is installed."""
    try:
        importlib.metadata.distribution(package)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


SVML_INSTALLED: bool = any(installed(package) for package in svml.SVML.packages)
TBB_INSTALLED: bool = any(installed(package) for package in tbb.TBB.packages)


def run_module(*args: str) -> subprocess.CompletedProcess:
    """Run a module with ``python -m`` and return the result."""
    # PTtools initialises colorama on import, which writes a reset escape code to stdout at exit
    # if stdout is a terminal. Colorama treats stdout as a terminal whenever PYCHARM_HOSTED is set,
    # even if it's a pipe, so the variable is removed to get the same output in PyCharm as elsewhere.
    env = {key: value for key, value in os.environ.items() if key != "PYCHARM_HOSTED"}
    return subprocess.run([sys.executable, "-m", *args], check=False, capture_output=True, text=True, env=env)


@unittest.skipUnless(SVML_INSTALLED, "No package providing SVML is installed.")
class TestSVML(unittest.TestCase):
    """Test that the SVML library is found for Numba."""

    @staticmethod
    def test_loaded() -> None:
        """The SVML library is loaded on import."""
        assert svml.SVML_LOADED
        assert svml.SVML.loaded

    @staticmethod
    def test_llvmlite_load() -> None:
        """The SVML library can be loaded by its name with llvmlite, as Numba does on import."""
        import llvmlite.binding  # noqa: PLC0415

        assert svml.SVML.file_name is not None
        # On Windows, Numba gives the name without the extension.
        llvmlite.binding.load_library_permanently(svml.SVML.file_name.removesuffix(".dll"))

    @staticmethod
    def test_numba_uses_svml() -> None:
        """Numba uses SVML if llvmlite has been built with it."""
        assert svml.numba_uses_svml() == svml.llvmlite_has_svml()

    @staticmethod
    def test_main_module() -> None:
        """The SVML check can be run with "python -m pttools.speedup.libraries svml" without warnings."""
        result = run_module("pttools.speedup.libraries", "svml")
        assert result.returncode == 0, result.stderr
        assert re.search(r"^SVML loaded: True$", result.stdout, re.MULTILINE), result.stdout
        assert "Warning" not in result.stderr


@unittest.skipUnless(TBB_INSTALLED, "No package providing TBB is installed.")
class TestTBB(unittest.TestCase):
    """Test that the TBB library of the tbb package is found for Numba."""

    def test_load_tbb(self) -> None:
        """Test that the TBB library is loaded and is recent enough."""
        version = tbb.load_tbb()
        assert version is not None
        assert version >= tbb.TBB_MIN_VERSION

    @staticmethod
    def test_numba_tbb_layer() -> None:
        """Test that the Numba TBB threading layer can be imported and passes the TBB version check."""
        # Importing the TBB extension of Numba fails if the loader cannot find the TBB library.
        importlib.import_module("numba.np.ufunc.tbbpool")
        # This is the check that Numba runs before using the TBB threading layer.
        importlib.import_module("numba.np.ufunc.parallel")._check_tbb_version_compatible()  # noqa: SLF001

    def test_tbb_version(self) -> None:
        """The TBB library is loaded on import."""
        assert tbb.TBB_VERSION is not None
        assert tbb.TBB_VERSION >= tbb.TBB_MIN_VERSION

    def test_package_library(self) -> None:
        """The TBB library of the installed package can be loaded by its path and is recent enough."""
        dirs = tbb.tbb_library_dirs()
        assert dirs
        assert tbb.get_tbb_version(dirs[0]) >= tbb.TBB_MIN_VERSION

    @staticmethod
    def test_main_module() -> None:
        """The TBB check can be run with "python -m pttools.speedup.libraries tbb" without warnings."""
        result = run_module("pttools.speedup.libraries", "tbb")
        assert result.returncode == 0, result.stderr
        assert re.search(r"^TBB version: \d+$", result.stdout, re.MULTILINE), result.stdout
        assert "Warning" not in result.stderr


class TestCheck(unittest.TestCase):
    """Test the output and the exit code of the library checks, i.e. ``python -m pttools.speedup.libraries``."""

    NOTE: tp.ClassVar[str] = "is not required"

    @staticmethod
    def run_check(
            lib: library.NativeLibrary,
            loaded: bool,
            version: int | None = None,
            is_x86_64: bool = True,
            packages: tuple[str, ...] = ()) -> tuple[int, str]:
        """Run the check of the library with the given state, and return the exit code and the output."""
        stdout = io.StringIO()
        with mock.patch.object(lib, "library", mock.Mock() if loaded else None), \
                mock.patch.object(lib, "path", None), \
                mock.patch.object(lib, "version", version), \
                mock.patch.object(lib, "installed_packages", return_value=list(packages)), \
                mock.patch.object(library, "IS_X86_64", is_x86_64), \
                contextlib.redirect_stdout(stdout):
            returncode = lib.check()
        return returncode, stdout.getvalue()

    def test_tbb_compatible(self) -> None:
        """A compatible TBB version gives the exit code 0."""
        version = tbb.TBB_MIN_VERSION
        assert self.run_check(tbb.TBB, True, version) == (0, f"TBB version: {version}\n")

    def test_tbb_too_old(self) -> None:
        """A too old TBB version gives the exit code 1."""
        version = tbb.TBB_MIN_VERSION - 1
        assert self.run_check(tbb.TBB, False, version) == (1, f"TBB version: {version}\n")

    def test_tbb_not_found(self) -> None:
        """A missing TBB library gives the exit code 1 on x86-64."""
        assert self.run_check(tbb.TBB, False) == (1, "TBB version: None\n")

    def test_tbb_not_found_package_installed(self) -> None:
        """A missing TBB library gives the exit code 1 if its package is installed, e.g. tbb4u on ARM."""
        assert self.run_check(tbb.TBB, False, is_x86_64=False, packages=("tbb4u",)) == (1, "TBB version: None\n")

    def test_tbb_not_found_other_architecture(self) -> None:
        """TBB is not required on other CPU architectures if none of its packages is installed."""
        returncode, output = self.run_check(tbb.TBB, False, is_x86_64=False)
        assert returncode == 0
        assert output.startswith("TBB version: None\n"), output
        assert self.NOTE in output

    def test_tbb_compatible_other_architecture(self) -> None:
        """If a compatible TBB library is found, e.g. from the operating system, the note is not printed."""
        version = tbb.TBB_MIN_VERSION
        assert self.run_check(tbb.TBB, True, version, is_x86_64=False) == (0, f"TBB version: {version}\n")

    def test_library_loaded(self) -> None:
        """A loaded library without a version gives the exit code 0."""
        lib = library.NativeLibrary("Test", {}, ("test-package",))
        assert self.run_check(lib, True) == (0, "Test loaded: True\n")

    def test_library_not_found(self) -> None:
        """A missing library without a version gives the exit code 1 on x86-64."""
        lib = library.NativeLibrary("Test", {}, ("test-package",))
        assert self.run_check(lib, False) == (1, "Test loaded: False\n")

    def test_library_not_found_other_architecture(self) -> None:
        """A missing library is not required on other CPU architectures, e.g. SVML on ARM."""
        lib = library.NativeLibrary("Test", {}, ("test-package",))
        returncode, output = self.run_check(lib, False, is_x86_64=False)
        assert returncode == 0
        assert output.startswith("Test loaded: False\n"), output
        assert self.NOTE in output

    def test_check_libraries(self) -> None:
        """The exit code of checking multiple libraries is 1 if any of them fails."""
        with mock.patch.object(svml.SVML, "check", return_value=0), mock.patch.object(tbb.TBB, "check", return_value=1):
            assert libraries.check_libraries() == 1
            assert libraries.check_libraries(["svml"]) == 0

    @staticmethod
    def test_main_module() -> None:
        """All the libraries can be checked with "python -m pttools.speedup.libraries" without warnings."""
        result = run_module("pttools.speedup.libraries")
        assert result.returncode == 0, result.stderr
        assert re.search(r"^SVML loaded: (True|False)$", result.stdout, re.MULTILINE), result.stdout
        assert re.search(r"^TBB version: (\d+|None)$", result.stdout, re.MULTILINE), result.stdout
        assert "Warning" not in result.stderr


class TestLoad(unittest.TestCase):
    """Test loading a library that is not installed."""

    def test_not_found(self) -> None:
        """A library that is not found is not loaded, and the loader does not raise."""
        lib = library.NativeLibrary(
            "Test", {"linux": "libpttools_missing.so", "darwin": "libpttools_missing.dylib", "win32": "missing.dll"},
            ("pttools-missing-package",)
        )
        assert lib.search_dirs() == []
        with self.assertLogs(library.logger, "INFO"):
            assert not lib.load()
        assert not lib.loaded
        assert lib.version is None
