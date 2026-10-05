"""Tests for importing TBB."""

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

from pttools.speedup import tbb

TBB_INSTALLED: bool
try:
    importlib.metadata.distribution("tbb")
    TBB_INSTALLED = True
except importlib.metadata.PackageNotFoundError:
    TBB_INSTALLED = False


@unittest.skipUnless(TBB_INSTALLED, "The tbb package is not installed.")
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

    def test_main_module(self) -> None:
        """The TBB check can be run with "python -m pttools.speedup.tbb" without warnings."""
        # PTtools initialises colorama on import, which writes a reset escape code to stdout at exit
        # if stdout is a terminal. Colorama treats stdout as a terminal whenever PYCHARM_HOSTED is set,
        # even if it's a pipe, so the variable is removed to get the same output in PyCharm as elsewhere.
        env = {key: value for key, value in os.environ.items() if key != "PYCHARM_HOSTED"}
        result = subprocess.run(
            [sys.executable, "-m", "pttools.speedup.tbb"], check=False, capture_output=True, text=True, env=env
        )
        assert result.returncode == 0, result.stderr
        assert re.search(r"^TBB version: \d+$", result.stdout)
        assert "Warning" not in result.stderr


class TestTBBMain(unittest.TestCase):
    """Test the exit code of the TBB check, which is run with "python -m pttools.speedup.tbb"."""

    NOTE: tp.ClassVar[str] = "TBB is not available for the current CPU architecture"

    @staticmethod
    def run_main(version: int | None, is_x86_64: bool = True) -> tuple[int, str]:
        """Run the TBB check with the given TBB version and architecture, and return the exit code and the output."""
        tbb_main = importlib.import_module("pttools.speedup.tbb.__main__")
        stdout = io.StringIO()
        with mock.patch.object(tbb_main, "TBB_VERSION", version), \
                mock.patch.object(tbb_main, "IS_X86_64", is_x86_64), \
                contextlib.redirect_stdout(stdout):
            returncode = tbb_main.main()
        return returncode, stdout.getvalue()

    def test_main_compatible(self) -> None:
        """A compatible TBB version gives the exit code 0."""
        assert self.run_main(tbb.TBB_MIN_VERSION) == (0, f"TBB version: {tbb.TBB_MIN_VERSION}\n")

    def test_main_too_old(self) -> None:
        """A too old TBB version gives the exit code 1."""
        assert self.run_main(tbb.TBB_MIN_VERSION - 1) == (1, f"TBB version: {tbb.TBB_MIN_VERSION - 1}\n")

    def test_main_not_found(self) -> None:
        """A missing TBB library gives the exit code 1."""
        assert self.run_main(None) == (1, "TBB version: None\n")

    def test_main_not_found_other_architecture(self) -> None:
        """TBB is not required on CPU architectures for which the tbb package is not available."""
        returncode, output = self.run_main(None, is_x86_64=False)
        assert returncode == 0
        assert output.startswith("TBB version: None\n"), output
        assert self.NOTE in output

    def test_main_too_old_other_architecture(self) -> None:
        """A too old TBB version is not an error on other CPU architectures, but the note is printed."""
        returncode, output = self.run_main(tbb.TBB_MIN_VERSION - 1, is_x86_64=False)
        assert returncode == 0
        assert self.NOTE in output

    def test_main_compatible_other_architecture(self) -> None:
        """If a compatible TBB library is found, e.g. from the operating system, the note is not printed."""
        assert self.run_main(tbb.TBB_MIN_VERSION, is_x86_64=False) == (0, f"TBB version: {tbb.TBB_MIN_VERSION}\n")
