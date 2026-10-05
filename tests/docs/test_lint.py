"""Unit tests for linting the documentation."""

import contextlib
import io
from pathlib import Path
import shutil
import tempfile
import typing as tp
import unittest

from pttools.docs import lint
from pttools.docs.paths import LINT_DIR_NAME


@unittest.skipIf(shutil.which("make") is None, "make is not installed")
class DocsLintRunMakeTest(unittest.TestCase):
    """Tests for running make with the documentation lint, using a stub Makefile."""

    @tp.override
    def setUp(self) -> None:
        self.tmp_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self.docs_dir: Path = Path(self.tmp_dir.name).resolve()
        (self.docs_dir / "Makefile").write_text(
            "BUILDDIR = _build\nhello:\n\t@echo line1\n\t@echo line2\nbuilddir:\n\t@echo \"$(BUILDDIR)\"\n")
        self.log_path: Path = self.docs_dir / "test.log"

    @tp.override
    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def run_make(self, verbose: bool) -> tuple[int, list[str], str]:
        """Run the stub make target and return the return code, the output lines and the printed output."""
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            returncode, lines = lint.run_make("hello", self.log_path, self.docs_dir, verbose=verbose)
        return returncode, lines, stdout.getvalue()

    def test_run_make(self) -> None:
        """The console output of make is returned and appended to the log file, but not printed."""
        returncode, lines, printed = self.run_make(verbose=False)
        assert returncode == 0
        assert "line1" in lines
        assert "line2" in lines
        assert printed == ""
        log = self.log_path.read_text()
        assert 'Console output of "make hello"' in log
        assert "line1\nline2" in log

    def test_run_make_verbose(self) -> None:
        """With verbose, the console output of make is also printed."""
        returncode, lines, printed = self.run_make(verbose=True)
        assert returncode == 0
        assert "line1" in lines
        assert printed == "line1\nline2\n"

    def test_run_make_failure(self) -> None:
        """A failing make target gives a non-zero return code."""
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            returncode, _ = lint.run_make("nonexistent", self.log_path, self.docs_dir)
        assert returncode != 0

    def test_run_make_build_dir(self) -> None:
        """The build directory overrides BUILDDIR of the Makefile, and its LaTeX directory is removed."""
        build_dir = self.docs_dir / "_build" / "custom"
        latex_dir = build_dir / lint.LATEX_SUBDIR_NAME
        latex_dir.mkdir(parents=True)
        with contextlib.redirect_stdout(io.StringIO()):
            returncode, lines = lint.run_make("builddir", self.log_path, self.docs_dir, build_dir=build_dir)
        assert returncode == 0
        assert str(build_dir) in lines
        assert not latex_dir.exists()

    def test_run_make_default_build_dir(self) -> None:
        """Without a build directory, the default build directory of the Makefile is used."""
        with contextlib.redirect_stdout(io.StringIO()):
            returncode, lines = lint.run_make("builddir", self.log_path, self.docs_dir)
        assert returncode == 0
        assert "_build" in lines


class DocsLintPathsTest(unittest.TestCase):
    """Tests for the build directories and log files of the documentation lint."""

    @tp.override
    def setUp(self) -> None:
        self.tmp_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self.path: Path = Path(self.tmp_dir.name).resolve()

    @tp.override
    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_create_temp_build_dir(self) -> None:
        """Each call creates a new directory in the lint directory."""
        first = lint.create_temp_build_dir(self.path, "2026-01-01_00-00-00")
        second = lint.create_temp_build_dir(self.path, "2026-01-01_00-00-00")
        assert first != second
        for build_dir in (first, second):
            assert build_dir.is_dir()
            assert build_dir.parent == self.path / LINT_DIR_NAME
            assert build_dir.name.startswith("lint_2026-01-01_00-00-00_")

    def test_create_log_file(self) -> None:
        """Log files with the same timestamp get unique names."""
        log_dir = self.path / "logs"
        first = lint.create_log_file(log_dir, "2026-01-01_00-00-00")
        second = lint.create_log_file(log_dir, "2026-01-01_00-00-00")
        assert first == log_dir / "sphinx_2026-01-01_00-00-00.log"
        assert second == log_dir / "sphinx_2026-01-01_00-00-00_1.log"
        assert first.is_file()
        assert second.is_file()
