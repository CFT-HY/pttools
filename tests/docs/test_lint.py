"""Unit tests for linting the documentation."""

import contextlib
import io
import os
from pathlib import Path
import shutil
import tempfile
import typing as tp
import unittest
from unittest import mock

from pttools.docs import lint
from pttools.docs.paths import LINT_DIR_NAME, NO_PLOTS_ENV_VAR


@unittest.skipIf(shutil.which("make") is None, "make is not installed")
class DocsLintRunMakeTest(unittest.TestCase):
    """Tests for running make with the documentation lint, using a stub Makefile."""

    @tp.override
    def setUp(self) -> None:
        self.tmp_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self.docs_dir: Path = Path(self.tmp_dir.name).resolve()
        (self.docs_dir / "Makefile").write_text(
            "BUILDDIR = _build\nhello:\n\t@echo line1\n\t@echo line2\nbuilddir:\n\t@echo \"$(BUILDDIR)\"\n"
            # These print the Sphinx options given with O and the environment variable for disabling the figures.
            f"opts:\n\t@echo \"O=$(O) plots=$${{{NO_PLOTS_ENV_VAR}}}\"\n"
            f"opts-noplot:\n\t@echo \"O=$(O) plots=$${{{NO_PLOTS_ENV_VAR}}}\"\n")
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

    def run_opts(self, target: str, env: dict[str, str] | None = None, jobs: int | None = None) -> str:
        """Run a stub target that prints the Sphinx options and the environment variable for disabling the figures."""
        environ = {key: value for key, value in os.environ.items() if key not in (NO_PLOTS_ENV_VAR, "O")}
        environ.update(env or {})
        with mock.patch.dict(os.environ, environ, clear=True), contextlib.redirect_stdout(io.StringIO()):
            returncode, lines = lint.run_make(target, self.log_path, self.docs_dir, jobs=jobs)
        assert returncode == 0
        return next(line for line in lines if line.startswith("O="))

    def test_run_make_noplot(self) -> None:
        """The -noplot targets are run without the figures, unless the user enables them."""
        assert self.run_opts("opts-noplot") == "O= plots=1"
        assert self.run_opts("opts-noplot", env={NO_PLOTS_ENV_VAR: "0"}) == "O= plots=0"

    def test_run_make_plot(self) -> None:
        """The other targets are run with the figures."""
        assert self.run_opts("opts") == "O= plots="

    def test_run_make_jobs(self) -> None:
        """The number of parallel Sphinx processes is given to make with O."""
        assert self.run_opts("opts", jobs=3) == "O=--jobs 3 plots="
        # The O of the user comes last, so that its --jobs takes precedence.
        assert self.run_opts("opts", env={"O": "--jobs 5"}, jobs=3) == "O=--jobs 3 --jobs 5 plots="


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
