"""Unit tests for linting the documentation."""

import contextlib
import io
from pathlib import Path
import shutil
import tempfile
import typing as tp
import unittest

from pttools.docs import lint


@unittest.skipIf(shutil.which("make") is None, "make is not installed")
class DocsLintRunMakeTest(unittest.TestCase):
    """Tests for running make with the documentation lint, using a stub Makefile."""

    @tp.override
    def setUp(self) -> None:
        self.tmp_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self.docs_dir: Path = Path(self.tmp_dir.name).resolve()
        (self.docs_dir / "Makefile").write_text("hello:\n\t@echo line1\n\t@echo line2\n")
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
