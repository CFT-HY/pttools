"""Documentation tests."""

import contextlib
import io
from pathlib import Path
import shutil
import tempfile
import typing as tp
import unittest
from unittest import mock

from pttools.docs import cloc, lint, paths
from pttools.utils import IS_GITHUB_ACTIONS
from pttools.utils.system import PTTOOLS_DIR


class DocsTest(unittest.TestCase):
    """Tests for the Sphinx configuration."""

    @unittest.skipIf(IS_GITHUB_ACTIONS, "Docs dependencies are not installed for CI test job")
    def test_docs_conf(self) -> None:
        """Test that the Sphinx configuration can be imported and has the correct project name."""
        from docs import conf  # noqa: PLC0415
        assert conf.project == "PTtools"


class DocsPathsTest(unittest.TestCase):
    """Tests for finding the documentation directory of the project being documented."""

    @tp.override
    def setUp(self) -> None:
        self.tmp_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self.root: Path = Path(self.tmp_dir.name).resolve()

    @tp.override
    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def make_docs_dir(self, *parts: str) -> Path:
        """Create a directory with the files of a documentation directory under the temporary directory."""
        path = self.root.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        for name in paths.DOCS_DIR_FILES:
            (path / name).write_text("")
        return path

    def test_is_docs_dir(self) -> None:
        """Test that a directory is recognised as a docs directory only if it has all the required files."""
        assert not paths.is_docs_dir(self.root)
        docs = self.make_docs_dir("docs")
        assert paths.is_docs_dir(docs)
        (docs / "Makefile").unlink()
        assert not paths.is_docs_dir(docs)

    def test_find_docs_dir_env(self) -> None:
        """The docs directory alongside the virtual environment is preferred."""
        docs = self.make_docs_dir("project", "docs")
        env = self.root / "project" / "venv"
        env.mkdir()
        with mock.patch.object(paths, "env_dir", return_value=env):
            assert paths.find_docs_dir(cwd=self.root) == docs

    def test_find_docs_dir_cwd_subdir(self) -> None:
        """The docs subdirectory of the working directory is found."""
        docs = self.make_docs_dir("project", "docs")
        with mock.patch.object(paths, "env_dir", return_value=None):
            assert paths.find_docs_dir(cwd=self.root / "project") == docs

    def test_find_docs_dir_cwd(self) -> None:
        """The working directory is found if it is itself a docs directory."""
        docs = self.make_docs_dir("project", "docs")
        with mock.patch.object(paths, "env_dir", return_value=None):
            assert paths.find_docs_dir(cwd=docs) == docs

    def test_find_docs_dir_pttools_repo(self) -> None:
        """When run from the PTtools repository, its docs are found even if not in a virtual environment."""
        with mock.patch.object(paths, "env_dir", return_value=None):
            found = paths.find_docs_dir(cwd=self.root)
        pttools_docs = paths.PTTOOLS_DIR.parent / paths.DOCS_DIR_NAME
        if paths.is_docs_dir(pttools_docs):
            assert found == pttools_docs
        else:
            assert found is None

    def test_find_docs_dir_not_found(self) -> None:
        """None is returned if no docs directory is found."""
        with mock.patch.object(paths, "env_dir", return_value=None), \
                mock.patch.object(paths, "PTTOOLS_DIR", self.root / "site-packages" / "pttools"):
            assert paths.find_docs_dir(cwd=self.root) is None

    def test_default_log_dir(self) -> None:
        """The log directory is next to the docs directory, or in the working directory if there are no docs."""
        docs = self.make_docs_dir("project", "docs")
        assert paths.default_log_dir(docs) == self.root / "project" / "logs"
        with mock.patch.object(paths, "find_docs_dir", return_value=None):
            assert paths.default_log_dir() == Path.cwd() / "logs"


class ClocTest(unittest.TestCase):
    """Tests for the compact formatting of the output of cloc."""

    DATA: tp.ClassVar[dict[str, dict[str, dict[str, tp.Any]]]] = {
        "by_file": {
            "header": {
                "cloc_url": "github.com/AlDanial/cloc", "cloc_version": "1.98", "elapsed_seconds": 0.5,
                "n_files": 3, "n_lines": 1234, "files_per_second": 6.0, "lines_per_second": 2468.0,
            },
            "pttools/bubble/very_long_file_name_to_test_the_column_width.py": {
                "blank": 100, "comment": 200, "code": 800, "language": "Python",
            },
            "README.md": {"blank": 10, "comment": 0, "code": 50, "language": "Markdown"},
            "pttools/bubble/a.py": {"blank": 1, "comment": 2, "code": 3, "language": "Python"},
            "SUM": {"blank": 111, "comment": 202, "code": 853, "nFiles": 3},
        },
        "by_lang": {
            "Python": {"nFiles": 2, "blank": 101, "comment": 202, "code": 803},
            "Markdown": {"nFiles": 1, "blank": 10, "comment": 0, "code": 50},
            "SUM": {"blank": 111, "comment": 202, "code": 853, "nFiles": 3},
        },
    }

    def test_format_compact(self) -> None:
        """Test that the cloc JSON output is formatted compactly, grouped by directory."""
        assert cloc.format_compact(self.DATA) == (
            "github.com/AlDanial/cloc v 1.98  T=0.50 s (6.0 files/s, 2468.0 lines/s)\n"
            "-----------------------------------------------------------------------\n"
            "File                                               blank  comment  code\n"
            "-----------------------------------------------------------------------\n"
            "./\n"
            "  README.md                                           10        0    50\n"
            "pttools/bubble/\n"
            "  very_long_file_name_to_test_the_column_width.py    100      200   800\n"
            "  a.py                                                 1        2     3\n"
            "-----------------------------------------------------------------------\n"
            "SUM:                                                 111      202   853\n"
            "-----------------------------------------------------------------------\n"
            "\n"
            "-------------------------------------\n"
            "Language  files  blank  comment  code\n"
            "-------------------------------------\n"
            "Python        2    101      202   803\n"
            "Markdown      1     10        0    50\n"
            "-------------------------------------\n"
            "SUM:          3    111      202   853\n"
            "-------------------------------------\n"
        )

    @unittest.skipIf(shutil.which("cloc") is None, "cloc is not installed")
    def test_cloc_compact(self) -> None:
        """The compact output has the same counts as the output of cloc, and shorter lines."""
        path = PTTOOLS_DIR
        full = cloc.cloc(path).splitlines()
        compact = cloc.cloc_compact(path).splitlines()

        def counts(lines: list[str]) -> list[tuple[str, ...]]:
            return sorted(tuple(line.split()[-3:]) for line in lines if line and line[-1].isdigit())

        # The header line with the timing is excluded, as it differs between the runs.
        assert counts(full[1:]) == counts(compact[1:])
        assert max(len(line) for line in compact[1:]) < max(len(line) for line in full[1:])


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


if __name__ == "__main__":
    unittest.main()
