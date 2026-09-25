"""Documentation tests."""

from pathlib import Path
import shutil
import tempfile
import typing as tp
import unittest
from unittest import mock

from pttools.docs import cloc, paths
from pttools.utils import IS_GITHUB_ACTIONS
from pttools.utils.system import PTTOOLS_DIR


class DocsTest(unittest.TestCase):
    @unittest.skipIf(IS_GITHUB_ACTIONS, "Docs dependencies are not installed for CI test job")
    def test_docs_conf(self) -> None:
        from docs import conf  # noqa: PLC0415
        self.assertEqual(conf.project, "PTtools")


class DocsPathsTest(unittest.TestCase):
    """Tests for finding the documentation directory of the project being documented."""

    def setUp(self) -> None:
        self.tmp_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self.root: Path = Path(self.tmp_dir.name).resolve()

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def make_docs_dir(self, *parts: str) -> Path:
        path = self.root.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        for name in paths.DOCS_DIR_FILES:
            (path / name).write_text("")
        return path

    def test_is_docs_dir(self) -> None:
        self.assertFalse(paths.is_docs_dir(self.root))
        docs = self.make_docs_dir("docs")
        self.assertTrue(paths.is_docs_dir(docs))
        (docs / "Makefile").unlink()
        self.assertFalse(paths.is_docs_dir(docs))

    def test_find_docs_dir_env(self) -> None:
        """The docs directory alongside the virtual environment is preferred."""
        docs = self.make_docs_dir("project", "docs")
        env = self.root / "project" / "venv"
        env.mkdir()
        with mock.patch.object(paths, "env_dir", return_value=env):
            self.assertEqual(paths.find_docs_dir(cwd=self.root), docs)

    def test_find_docs_dir_cwd_subdir(self) -> None:
        docs = self.make_docs_dir("project", "docs")
        with mock.patch.object(paths, "env_dir", return_value=None):
            self.assertEqual(paths.find_docs_dir(cwd=self.root / "project"), docs)

    def test_find_docs_dir_cwd(self) -> None:
        docs = self.make_docs_dir("project", "docs")
        with mock.patch.object(paths, "env_dir", return_value=None):
            self.assertEqual(paths.find_docs_dir(cwd=docs), docs)

    def test_find_docs_dir_pttools_repo(self) -> None:
        """When run from the PTtools repository, its docs are found even if not in a virtual environment."""
        with mock.patch.object(paths, "env_dir", return_value=None):
            found = paths.find_docs_dir(cwd=self.root)
        pttools_docs = paths.PTTOOLS_DIR.parent / paths.DOCS_DIR_NAME
        if paths.is_docs_dir(pttools_docs):
            self.assertEqual(found, pttools_docs)
        else:
            self.assertIsNone(found)

    def test_find_docs_dir_not_found(self) -> None:
        with mock.patch.object(paths, "env_dir", return_value=None), \
                mock.patch.object(paths, "PTTOOLS_DIR", self.root / "site-packages" / "pttools"):
            self.assertIsNone(paths.find_docs_dir(cwd=self.root))

    def test_default_log_dir(self) -> None:
        docs = self.make_docs_dir("project", "docs")
        self.assertEqual(paths.default_log_dir(docs), self.root / "project" / "logs")
        with mock.patch.object(paths, "find_docs_dir", return_value=None):
            self.assertEqual(paths.default_log_dir(), Path.cwd() / "logs")


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
        self.assertEqual(
            cloc.format_compact(self.DATA),
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
        self.assertEqual(counts(full[1:]), counts(compact[1:]))
        self.assertLess(max(len(line) for line in compact[1:]), max(len(line) for line in full[1:]))


if __name__ == "__main__":
    unittest.main()
