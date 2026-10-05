"""Unit tests for cloc."""

import shutil
import typing as tp
import unittest

from pttools.docs import cloc
from pttools.utils.system import PTTOOLS_DIR


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
