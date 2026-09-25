"""Unit tests for the cProfile utilities."""

import cProfile
import os.path
import pstats
import tempfile
import unittest

from tests.profiling.utils_cprofile import save_sorted


class TestSaveSorted(unittest.TestCase):
    """Tests for save_sorted."""

    def test_file_names(self) -> None:
        profile = cProfile.Profile()
        profile.enable()
        sum(range(10))
        profile.disable()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "profile")
            for sort in ("cumulative", pstats.SortKey.TIME):
                with self.subTest(sort=sort):
                    save_sorted(profile, path, sort)
            for name in ("cumulative", "time"):
                for suffix in ("", "_numba", "_all"):
                    self.assertTrue(os.path.isfile(f"{path}_{name}{suffix}.txt"))
