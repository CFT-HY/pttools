r"""Unit tests for calling the $c_s^2$ functions by their pointers."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from pttools.bubble.const import CS0_2
from pttools.bubble.cs2 import cs2_from_ptr, cs2_to_ptr
from pttools.bubble.cs2_bag import CS2_BAG_SCALAR_PTR
from pttools.bubble.phase import Phase
from pttools.models import BagModel, ConstCSModel
from pttools.models.model import Model
from pttools.speedup import njit
from pttools.speedup.options import NUMBA_DISABLE_JIT
import pttools.type_hints as th
from pttools.utils import assert_allclose
from tests.utils import REPO_DIR

#: Script that solves a bag model bubble and reports the number of keys in each Numba cache index
CACHE_SCRIPT_PATH: Path = Path(__file__).resolve().parent / "numba_cache.py"


@njit
def cs2_from_ptr_jit(cs2_ptr: th.CS2FunScalarPtr, w: float, phase: float) -> float:
    r"""Compute $c_s^2$ by a pointer in a jitted function."""
    return cs2_from_ptr(cs2_ptr, w, phase)


class TestCS2Ptr(unittest.TestCase):
    r"""Test that the $c_s^2$ functions can be called by their pointers."""

    W = np.array([0.5, 1., 2., 10.])

    def check_model(self, model: Model) -> None:
        r"""Check that the pointer of the model gives the same $c_s^2$ as the function of the model."""
        cs2_ptr = model.cs2_ptr()
        for phase in (Phase.SYMMETRIC, Phase.BROKEN):
            for w in self.W:
                ref = float(model.cs2(w, phase.value))
                assert_allclose(cs2_from_ptr(cs2_ptr, w, phase.value), ref)
                assert_allclose(cs2_from_ptr_jit(cs2_ptr, w, phase.value), ref)

    def test_bag_ptr(self) -> None:
        r"""The pointer of the Bag Model should give $c_s^2 = \frac{1}{3}$."""
        for phase in (Phase.SYMMETRIC, Phase.BROKEN):
            assert_allclose(cs2_from_ptr(CS2_BAG_SCALAR_PTR, 1., phase.value), CS0_2)
            assert_allclose(cs2_from_ptr_jit(CS2_BAG_SCALAR_PTR, 1., phase.value), CS0_2)

    def test_bag_model(self) -> None:
        self.check_model(BagModel(a_s=1.1, a_b=1, V_s=1))

    def test_const_cs_model(self) -> None:
        model = ConstCSModel(a_s=1.5, a_b=1, V_s=1, css2=1/3 - 0.01, csb2=1/3 - 0.02)
        self.check_model(model)
        # The pointer should be the same on every call, so that the callers don't have to be recompiled.
        self.assertEqual(model.cs2_ptr(), model.cs2_ptr())

    def test_const_cs_model_bag(self) -> None:
        """A ConstCSModel that is equivalent to the Bag Model should use the pointer of the Bag Model."""
        model = ConstCSModel(a_s=1.5, a_b=1, V_s=1, css2=1/3, csb2=1/3)
        self.assertEqual(model.cs2_ptr(), CS2_BAG_SCALAR_PTR)
        self.check_model(model)

    def test_custom_function(self) -> None:
        r"""A custom $c_s^2$ function should be callable by its pointer."""
        @njit
        def cs2(w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
            return (0.2 * phase + 0.3 * (1 - phase)) * np.ones_like(w)

        cs2_ptr = cs2_to_ptr(cs2)
        assert_allclose(cs2_from_ptr(cs2_ptr, 1., Phase.SYMMETRIC.value), 0.3)
        assert_allclose(cs2_from_ptr(cs2_ptr, 1., Phase.BROKEN.value), 0.2)
        assert_allclose(cs2_from_ptr_jit(cs2_ptr, 1., Phase.SYMMETRIC.value), 0.3)
        assert_allclose(cs2_from_ptr_jit(cs2_ptr, 1., Phase.BROKEN.value), 0.2)


class TestNumbaCache(unittest.TestCase):
    """Test that the Numba cache works for the functions that take a $c_s^2$ pointer.

    Passing a jitted function as an argument would result in a different cache key on every run,
    which would prevent the cache from ever hitting and make the cache files grow without bound.
    See the "Numba caching" section of the developer documentation.
    """

    @unittest.skipIf(NUMBA_DISABLE_JIT, "Nothing is compiled when jitting is disabled.")
    def test_cache_does_not_grow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "numba_cache"
            sizes = [
                self.run_solver(cache_dir, Path(temp_dir) / f"cache_index_sizes{i}.json")
                for i in range(2)
            ]
        self.assertTrue(sizes[0], "No Numba cache files were created.")
        self.assertEqual(
            sizes[1], sizes[0],
            "The Numba cache indexes grew on the second run. "
            "Are the cached functions given arguments that are not the same on every run, "
            "such as jitted functions? "
            "See the 'Numba caching' section of the developer documentation."
        )

    @staticmethod
    def run_solver(cache_dir: Path, output_path: Path) -> dict[str, int]:
        """Solve a bag model bubble in a subprocess and get the number of keys in each Numba cache index."""
        env = {
            **os.environ,
            "NUMBA_CACHE_DIR": str(cache_dir),
            "NUMBA_ENABLE_CACHE": "1",
            "PYTHONPATH": str(REPO_DIR),
        }
        proc = subprocess.run(
            [sys.executable, CACHE_SCRIPT_PATH, output_path],
            capture_output=True, check=False, cwd=REPO_DIR, env=env, text=True
        )
        if proc.returncode or not output_path.is_file():
            raise RuntimeError(
                f"Running \"{CACHE_SCRIPT_PATH}\" failed with the return code {proc.returncode}.\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        with output_path.open() as file:
            return json.load(file)


if __name__ == "__main__":
    unittest.main()
