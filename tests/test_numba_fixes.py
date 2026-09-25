"""Unit tests for the workarounds for Numba bugs."""

import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

from pttools.speedup.options import NUMBA_DISABLE_JIT
from tests.utils import REPO_DIR

#: Script that compiles a cached function, which calls a cached parallel function
SCRIPT_PATH: Path = Path(__file__).resolve().parent / "numba_reload_init.py"


@unittest.skipIf(NUMBA_DISABLE_JIT, "Nothing is compiled when jitting is disabled.")
class TestReloadInit(unittest.TestCase):
    """Test that a function compiled against a parallel function loaded from the cache can be loaded from the cache.

    See :mod:`pttools.speedup.numba_fixes` for the details.
    """

    def test_reload_init(self) -> None:
        with tempfile.TemporaryDirectory() as cache_dir:
            # Cache the parallel function.
            self.run_script(cache_dir, "callee")
            # Load the parallel function from the cache, and compile and cache the caller against it.
            self.run_script(cache_dir, "caller")
            reload_init = self.reload_init(cache_dir, "caller")
            self.assertTrue(
                reload_init,
                "The cached caller does not have the reload_init of the cached parallel function. "
                "Loading it in a new process would crash, as the Numba threading layer would not be launched."
            )
            # Load the caller from the cache in a new process, in which the threading layer has not been launched.
            self.run_script(cache_dir, "caller")

    @staticmethod
    def run_script(cache_dir: str, mode: str) -> None:
        env = {
            **os.environ,
            "NUMBA_CACHE_DIR": cache_dir,
            "PYTHONPATH": str(REPO_DIR),
        }
        proc = subprocess.run(
            [sys.executable, SCRIPT_PATH, mode],
            capture_output=True, check=False, cwd=REPO_DIR, env=env, text=True
        )
        if proc.returncode:
            raise RuntimeError(
                f"Running \"{SCRIPT_PATH} {mode}\" failed with the return code {proc.returncode}.\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )

    @staticmethod
    def reload_init(cache_dir: str, func_name: str) -> list:
        """Get the reload_init of the cached function with the given name.

        The data file of a cached function is a pickle of the tuple returned by
        ``numba.core.compiler.CompileResult._reduce()``, and reload_init is its eighth element.
        """
        paths = list(Path(cache_dir).rglob(f"*.{func_name}-*.nbc"))
        if len(paths) != 1:
            raise FileNotFoundError(f"Expected exactly one cache data file for {func_name}, found: {paths}")
        with paths[0].open("rb") as file:
            payload = pickle.load(file)
        return payload[7]


if __name__ == "__main__":
    unittest.main()
