"""Script for inspecting the Numba cache of a solved bubble.

This is run in a separate process by
:class:`tests.bubble.test_cs2.TestNumbaCache`,
which cannot inspect the Numba cache by itself,
as the caching behaviour that is being tested can be observed only between processes.

Usage: ``python tests/bubble/numba_cache.py <output_path>``,
with ``NUMBA_CACHE_DIR`` and ``NUMBA_ENABLE_CACHE`` set in the environment,
and with the repository directory in ``PYTHONPATH``.
"""

import glob
import json
import os.path
import pickle
import sys

from pttools.bubble.bubble import Bubble
from pttools.models import BagModel


def cache_index_sizes(cache_dir: str) -> dict[str, int]:
    """Get the number of keys in each Numba cache index in the given directory.

    The format of an index file is:
    one pickle holding the Numba version, and a second one holding ``(stamp, overloads)``,
    where ``stamp`` is the modification time and the size of the source file.

    :param cache_dir: directory of the Numba cache
    :return: number of keys by the name of the index file
    """
    sizes: dict[str, int] = {}
    for path in glob.glob(os.path.join(cache_dir, "**", "*.nbi"), recursive=True):
        with open(path, "rb") as file:
            pickle.load(file)
            _, overloads = pickle.loads(file.read())
        sizes[os.path.basename(path)] = len(overloads)
    return sizes


def main(output_path: str) -> None:
    """Solve a bubble and save the sizes of the Numba cache indexes to the given file.

    The results are saved to a file instead of being printed,
    since other libraries write to the standard output as well.
    For example, Colorama, which is initialised by :mod:`pttools.utils.printing`,
    writes an ANSI reset sequence to the standard output when the process exits,
    if the standard output looks like a terminal.

    :param output_path: path to the JSON file to be created
    """
    Bubble(BagModel(a_s=1.1, a_b=1, V_s=1), v_wall=0.5, alpha_n=0.2).solve()
    with open(output_path, "w") as file:
        json.dump(cache_index_sizes(os.environ["NUMBA_CACHE_DIR"]), file)


if __name__ == "__main__":
    main(sys.argv[1])
