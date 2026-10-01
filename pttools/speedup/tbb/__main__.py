"""Check that a TBB library compatible with Numba can be loaded.

Usage: ``python -m pttools.speedup.tbb``

This prints the TBB runtime interface version of the library that was loaded when :py:mod:`pttools.speedup.tbb`
was imported, or None if no library was found.
The exit code is 0 if a library compatible with Numba was loaded, and 1 otherwise.
However, the ``tbb`` package from PyPI is available only for x86-64.
Therefore, on other CPU architectures, the exit code is always 0, and a note is printed if no compatible library
was loaded.
"""

import platform
import sys

from pttools.speedup.tbb import TBB_MIN_VERSION, TBB_VERSION
from pttools.utils.system import IS_X86_64


def main() -> int:
    """Print the version of the loaded TBB library.

    :return: exit code, 0 if a TBB library compatible with Numba was loaded or the CPU architecture is not x86-64,
        and 1 otherwise
    """
    print("TBB version:", TBB_VERSION)
    if TBB_VERSION is not None and TBB_VERSION >= TBB_MIN_VERSION:
        return 0
    if not IS_X86_64:
        print(
            f"Note: TBB is not available for the current CPU architecture ({platform.machine()}) "
            "from the tbb package of PyPI, so it is not required."
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
