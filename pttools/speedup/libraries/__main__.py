"""Check that the native libraries for Numba can be loaded.

Usage: ``python -m pttools.speedup.libraries [svml] [tbb]``

This prints the state of the given libraries, or all of them if none are given.
The exit code is 0 if all the required libraries were loaded, and 1 otherwise.
A library is required if one of the PyPI packages that provide it is installed,
or if the CPU architecture is x86-64, for which the ``performance`` extra of PTtools installs all of them.
On other CPU architectures, a note is printed for the libraries that are not required and could not be loaded.
"""

import argparse
import sys

from pttools.speedup.libraries import LIBRARIES, check_libraries


def main(args: list[str] | None = None) -> int:
    """Print the state of the native libraries.

    :param args: command-line arguments. If not given, :data:`sys.argv` is used.
    :return: exit code, 0 if all the required libraries were loaded, and 1 otherwise
    """
    parser = argparse.ArgumentParser(
        prog="python -m pttools.speedup.libraries",
        description="Check that the native libraries for Numba can be loaded."
    )
    parser.add_argument(
        "libraries", nargs="*", choices=list(LIBRARIES),
        help="libraries to check (default: all)"
    )
    return check_libraries(parser.parse_args(args).libraries)


if __name__ == "__main__":
    sys.exit(main())
