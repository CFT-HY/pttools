"""
Lines of code
=============

Count the lines of code of the project repository and the PTtools library with
`cloc <https://github.com/AlDanial/cloc>`_ using :py:mod:`pttools.docs.cloc`.
The files are grouped by directory with :py:func:`pttools.docs.cloc.cloc_compact` to keep the lines short.
Only the files tracked by Git are counted, and the files matching the patterns of ``.clocignore`` are excluded.
The project repository is given by :py:data:`pttools.utils.system.PROJECT_DIR`,
which is the PTtools repository when PTtools is run from its source repository.
"""

from pttools.docs.cloc import cloc_compact
from pttools.utils.system import PROJECT_DIR, PTTOOLS_DIR


def main() -> None:
    """Print the lines of code of the project repository and the PTtools library."""
    if PROJECT_DIR is None:
        print("The project repository was not found, as PTtools is not installed in a virtual environment.")
    else:
        print(f"Lines of code in the project repository {PROJECT_DIR.name}")
        print(cloc_compact(PROJECT_DIR))
    print("Lines of code in the PTtools library")
    print(cloc_compact(PTTOOLS_DIR))


if __name__ == "__main__":
    main()
