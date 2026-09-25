"""
Lines of code
=============

Count the lines of code of the PTtools repository and the PTtools library with
`cloc <https://github.com/AlDanial/cloc>`_ using :py:mod:`pttools.docs.cloc`.
The files are grouped by directory with :py:func:`pttools.docs.cloc.cloc_compact` to keep the lines short.
Only the files tracked by Git are counted, and the files matching the patterns of ``.clocignore`` are excluded.
"""

from pathlib import Path

from pttools.docs.cloc import cloc_compact
from pttools.utils.system import PTTOOLS_DIR

REPO_DIR: Path = PTTOOLS_DIR.parent


def main() -> None:
    print("Lines of code in the PTtools repository")
    print(cloc_compact(REPO_DIR))
    print("Lines of code in the PTtools library")
    print(cloc_compact(PTTOOLS_DIR))


if __name__ == "__main__":
    main()
