"""
Lines of code
=============

Count the lines of code of the PTtools repository and the PTtools library with
`cloc <https://github.com/AlDanial/cloc>`_ using :py:mod:`pttools.docs.cloc`.
Only the files tracked by Git are counted, and the files matching the patterns of ``.clocignore`` are excluded.
"""

import os.path

from pttools.docs.cloc import cloc
from pttools.utils.system import PTTOOLS_DIR

REPO_DIR: str = os.path.dirname(PTTOOLS_DIR)


def main() -> None:
    print("Lines of code in the PTtools repository")
    print(cloc(REPO_DIR))
    print("Lines of code in the PTtools library")
    print(cloc(PTTOOLS_DIR))


if __name__ == "__main__":
    main()
