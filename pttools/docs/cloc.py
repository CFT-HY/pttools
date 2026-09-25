#!/usr/bin/env python3

"""Count the lines of code with `cloc <https://github.com/AlDanial/cloc>`_.

Only the files tracked by Git are counted, and the files matching the patterns
in the ``.clocignore`` files are excluded.
The ``.clocignore`` files have the same syntax and semantics as ``.gitignore`` files:
a ``.clocignore`` file applies to the directory in which it is located and its subdirectories,
and its patterns are relative to that directory.
Therefore, a ``.clocignore`` file in the given directory or in any of its parent directories
within the Git repository is used.
The file list is built with Git instead of the ``--exclude-list-file`` option of cloc,
as that does not support relative paths in older versions of cloc.
https://github.com/AlDanial/cloc/issues/784

Usage: ``python -m pttools.docs.cloc [DIRECTORY] [-- CLOC_OPTIONS...]``

The directory defaults to the current working directory.
The options after ``--`` are passed to cloc instead of the default options ``--by-file-by-lang``.
For example, ``python -m pttools.docs.cloc pttools`` counts the lines of code of the PTtools library,
and ``python -m pttools.docs.cloc`` in the root of the repository counts those of the entire repository.

This can be used also in other projects, such as PTPlot, that have a ``.clocignore`` file.
The script uses only the Python standard library, and can therefore also be run
without installing PTtools with ``python3 path/to/pttools/docs/cloc.py``.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import typing as tp

#: Name of the files that contain the patterns of the files to be excluded
CLOCIGNORE: str = ".clocignore"
#: Options that are passed to cloc if no other options are given
DEFAULT_CLOC_ARGS: tuple[str, ...] = ("--by-file-by-lang",)


def git_ls_files(path: str | os.PathLike[str], *args: str) -> list[str]:
    """List the files tracked by Git in the given directory.

    :param path: the directory
    :param args: additional arguments to ``git ls-files``
    :return: paths of the files relative to the given directory
    """
    # With -z the paths are not quoted, even if they contain special characters.
    output = subprocess.run(
        ["git", "ls-files", "-z", "--cached", *args, "--", "."],
        cwd=path, check=True, stdout=subprocess.PIPE, text=True
    ).stdout
    return [file for file in output.split("\0") if file]


def git_toplevel(path: str | os.PathLike[str]) -> Path:
    """The root directory of the Git repository that contains the given directory."""
    return Path(subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=path, check=True, stdout=subprocess.PIPE, text=True
    ).stdout.strip())


def find_clocignore(path: str | os.PathLike[str]) -> list[Path]:
    """Find the ``.clocignore`` files in the given directory and its parent directories within the Git repository.

    :param path: the directory
    :return: paths of the ``.clocignore`` files, starting from the given directory
    """
    path = Path(path).resolve()
    toplevel = git_toplevel(path).resolve()
    found: list[Path] = []
    for directory in (path, *path.parents):
        candidate = directory / CLOCIGNORE
        if candidate.is_file():
            found.append(candidate)
        if directory == toplevel:
            break
    return found


def counted_files(path: str | os.PathLike[str]) -> list[str]:
    """The files that are tracked by Git and are not excluded by the ``.clocignore`` files.

    The files that have been deleted from the working tree but not from the Git index are skipped,
    as are the Git submodules.

    :param path: the directory
    :return: paths of the files relative to the given directory
    """
    # Git reads the .clocignore files of the given directory, its parent directories and its subdirectories.
    ignored = set(git_ls_files(path, "--ignored", f"--exclude-per-directory={CLOCIGNORE}"))
    return [
        file for file in git_ls_files(path)
        if file not in ignored and (Path(path) / file).is_file()
    ]


def cloc(path: str | os.PathLike[str] | None = None, cloc_args: tp.Sequence[str] = DEFAULT_CLOC_ARGS) -> str:
    """Count the lines of code with cloc.

    :param path: the directory, defaults to the current working directory
    :param cloc_args: options for cloc
    :return: the output of cloc
    :raises FileNotFoundError: if the directory does not exist or cloc or Git is not installed
    :raises subprocess.CalledProcessError: if cloc or Git fails
    """
    directory = Path.cwd() if path is None else Path(path).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} is not a directory.")
    files = counted_files(directory)
    return subprocess.run(
        ["cloc", "--list-file=-", *cloc_args],
        cwd=directory, check=True, input="".join(f"{file}\n" for file in files), stdout=subprocess.PIPE, text=True
    ).stdout


def main(argv: tp.Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # The options after "--" are passed to cloc.
    if "--" in argv:
        i = argv.index("--")
        argv, cloc_args = argv[:i], argv[i+1:]
    else:
        cloc_args = DEFAULT_CLOC_ARGS
    parser = argparse.ArgumentParser(
        usage="%(prog)s [-h] [path] [-- CLOC_OPTIONS...]",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "path", nargs="?", default=None,
        help="the directory to be counted (default: the current working directory)")
    args = parser.parse_args(argv)

    path = Path.cwd() if args.path is None else Path(args.path).resolve()
    try:
        clocignores = find_clocignore(path)
        if clocignores:
            print(f"Counting {path} excluding the patterns of: {', '.join(str(file) for file in clocignores)}")
        else:
            print(f"Counting {path}. No {CLOCIGNORE} files were found.")
        print(cloc(path, cloc_args), end="")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
