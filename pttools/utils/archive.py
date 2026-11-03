"""Archiving and compression of files."""

import collections.abc
import concurrent.futures
import fnmatch
import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
from typing import IO, cast

from pttools.logging import log_console
from pttools.utils.files import dir_size, format_size
from pttools.utils.system import physical_cpu_count
from pttools.utils.time import format_duration

__all__ = ["OPTIPNG_ARGS", "TAR_XZ_SUFFIX", "archive", "optimize_pngs"]

#: The arguments of optipng for the lossless optimization of the PNG files.
#: ``-o7`` is the highest optimization level, ``-preserve`` keeps the modification times,
#: and the metadata, such as the software that created the figure, is kept.
OPTIPNG_ARGS: tuple[str, ...] = ("-o7", "-preserve", "-quiet")
#: The filename extension of xz-compressed tar archives
TAR_XZ_SUFFIX: str = ".tar.xz"


def archive(archive_path: Path, paths: dict[Path, str], exclude: collections.abc.Iterable[str] = ()) -> Path:
    """Compress directories and files to an xz-compressed tar archive.

    The tar stream is created with :mod:`tarfile` and compressed with the ``xz`` command using all CPU cores,
    as the :mod:`lzma` module of the standard library compresses with only one core.
    If there is already an archive with the same name, a suffix _2, _3 etc. is added to the filename,
    so that the existing archive is not overwritten.
    A message is logged after the archive has been created.

    :param archive_path: the path of the archive, ending with :data:`TAR_XZ_SUFFIX`
    :param paths: the directories and files to compress, and their names at the root of the archive
    :param exclude: the wildcards of the filenames to leave out of the archive, e.g. ``*.h5``
    :return: the path of the created archive
    :raises FileNotFoundError: if a directory or a file does not exist, or if ``xz`` is not installed
    :raises subprocess.CalledProcessError: if ``xz`` fails
    """
    for item in paths:
        if not item.exists():
            raise FileNotFoundError(f"{item} does not exist.")
    if shutil.which("xz") is None:
        raise FileNotFoundError("xz was not found in PATH.")
    exclude = tuple(exclude)
    stem = archive_path.name.removesuffix(TAR_XZ_SUFFIX)
    path = archive_path
    i = 2
    while path.exists():
        path = archive_path.with_name(f"{stem}_{i}{TAR_XZ_SUFFIX}")
        i += 1
    path.parent.mkdir(parents=True, exist_ok=True)

    def exclude_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        name = info.name.rsplit("/", 1)[-1]
        return None if any(fnmatch.fnmatchcase(name, pattern) for pattern in exclude) else info

    # -T0 uses as many threads as there are CPU cores.
    # The compressed output is written to the archive file, which must not exist yet.
    command = ["xz", "-9", "-T0", "--stdout"]
    try:
        with path.open("xb") as file, subprocess.Popen(command, stdin=subprocess.PIPE, stdout=file) as xz:
            stdin = cast(IO[bytes], xz.stdin)
            try:
                with tarfile.open(fileobj=stdin, mode="w|") as tar:
                    for item, name in paths.items():
                        tar.add(item, arcname=name, filter=exclude_filter)
            finally:
                stdin.close()
        if xz.returncode:
            raise subprocess.CalledProcessError(xz.returncode, command)
    except BaseException:
        # A partial archive is removed, so that it's not mistaken for a complete one.
        path.unlink(missing_ok=True)
        raise
    log_console(f"Created the archive {path} of {', '.join(f'{item} as {name}' for item, name in paths.items())}.")
    return path


def optimize_pngs(
        dirs: collections.abc.Iterable[Path],
        workers: int | None = None,
        args: collections.abc.Sequence[str] = OPTIPNG_ARGS) -> None:
    """Optimize the PNG files in directories losslessly with optipng.

    optipng uses only one CPU core, so the files are optimized in parallel, one optipng process per physical CPU core,
    starting from the largest files. Using also the SMT threads would slow down the optimization of the largest file,
    which usually determines the total time.
    Identical files, such as the copies of the same figure in the HTML and EPUB builds of the documentation,
    are optimized only once, and the optimized file is copied over the others.
    The total sizes of the directories and of the PNG files before and after the optimization,
    and the time spent, are logged.
    If optipng is not installed or fails for a file, it's logged, and the files are left as they are.

    :param dirs: the directories, in which the PNG files are searched recursively
    :param workers: the number of parallel optipng processes,
        by default :func:`pttools.utils.system.physical_cpu_count`
    :param args: the arguments of optipng, see :data:`OPTIPNG_ARGS`
    """
    optipng = shutil.which("optipng")
    if optipng is None:
        log_console("optipng was not found in PATH. The PNG files are archived without optimization.")
        return
    dirs = [directory for directory in dirs if directory.is_dir()]
    start = time.perf_counter()
    dir_sizes_before = [dir_size(directory) for directory in dirs]
    groups: dict[str, list[Path]] = {}
    for directory in dirs:
        for path in sorted(directory.rglob("*.png")):
            if path.is_file() and not path.is_symlink():
                groups.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(), []).append(path)
    files = [path for paths in groups.values() for path in paths]
    png_size_before = sum(path.stat().st_size for path in files)

    def optimize(paths: list[Path]) -> str | None:
        result = subprocess.run(
            [optipng, *args, "--", str(paths[0])], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, check=False,
        )
        if result.returncode:
            return f"{paths[0]} (exit code {result.returncode}): {result.stdout.strip()}"
        for path in paths[1:]:
            shutil.copy2(paths[0], path)
        return None

    # The largest files take the longest, so they are started first to keep all the cores busy until the end.
    jobs = sorted(groups.values(), key=lambda paths: paths[0].stat().st_size, reverse=True)
    if workers is None:
        workers = physical_cpu_count()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        errors = [error for error in executor.map(optimize, jobs) if error is not None]
    for error in errors:
        log_console(f"optipng failed for {error}")

    png_size_after = sum(path.stat().st_size for path in files)
    sizes = ", ".join(
        f"{directory} {format_size(before)} -> {format_size(dir_size(directory))}"
        for directory, before in zip(dirs, dir_sizes_before, strict=True)
    )
    reduction = f" ({1 - png_size_after / png_size_before:.1%} smaller)" if png_size_before else ""
    log_console(
        f"Optimized {len(files)} PNG files ({len(groups)} unique) with optipng {' '.join(args)} "
        f"using {workers} parallel processes in {format_duration(time.perf_counter() - start)}. "
        f"PNG files: {format_size(png_size_before)} -> {format_size(png_size_after)}{reduction}. "
        f"Directories: {sizes}."
    )
