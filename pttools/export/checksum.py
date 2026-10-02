"""SHA-256 checksums for verifying the integrity of the exported files.

The checksum files are compatible with the ``sha256sum`` command-line utility,
and can therefore also be verified with ``sha256sum --check FILE.sha256``.
"""

import hashlib
import logging
import os
from pathlib import Path
import time

__all__ = [
    "CHECKSUM_SUFFIX",
    "ChecksumError",
    "checksum_path",
    "compute_checksum",
    "verify_checksum",
    "write_checksum",
]

logger: logging.Logger = logging.getLogger(__name__)

#: Suffix of the checksum files
CHECKSUM_SUFFIX: str = ".sha256"


class ChecksumError(ValueError):
    """The checksum of a file does not match its checksum file."""


def checksum_path(path: str | os.PathLike[str]) -> Path:
    """Path of the checksum file corresponding to the given file."""
    path = Path(path)
    return path.with_name(path.name + CHECKSUM_SUFFIX)


def compute_checksum(path: str | os.PathLike[str]) -> str:
    """Compute the SHA-256 checksum of a file.

    The file is read in blocks with :py:func:`hashlib.file_digest`,
    which avoids loading the entire file to memory and releases the GIL during the hashing.
    On CPUs with SHA hardware acceleration the speed is typically limited by the storage.

    :param path: path of the file
    :return: the checksum as a hexadecimal string
    """
    start_time = time.perf_counter()
    with Path(path).open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    logger.debug("Computed the checksum of %s in %.3f s", path, time.perf_counter() - start_time)
    return digest


def write_checksum(path: str | os.PathLike[str]) -> Path:
    """Compute the SHA-256 checksum of a file and save it next to the file in the ``sha256sum`` format.

    :param path: path of the file
    :return: path of the checksum file
    """
    path = Path(path)
    digest = compute_checksum(path)
    sum_path = checksum_path(path)
    # The asterisk denotes binary mode in the sha256sum format.
    sum_path.write_text(f"{digest} *{path.name}\n", encoding="utf-8")
    return sum_path


def verify_checksum(path: str | os.PathLike[str], raise_error: bool = False) -> bool:
    """Verify a file against its checksum file.

    :param path: path of the file, not of the checksum file
    :param raise_error: whether to raise an error if the checksums do not match
    :return: whether the checksums match
    :raises FileNotFoundError: if the checksum file does not exist
    :raises ChecksumError: if the checksums do not match and raise_error is True
    """
    sum_path = checksum_path(path)
    content = sum_path.read_text(encoding="utf-8").split()
    if not content:
        raise ChecksumError(f"The checksum file is empty: {sum_path}")
    expected = content[0].lower()
    actual = compute_checksum(path)
    if actual != expected:
        msg = f"The checksum of {path} does not match. Expected: {expected}, got: {actual}"
        if raise_error:
            raise ChecksumError(msg)
        logger.error(msg)
        return False
    return True
