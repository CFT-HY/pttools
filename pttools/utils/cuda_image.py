#!/usr/bin/env -S python3 -P

"""Print the current and the latest versions of the CUDA base image of a ``Dockerfile``.

The image tags are of the form ``nvidia/cuda:<CUDA version>-base-ubuntu<Ubuntu version>``,
e.g. ``nvidia/cuda:13.4.2-base-ubuntu26.04``, and both of the version numbers can change.
The current tag is read from the ``Dockerfile``, e.g. from ``ARG CUDA_IMAGE="nvidia/cuda:..."``
or ``FROM nvidia/cuda:...``, and the available tags are fetched from Docker Hub.
Only the tags for Ubuntu LTS releases that are available for all the platforms of the Docker build are considered.

Usage: ``uv run python -m pttools.utils.cuda_image [--all] [--dockerfile Dockerfile]``
or ``./pttools/utils/cuda_image.py [--all] [--dockerfile Dockerfile]``

With ``--all``, all the matching tags are listed instead of only the latest CUDA version for each Ubuntu version.

This can be used also in other projects in which PTtools is installed with uv, such as PTPlot,
by running ``uv run python -m pttools.utils.cuda_image`` in the project directory.
The ``Dockerfile`` is found with :py:func:`find_dockerfile` similarly to the documentation directory
in :py:mod:`pttools.docs.lint`: it is looked for alongside the virtual environment (e.g. ``venv`` or ``.venv``)
in which Python is running, then in the current working directory,
and finally in the Git repository of PTtools, if PTtools is run from a source checkout.
"""

import argparse
import json
import os
import pathlib
import re
import sys
import urllib.request

# This file is run also as a script with the system Python, in which PTtools may not be installed,
# and therefore it does not import anything from PTtools, such as PTTOOLS_DIR or pttools.docs.paths.
# The -P in the shebang prevents adding the directory of this file to sys.path,
# as otherwise e.g. pttools/utils/json.py would shadow the json module of the standard library.
#: The root directory of the Git repository of PTtools, if PTtools is run from a source checkout
REPO_DIR: pathlib.Path = pathlib.Path(__file__).resolve().parents[2]
#: Name of the Dockerfile
DOCKERFILE_NAME: str = "Dockerfile"
#: The name of the base image on Docker Hub
IMAGE: str = "nvidia/cuda"
#: The part of the tag between the CUDA version and the Ubuntu version
TAG_SUFFIX: str = "-base-ubuntu"
#: Regular expression for the tags with the groups for the CUDA and Ubuntu version numbers
TAG_RE: re.Pattern[str] = re.compile(r"^(\d+)\.(\d+)\.(\d+)-base-ubuntu(\d+)\.(\d+)$")
#: Regular expression for the references to the image in the Dockerfile, with the tag as the group
IMAGE_REF_RE: re.Pattern[str] = re.compile(r"\bnvidia/cuda:([\w.-]+)")
#: The platforms (OS, architecture) for which the Docker image is built.
#: These should be the same as the platforms in ``.github/actions/deploy-docker/action.yml``.
PLATFORMS: set[tuple[str, str]] = {("linux", "amd64"), ("linux", "arm64")}
#: The month of the Ubuntu LTS releases, which are YY.04 with an even YY
LTS_MONTH: int = 4

type Version = tuple[int, ...]
# (CUDA version, Ubuntu version)
type Key = tuple[Version, Version]


def parse_tag(tag: str) -> Key | None:
    """Parse an image tag.

    :param tag: image tag, e.g. ``13.4.2-base-ubuntu26.04``
    :return: (CUDA version, Ubuntu version), or None if the tag is not of the expected form
    """
    match = TAG_RE.match(tag)
    if match is None:
        return None
    nums = tuple(int(num) for num in match.groups())
    return nums[:3], nums[3:]


def is_lts(ubuntu: Version) -> bool:
    """Check whether an Ubuntu version is an LTS release.

    :param ubuntu: Ubuntu version, e.g. (26, 4)
    :return: whether the version is an LTS release
    """
    return ubuntu[1] == LTS_MONTH and ubuntu[0] % 2 == 0


def fmt_cuda(version: Version) -> str:
    """Format a CUDA version as a string.

    :param version: CUDA version, e.g. (13, 4, 2)
    :return: the version as a string, e.g. ``13.4.2``
    """
    return ".".join(str(num) for num in version)


def fmt_ubuntu(version: Version) -> str:
    """Format an Ubuntu version as a string.

    :param version: Ubuntu version, e.g. (26, 4)
    :return: the version as a string, e.g. ``26.04``
    """
    return f"{version[0]}.{version[1]:02d}"


def ubuntu_first(key: Key) -> Key:
    """Sort key for ordering by the Ubuntu version first and then by the CUDA version.

    :param key: (CUDA version, Ubuntu version)
    :return: (Ubuntu version, CUDA version)
    """
    return key[1], key[0]


def env_dir() -> pathlib.Path | None:
    """Path of the Python virtual environment in which Python is running.

    This is the same as :py:func:`pttools.docs.paths.env_dir`, which cannot be imported here.

    :return: path of the environment, or None if not running in a virtual environment
    """
    if sys.prefix != sys.base_prefix:
        return pathlib.Path(sys.prefix).absolute()
    return None


def find_dockerfile(cwd: str | os.PathLike[str] | None = None) -> pathlib.Path | None:
    """Find the ``Dockerfile`` of the project.

    The following locations are checked in order, and the first existing file is returned:

    1. The ``Dockerfile`` alongside the virtual environment (e.g. ``venv`` or ``.venv``)
       in which Python is running (see :py:func:`env_dir`).
       This is the case when PTtools is installed as a package in the environment of another project,
       such as PTPlot, and when PTtools itself is run with ``uv run``.
    2. The ``Dockerfile`` in the current working directory.
    3. The ``Dockerfile`` of the PTtools repository, if PTtools is run from a source checkout.

    :param cwd: the directory to use as the current working directory, or None for the actual one
    :return: path of the ``Dockerfile``, or None if not found
    """
    cwd_path = pathlib.Path.cwd() if cwd is None else pathlib.Path(cwd).absolute()
    candidates: list[pathlib.Path] = []
    if (env := env_dir()) is not None:
        candidates.append(env.parent / DOCKERFILE_NAME)
    candidates.append(cwd_path / DOCKERFILE_NAME)
    candidates.append(REPO_DIR / DOCKERFILE_NAME)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def read_current(dockerfile: str | os.PathLike[str]) -> tuple[str, Key]:
    """Read the current image tag from the ``Dockerfile``.

    The comment lines are ignored. All the references to the image must have the same tag.

    :param dockerfile: path of the ``Dockerfile``
    :return: the tag and its (CUDA version, Ubuntu version)
    :raises ValueError: if the tag is not found, there are several different tags,
        or the tag is not of the expected form
    """
    lines = pathlib.Path(dockerfile).read_text().splitlines()
    tags = sorted({
        match.group(1)
        for line in lines if not line.lstrip().startswith("#")
        for match in IMAGE_REF_RE.finditer(line)
    })
    if not tags:
        raise ValueError(f"No references to {IMAGE}:<tag> found in {dockerfile}")
    if len(tags) > 1:
        raise ValueError(f"Several different tags of {IMAGE} found in {dockerfile}: {', '.join(tags)}")
    tag = tags[0]
    parsed = parse_tag(tag)
    if parsed is None:
        raise ValueError(f"Unexpected tag format in {dockerfile}: {tag}")
    return tag, parsed


def fetch_tags() -> list[dict]:
    """Fetch all the tags of the image that contain :py:data:`TAG_SUFFIX` from Docker Hub.

    :return: the tags as returned by the Docker Hub API
    """
    url: str | None = f"https://hub.docker.com/v2/repositories/{IMAGE}/tags?page_size=100&name={TAG_SUFFIX}"
    tags = []
    while url:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.load(response)
        tags.extend(data["results"])
        url = data.get("next")
    return tags


def find_candidates(min_ubuntu: Version) -> tuple[dict[Key, tuple[str, str]], list[str]]:
    """Find the usable tags.

    Only the Ubuntu LTS releases that have images for all the platforms of :py:data:`PLATFORMS` are included.

    :param min_ubuntu: minimum Ubuntu version for listing the tags that lack some of the platforms
    :return: ``{(CUDA version, Ubuntu version): (tag, last updated)}``
        and the tags with Ubuntu >= ``min_ubuntu`` that lack some of the platforms
    """
    candidates: dict[Key, tuple[str, str]] = {}
    skipped: list[str] = []
    for tag in fetch_tags():
        parsed = parse_tag(tag["name"])
        if parsed is None or not is_lts(parsed[1]):
            continue
        platforms = {(image["os"], image["architecture"]) for image in tag.get("images", [])}
        if not platforms >= PLATFORMS:
            if parsed[1] >= min_ubuntu:
                skipped.append(tag["name"])
            continue
        candidates[parsed] = (tag["name"], tag["last_updated"][:10])
    return candidates, skipped


def latest_per_ubuntu(candidates: dict[Key, tuple[str, str]]) -> list[Key]:
    """Find the latest CUDA version for each Ubuntu version.

    :param candidates: the usable tags from :py:func:`find_candidates`
    :return: (CUDA version, Ubuntu version) pairs, the newest Ubuntu version first
    """
    latest: dict[Version, Version] = {}
    for cuda, ubuntu in candidates:
        latest[ubuntu] = max(cuda, latest.get(ubuntu, cuda))
    return sorted(((cuda, ubuntu) for ubuntu, cuda in latest.items()), key=ubuntu_first, reverse=True)


def main() -> int:
    """Print the current and the latest image tags.

    :return: exit code
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="list all the matching tags, not only the latest ones")
    parser.add_argument(
        "--dockerfile", type=pathlib.Path,
        help="path of the Dockerfile (default: found alongside the virtual environment or in the current directory)"
    )
    args = parser.parse_args()

    dockerfile: pathlib.Path | None = args.dockerfile if args.dockerfile is not None else find_dockerfile()
    if dockerfile is None:
        print(
            "The Dockerfile was not found alongside the virtual environment "
            f"or in the current working directory {pathlib.Path.cwd()}. Specify it with --dockerfile.",
            file=sys.stderr
        )
        return 1
    try:
        current_tag, current = read_current(dockerfile)
    except (OSError, ValueError) as err:
        print(err, file=sys.stderr)
        return 1
    print(f"Dockerfile: {dockerfile}")
    candidates, skipped = find_candidates(min_ubuntu=current[1])
    if not candidates:
        print("No matching tags found on Docker Hub", file=sys.stderr)
        return 1

    platforms_str = ", ".join(f"{os_name}/{arch}" for os_name, arch in sorted(PLATFORMS))
    print(f"Tags of {IMAGE} for Ubuntu LTS and {platforms_str}:")
    rows = sorted(candidates, reverse=True) if args.all else latest_per_ubuntu(candidates)
    for key in rows:
        name, updated = candidates[key]
        note = " <- current" if key == current else ""
        print(f"  {name:28} updated {updated}{note}")
    if skipped:
        print(f"Skipped tags that lack some of the platforms: {', '.join(sorted(skipped))}")

    # The newest Ubuntu version, and the newest CUDA version for it.
    latest = max(candidates, key=ubuntu_first)
    print()
    print(f"Current: {IMAGE}:{current_tag}")
    print(f"Latest:  {IMAGE}:{candidates[latest][0]}")
    if current not in candidates:
        print("Warning: the current tag was not found or lacks some of the platforms.")
    if latest == current:
        print("The base image is up to date.")
        return 0
    changes = []
    if latest[0] != current[0]:
        changes.append(f"CUDA {fmt_cuda(current[0])} -> {fmt_cuda(latest[0])}")
    if latest[1] != current[1]:
        changes.append(f"Ubuntu {fmt_ubuntu(current[1])} -> {fmt_ubuntu(latest[1])}")
    print(f"Update available: {', '.join(changes)}")
    if latest[0] < current[0]:
        print("Warning: the latest Ubuntu version has an older CUDA version than the current image.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
