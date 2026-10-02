#!/usr/bin/env -S python3 -P

"""Print the current and the latest versions of the CUDA base image of the ``Dockerfile`` of PTtools.

The image tags are of the form ``nvidia/cuda:<CUDA version>-base-ubuntu<Ubuntu version>``,
e.g. ``nvidia/cuda:13.4.2-base-ubuntu26.04``, and both of the version numbers can change.
The tags are fetched from Docker Hub.
Only the tags for Ubuntu LTS releases that are available for all the platforms of the Docker build are considered.
This requires the Git repository of PTtools, as the ``Dockerfile`` is not included in the package.

Usage: ``uv run python -m pttools.utils.cuda_image [--all]`` or ``./pttools/utils/cuda_image.py [--all]``

With ``--all``, all the matching tags are listed instead of only the latest CUDA version for each Ubuntu version.
"""

import argparse
import json
import pathlib
import re
import sys
import urllib.request

# This file is run also as a script, and therefore it cannot use relative imports such as PTTOOLS_DIR.
# The -P in the shebang prevents adding the directory of this file to sys.path,
# as otherwise e.g. pttools/utils/json.py would shadow the json module of the standard library.
#: The root directory of the Git repository of PTtools
REPO_DIR: pathlib.Path = pathlib.Path(__file__).resolve().parents[2]
#: The Dockerfile of PTtools
DOCKERFILE: pathlib.Path = REPO_DIR / "Dockerfile"
#: The name of the base image on Docker Hub
IMAGE: str = "nvidia/cuda"
#: The part of the tag between the CUDA version and the Ubuntu version
TAG_SUFFIX: str = "-base-ubuntu"
#: Regular expression for the tags with the groups for the CUDA and Ubuntu version numbers
TAG_RE: re.Pattern[str] = re.compile(r"^(\d+)\.(\d+)\.(\d+)-base-ubuntu(\d+)\.(\d+)$")
#: Regular expression for the line of the Dockerfile that defines the base image
ARG_RE: re.Pattern[str] = re.compile(r'^ARG CUDA_IMAGE="nvidia/cuda:([^"]+)"$', re.MULTILINE)
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


def fmt(version: Version) -> str:
    """Format a version as a string.

    :param version: version number, e.g. (13, 4, 2)
    :return: the version as a string, e.g. ``13.4.2``
    """
    return ".".join(str(num) for num in version)


def ubuntu_first(key: Key) -> Key:
    """Sort key for ordering by the Ubuntu version first and then by the CUDA version.

    :param key: (CUDA version, Ubuntu version)
    :return: (Ubuntu version, CUDA version)
    """
    return key[1], key[0]


def read_current() -> tuple[str, Key]:
    """Read the current image tag from the ``Dockerfile``.

    :return: the tag and its (CUDA version, Ubuntu version)
    :raises ValueError: if the tag is not found or is not of the expected form
    """
    match = ARG_RE.search(DOCKERFILE.read_text())
    if match is None:
        raise ValueError(f'ARG CUDA_IMAGE="{IMAGE}:..." not found in {DOCKERFILE}')
    tag = match.group(1)
    parsed = parse_tag(tag)
    if parsed is None:
        raise ValueError(f"Unexpected tag format in the Dockerfile: {tag}")
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
    args = parser.parse_args()

    current_tag, current = read_current()
    candidates, skipped = find_candidates(min_ubuntu=current[1])
    if not candidates:
        print("No matching tags found on Docker Hub", file=sys.stderr)
        return 1

    platforms_str = ", ".join(f"{os}/{arch}" for os, arch in sorted(PLATFORMS))
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
        changes.append(f"CUDA {fmt(current[0])} -> {fmt(latest[0])}")
    if latest[1] != current[1]:
        changes.append(f"Ubuntu {fmt(current[1])} -> {fmt(latest[1])}")
    print(f"Update available: {', '.join(changes)}")
    if latest[0] < current[0]:
        print("Warning: the latest Ubuntu version has an older CUDA version than the current image.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
