"""Utilities for Sphinx documentation"""

import functools
import json
import logging
import os.path
import sys
from threading import Lock
import typing as tp

from sphinx.application import Sphinx

logger = logging.getLogger(__name__)

#: Sentinel for objects that could not be resolved
MISSING = object()
BACKREFERENCES_WARNING_LOCK = Lock()


@functools.cache
def get_backreferences(path: str) -> dict[str, list]:
    """Load the Sphinx-Gallery backreferences, which tell which examples use each object.

    Sphinx-Gallery writes these when it generates the galleries,
    which happens before any docstring is processed, and therefore they are available here.
    They are loaded only once and then cached.
    """
    try:
        with open(path, "r", encoding="utf-8") as json_file:
            backrefs = json.load(json_file)
    except (OSError, json.JSONDecodeError):
        backrefs = {}
    if (not backrefs) and BACKREFERENCES_WARNING_LOCK.acquire(blocking=False):  # pylint: disable=consider-using-with
        logger.warning(
            "Sphinx-Gallery backreferences were not found at \"%s\", so no mini-galleries are created. "
            "Sphinx-Gallery collects the backreferences only from the examples that it (re)generates, "
            "so a full rebuild with \"make clean\" is needed for the mini-galleries to appear.",
            path
        )
    return backrefs


def resolve_object(module: str, attrs: list[str]) -> tp.Any:
    """Resolve the object that the given module and attribute names point to"""
    obj = sys.modules.get(module, MISSING)
    for attr in attrs:
        obj = getattr(obj, attr, MISSING)
        if obj is MISSING:
            break
    return obj


def is_same_object(obj: tp.Any, other: tp.Any) -> bool:
    """Check whether two references point to the same object.

    Accessing a class method creates a new bound method object every time,
    so the underlying functions have to be compared instead.
    """
    return obj is other or getattr(obj, "__func__", obj) is getattr(other, "__func__", other)


def backreference_names(app: Sphinx, name: str, obj: tp.Any) -> list[str]:
    """Find the names by which the given object is referred to in the examples.

    Sphinx-Gallery records an object by the name with which the example refers to it,
    e.g. "pttools.BagModel" or "pttools.models.BagModel" instead of "pttools.models.bag.BagModel",
    and the same object can therefore have backreferences under several names.
    """
    backrefs = get_backreferences(os.path.join(
        app.srcdir, app.config.sphinx_gallery_conf["backreferences_dir"], "backreferences_all.json"))
    if not backrefs:
        return []
    parts = name.split(".")
    # Find the longest prefix of the name that is a module, so that the rest of the parts are attribute names.
    for i_module in range(len(parts), 0, -1):
        if ".".join(parts[:i_module]) in sys.modules:
            break
    else:
        return []
    attrs = parts[i_module:]
    # A module can be referred to by its full name only.
    if not attrs:
        return [name] if name in backrefs else []
    # The object may also be accessible from the parent packages of its module.
    return [
        candidate
        for i in range(1, i_module + 1)
        if (candidate := ".".join(parts[:i] + attrs)) in backrefs
        and is_same_object(resolve_object(".".join(parts[:i]), attrs), obj)
    ]
