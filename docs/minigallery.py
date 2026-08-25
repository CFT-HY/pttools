"""Utilities for configuring the mini-galleries of Sphinx-Gallery

https://sphinx-gallery.github.io/stable/configuration.html#add-mini-galleries-for-api-documentation
"""

import re
import typing as tp

from docutils import nodes
from sphinx.application import Sphinx

from docs.utils import backreference_names

#: Object types that get an automatic mini-gallery.
#: These are the values of the "what" argument of the "autodoc-process-docstring" event.
MINIGALLERY_TYPES: frozenset[str] = frozenset({
    "attribute", "class", "data", "exception", "function", "method", "module", "property"
})
#: Regex for finding minigallery directives that have been added to a docstring manually
MINIGALLERY_REGEX: re.Pattern[str] = re.compile(r"^\s*\.\.\s+minigallery::(.*)$")
#: CSS class with which the automatically added mini-galleries are marked
MINIGALLERY_CLASS: str = "sphx-glr-minigallery-auto"


def add_minigalleries(  # pylint: disable=too-many-positional-arguments, unused-argument
        app: Sphinx, what: str, name: str, obj: tp.Any, options: tp.Any, lines: list[str]) -> None:
    """Add a mini-gallery of the examples that use the object being documented.

    If the docstring already contains a minigallery directive,
    e.g. for pointing to an example that does not refer to the object by name,
    then the names are added to that directive so that all the examples are shown in a single gallery.
    """
    if what not in MINIGALLERY_TYPES:
        return
    names = backreference_names(app, name, obj)
    if not names:
        return
    for i, line in enumerate(lines):
        match = MINIGALLERY_REGEX.match(line)
        if match:
            new_names = [ref for ref in names if ref not in match.group(1).split()]
            if new_names:
                lines[i] = f"{line.rstrip()} {' '.join(new_names)}"
            return
    # The mini-gallery is wrapped in a container so that the duplicates,
    # which are created when a class has both a class and an __init__ docstring, can be removed later.
    lines += [
        "",
        f".. container:: {MINIGALLERY_CLASS}",
        "",
        f"   .. rubric:: Examples using ``{name}``",
        "",
        f"   .. minigallery:: {' '.join(names)}",
        ""
    ]


def remove_duplicate_minigalleries(  # pylint: disable=unused-argument
        app: Sphinx, domain: str, objtype: str, contentnode: nodes.Element) -> None:
    """Remove all but the last of the automatic mini-galleries of an object.

    Autodoc processes the class and __init__ docstrings of a class separately when
    autoclass_content is "both", which would result in two mini-galleries.
    Keeping the last one places the mini-gallery after the parameter list.
    Only the direct children are inspected, so that the mini-galleries of
    the methods and attributes of a class are not affected.
    """
    galleries = [
        child for child in contentnode.children
        if isinstance(child, nodes.container) and MINIGALLERY_CLASS in child["classes"]
    ]
    for gallery in galleries[:-1]:
        contentnode.remove(gallery)
