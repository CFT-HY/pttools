r"""Documentation of the attributes from the descriptions of the exportable fields.

The exportable fields of the PTtools classes have descriptions,
which are stored in the exported files along with the data.
Many of these fields are plain attributes that are set in ``__init__``,
and Sphinx autodoc would document those only if they had ``#:`` comments.
To avoid writing the same descriptions twice,
:py:func:`add_field_docs` gives the field descriptions to autodoc as if they were ``#:`` comments.
Therefore, the resulting documentation is the same as with the comments,
including the type annotations of the attributes.

An existing ``#:`` comment or docstring of an attribute takes precedence over the field description.
Properties and methods are documented by their own docstrings.

Usage in ``docs/conf.py``:

.. code-block:: python

    def setup(app):
        app.connect("builder-inited", add_field_docs)
        app.connect("autodoc-process-docstring", note_field_dependencies)
"""

import functools
import importlib
import inspect
import logging
import typing as tp

from pttools.utils.fields import Extractable, Field

if tp.TYPE_CHECKING:
    from sphinx.application import Sphinx

__all__ = [
    "FIELD_DEFINITION_MODULES",
    "FIELD_MODULES",
    "add_field_docs",
    "attribute_name",
    "field_attribute_docs",
    "format_description",
    "note_field_dependencies",
]

logger: logging.Logger = logging.getLogger(__name__)

#: The modules that define the classes with exportable fields.
#: These are imported so that all the subclasses of :py:class:`pttools.utils.fields.Extractable` are found.
FIELD_MODULES: tuple[str, ...] = ("pttools.bubble", "pttools.models", "pttools.omgw0", "pttools.ssm")

#: The modules that contain the field definitions.
#: The documentation of the classes with exportable fields depends on these.
FIELD_DEFINITION_MODULES: tuple[str, ...] = (
    "pttools.bubble.export",
    "pttools.models.export",
    "pttools.omgw0.export",
    "pttools.ssm.export",
)

#: Key of an attribute: the module name, the qualified name of the class, and the name of the attribute
type AttributeKey = tuple[str, str, str]


def attribute_name(field: Field) -> str | None:
    """Name of the attribute from which the value of a field is read directly.

    :param field: the field
    :return: the attribute name, or None if the value is computed by a function or a method call,
        or if it's read from an attribute of another object, e.g. ``bubble.v_wall``
    """
    if field.call or field.index is not None or callable(field.getter):
        return None
    name = field.name if field.getter is None else field.getter
    return None if "." in name else name


def format_description(description: str) -> str:
    r"""Format a field description as the documentation of an attribute.

    The descriptions start with a lowercase letter or a symbol, e.g. ``"$v_\text{wall}$, wall speed"``,
    and therefore the first letter is capitalised.

    :param description: the field description
    :return: the documentation
    """
    if description[:1].islower():
        return description[0].upper() + description[1:]
    return description


def _subclasses(cls: type) -> list[type]:
    """All the subclasses of a class recursively, sorted by their names for a deterministic order."""
    found: dict[str, type] = {}
    stack = [cls]
    while stack:
        for sub in stack.pop().__subclasses__():
            name = f"{sub.__module__}.{sub.__qualname__}"
            if name not in found:
                found[name] = sub
                stack.append(sub)
    return [found[name] for name in sorted(found)]


def _defined_in(cls: type, attr: str) -> bool:
    """Whether the source code of the class assigns the attribute, e.g. ``self.attr = ...`` in ``__init__``."""
    from sphinx.errors import PycodeError  # noqa: PLC0415
    from sphinx.pycode import ModuleAnalyzer  # noqa: PLC0415

    try:
        analyzer = ModuleAnalyzer.for_module(cls.__module__)
        analyzer.analyze()
    except PycodeError:
        return False
    return f"{cls.__qualname__}.{attr}" in analyzer.tagorder


def _owner(cls: type[Extractable], attr: str) -> type[Extractable] | None:
    """The class whose documentation should contain the attribute.

    This is the most basic class in the method resolution order that assigns the attribute,
    so that the attribute is documented once in the class that introduces it,
    even if the subclasses assign it again.
    If the source code analysis does not find the assignment, e.g. due to tuple unpacking,
    the most basic class that has the field is used instead.

    :param cls: the class that has the field
    :param attr: name of the attribute
    :return: the class, or None if the attribute is a property or a method, which have docstrings of their own
    """
    bases = [base for base in reversed(cls.__mro__) if issubclass(base, Extractable) and base is not Extractable]
    static = inspect.getattr_static(cls, attr, None)
    if isinstance(static, (property, functools.cached_property)) or inspect.isroutine(static):
        return None
    for base in bases:
        if _defined_in(base, attr):
            return base
    return next(base for base in bases if attr in base.FIELDS)


def field_attribute_docs() -> dict[AttributeKey, str]:
    """Get the documentation of the attributes from the field descriptions.

    :return: the formatted descriptions by the attribute keys
    """
    for module in FIELD_MODULES:
        importlib.import_module(module)
    docs: dict[AttributeKey, str] = {}
    for cls in _subclasses(Extractable):
        for field in cls.FIELDS.values():
            attr = attribute_name(field)
            if attr is None or not field.description:
                continue
            owner = _owner(cls, attr)
            if owner is None:
                continue
            # The first description of an attribute is used, i.e. that of the most basic class.
            docs.setdefault((owner.__module__, owner.__qualname__, attr), format_description(field.description))
    return docs


def add_field_docs(app: "Sphinx | None" = None) -> int:
    """Give the field descriptions to Sphinx autodoc as if they were ``#:`` comments of the attributes.

    Autodoc reads the ``#:`` comments from the cached source code analyzers of the modules,
    and therefore the descriptions are added to those.
    This should be connected to the ``builder-inited`` event, so that the descriptions are added before
    the documents are read.

    :param app: the Sphinx application (not used)
    :return: the number of attributes whose documentation was added
    """
    from sphinx.errors import PycodeError  # noqa: PLC0415
    from sphinx.pycode import ModuleAnalyzer  # noqa: PLC0415

    count = 0
    for (module, qualname, attr), doc in field_attribute_docs().items():
        try:
            analyzer = ModuleAnalyzer.for_module(module)
            # The analysis has to be done before adding the docs, as it would otherwise overwrite them.
            analyzer.analyze()
        except PycodeError:
            continue
        key = (qualname, attr)
        if any(line.strip() for line in analyzer.attr_docs.get(key, [])):
            continue
        # The format is the same as for the "#:" comments, i.e. the lines followed by an empty line.
        analyzer.attr_docs[key] = [*doc.splitlines(), ""]
        count += 1
    logger.info("Added the documentation of %d attributes from the field descriptions.", count)
    return count


def note_field_dependencies(
        app: "Sphinx",
        what: str,
        name: str,
        obj: object,
        options: object,
        lines: list[str]) -> None:
    """Make the documents of the classes with exportable fields depend on the field definitions.

    Sphinx re-reads a document only if the source files of its documented objects have changed.
    As the documentation of the attributes comes from the field definitions,
    the modules of the field definitions are registered as dependencies of these documents.
    Otherwise, changes to the field descriptions would not be shown in incremental builds.
    This should be connected to the ``autodoc-process-docstring`` event.

    :param app: the Sphinx application
    :param what: the type of the documented object
    :param name: the name of the documented object (not used)
    :param obj: the documented object
    :param options: the autodoc options (not used)
    :param lines: the lines of the docstring (not used)
    """
    if what == "class" and isinstance(obj, type) and issubclass(obj, Extractable):
        for module in FIELD_DEFINITION_MODULES:
            path = importlib.import_module(module).__file__
            if path is not None:
                app.env.note_dependency(path)
