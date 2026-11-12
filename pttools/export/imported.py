r"""Common functionality of the bubbles and spectra that are imported from a file without recomputing them.

The imported objects are instances of the same base classes as the computed ones,
e.g. :py:class:`pttools.bubble.bubble.nucleated.BaseNucleatedBubble`,
so that the plotting and analysis functions that use only the available data can take either of them.
The quantities that are not stored in the file are computed from the stored data when possible,
e.g. $\kappa$ from the fluid profiles, in the same way as for the computed objects.
The quantities that need data that is not stored raise :py:class:`NotImportedError`.
"""

from collections.abc import Mapping
import functools
import inspect
import typing as tp

import numpy as np

from pttools.utils.fields import Fields, attribute_name

__all__ = [
    "Imported",
    "NotImportedError",
]


class NotImportedError(AttributeError):
    """The data needed for a quantity of an imported object is not available, as it was not exported to the file.

    This is a subclass of :py:class:`AttributeError`, so that e.g. :py:func:`hasattr` returns False.
    """


class Imported:
    """Mixin for the objects whose data is imported from a file.

    The stored fields are set as the attributes of the object, see :py:meth:`_import_fields`.
    The attributes that are not set raise :py:class:`NotImportedError`.
    """

    #: Description of the source of the data for the error messages, e.g. ``"bubble 3 of data.h5"``
    source: str
    #: Names of the fields that were imported
    imported_fields: frozenset[str]

    def _import_fields(self, data: Mapping[str, tp.Any], fields: Fields, source: str) -> None:
        """Set the stored fields as the attributes of this object.

        - Fields that have a ``decode`` function are constructor parameters, and they are decoded and set
          to the attribute of the name of the field, e.g. ``suppression`` from the name of the suppression dataset.
        - Fields that are read from an attribute of this object are set to that attribute.
          If the attribute is a :py:class:`functools.cached_property`, the stored value is used as its cached value.
          Plain properties and methods are computed by the class, and therefore their stored values are not used.
        - Fields that are computed by methods or read from other objects, e.g. ``bubble.v_wall``, are not set.
          The subclasses can use them from ``data``, e.g. the frequencies $f$ of the spectra.

        :param data: the stored values by the names of the fields
        :param fields: the fields of the class of the stored object
        :param source: description of the source of the data for the error messages
        """
        self.source = source
        self.imported_fields = frozenset(data)
        cls = type(self)
        for name, stored in data.items():
            field = fields.get(name)
            if field is None:
                continue
            value = stored.item() if isinstance(stored, np.generic) else stored
            if field.decode is not None:
                attr: str | None = field.name
                value = field.decode(value)
            else:
                attr = attribute_name(field)
            if attr is None:
                continue
            static = inspect.getattr_static(cls, attr, None)
            if isinstance(static, functools.cached_property):
                self.__dict__[attr] = value
            elif isinstance(static, property):
                if static.fset is not None:
                    setattr(self, attr, value)
            elif not inspect.isroutine(static):
                setattr(self, attr, value)

    def __getattr__(self, name: str) -> tp.Any:
        """Raise :py:class:`NotImportedError` for the attributes that were not imported.

        When a property raises :py:class:`AttributeError`, e.g. as it needs data that was not imported,
        Python calls this with the name of the property.
        Then the property is called again directly, so that the original error, which tells the missing data,
        is raised instead of an error about the property itself.

        :param name: name of the attribute
        :raises NotImportedError: if the attribute was not imported
        """
        static = inspect.getattr_static(type(self), name, None)
        if static is not None and hasattr(static, "__get__") and not name.startswith("__"):
            return static.__get__(self, type(self))
        # The attributes of the instance are read from __dict__ to avoid recursion.
        source = self.__dict__.get("source", "the file")
        raise NotImportedError(
            f"\"{name}\" is not available for the {type(self).__name__} of {source}, "
            "as it was not exported to the file. "
            "To include it, export the data with more fields, e.g. with pttools.utils.fields.Preset.FULL."
        )
