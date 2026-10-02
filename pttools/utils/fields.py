"""Field definitions for extracting the parameters and arrays of PTtools objects.

The classes whose data can be exported, such as
:py:class:`pttools.models.base.BaseModel`, :py:class:`pttools.bubble.bubble.Bubble`
and :py:class:`pttools.ssm.spectrum.SSMSpectrum`,
declare their exportable quantities as :py:class:`Field` objects in their :py:attr:`Extractable.FIELDS`.
A subset of these is then selected with a :py:data:`FieldSpec`,
which can consist of :py:class:`Preset` values, field names and custom :py:class:`Field` objects.
"""

from collections.abc import Callable, Iterable, Iterator, Mapping, Set
import dataclasses
import enum
import math
import operator
import typing as tp

__all__ = [
    "Extractable",
    "Field",
    "FieldShape",
    "FieldSpec",
    "FieldType",
    "Fields",
    "Preset",
    "decode_optional",
    "decode_optional_int",
    "extract",
]


class Preset(enum.StrEnum):
    """Predefined sets of fields."""

    #: The most relevant parameters and arrays, e.g. for training machine learning models.
    MINIMAL = "minimal"
    #: The fields of the JSON export.
    FULL = "full"
    #: The constructor parameters that are needed for recreating the object.
    INIT = "init"


class FieldType(enum.StrEnum):
    """Data type of a field."""

    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STR = "str"


class FieldShape(enum.StrEnum):
    """Shape of a field."""

    #: A single value per object
    SCALAR = "scalar"
    #: A 1D array, whose length has to be the same for all the objects stored in a file
    ARRAY = "array"
    #: A 1D array, whose values have to be the same for all the objects stored in a file, e.g. $y = kR_*$.
    GRID = "grid"
    #: A 1D array, whose length can vary by object, e.g. the fluid profiles of bubbles.
    #: The ragged fields that have the same :py:attr:`Field.axis` must have the same length for each object.
    RAGGED = "ragged"


@dataclasses.dataclass(frozen=True, slots=True)
class Field:
    r"""An exportable quantity of an object.

    :param name: name of the field
    :param getter: function that returns the value of the field for a given object,
        or the dotted name of the attribute, e.g. ``"bubble.v_wall"``.
        If None, the attribute with the name of the field is used.
        Note that only names and module-level functions can be pickled.
    :param type: data type of the field
    :param shape: shape of the field
    :param presets: the presets that include this field
    :param description: description of the field, e.g. ``"$v_\text{wall}$, wall speed"``
    :param axis: name of the axis of an array field, e.g. ``"y"``
    :param decode: function that converts the stored value back to the constructor argument,
        e.g. NaN to None
    """

    name: str
    getter: Callable[[tp.Any], tp.Any] | str | None = None
    type: FieldType = FieldType.FLOAT
    shape: FieldShape = FieldShape.SCALAR
    presets: Set[Preset] = frozenset()
    description: str = ""
    axis: str = ""
    decode: Callable[[tp.Any], tp.Any] | None = None

    def __post_init__(self) -> None:
        # Iterables such as sets and tuples are accepted for convenience.
        object.__setattr__(self, "presets", frozenset(self.presets))
        if self.shape == FieldShape.RAGGED and not self.axis:
            raise ValueError(f"The ragged field \"{self.name}\" must have an axis.")
        if self.shape != FieldShape.SCALAR and self.type not in (FieldType.FLOAT, FieldType.INT):
            raise ValueError(f"The array field \"{self.name}\" must be numerical. Got: {self.type}")

    def get(self, obj: tp.Any) -> tp.Any:
        """Get the value of the field for the given object."""
        if self.getter is None:
            return getattr(obj, self.name)
        if isinstance(self.getter, str):
            return operator.attrgetter(self.getter)(obj)
        return self.getter(obj)


#: Specification of fields: a preset, a field name, a custom field, or an iterable of these.
#: A string that is not a field name but the value of a :py:class:`Preset` is interpreted as the preset.
type FieldSpec = Preset | str | Field | Iterable[Preset | str | Field]


class Fields(Mapping[str, Field]):
    """An ordered collection of fields.

    :param fields: fields or collections of fields.
        If several fields have the same name, the last one replaces the previous ones in their original position.
        This can be used for changing the presets of an inherited field.
    """

    def __init__(self, *fields: "Field | Fields") -> None:
        self._fields: dict[str, Field] = {}
        for item in fields:
            if isinstance(item, Fields):
                self._fields.update(item._fields)
            else:
                self._fields[item.name] = item

    def __getitem__(self, key: str) -> Field:
        return self._fields[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __repr__(self) -> str:
        return f"Fields({', '.join(self._fields)})"

    def preset(self, preset: Preset) -> tuple[Field, ...]:
        """Get the fields of the given preset."""
        return tuple(field for field in self._fields.values() if preset in field.presets)

    def select(self, spec: FieldSpec) -> tuple[Field, ...]:
        """Select fields according to the given specification.

        The fields are returned in the order of the specification,
        and the fields of a preset in their order in this collection.
        Duplicates are removed.

        :param spec: a preset, a field name, a custom field, or an iterable of these
        :return: the selected fields
        :raises KeyError: if a field name is not found
        """
        items = [spec] if isinstance(spec, (str, Field)) else list(spec)
        selected: dict[str, Field] = {}
        for item in items:
            if isinstance(item, Field):
                selected[item.name] = item
            elif isinstance(item, Preset) or (item not in self._fields and item in Preset):
                for field in self.preset(Preset(item)):
                    selected.setdefault(field.name, field)
            elif item in self._fields:
                selected.setdefault(item, self._fields[item])
            else:
                raise KeyError(
                    f"Unknown field: \"{item}\". "
                    f"Available presets: {', '.join(Preset)}. Available fields: {', '.join(self._fields)}"
                )
        return tuple(selected.values())


def extract(obj: tp.Any, fields: Iterable[Field]) -> dict[str, tp.Any]:
    """Extract the values of the given fields from an object."""
    return {field.name: field.get(obj) for field in fields}


class Extractable:
    """Base class for the objects whose fields can be extracted."""

    #: The fields that can be extracted from the objects of this class.
    #: Subclasses can extend this with :py:class:`Fields`.
    FIELDS: tp.ClassVar[Fields] = Fields()

    def extract(self, fields: FieldSpec = Preset.MINIMAL) -> dict[str, tp.Any]:
        """Extract the given fields as a dictionary.

        :param fields: the fields to extract, see :py:data:`FieldSpec`
        :return: the values of the fields
        """
        return extract(self, self.FIELDS.select(fields))


def decode_optional(value: tp.Any) -> tp.Any:
    """Convert NaN to None."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def decode_optional_int(value: tp.Any) -> int | None:
    """Convert NaN to None and other numbers to int."""
    value = decode_optional(value)
    return None if value is None else int(value)
