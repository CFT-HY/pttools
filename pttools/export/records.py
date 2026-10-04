"""Records of the data extracted from models, bubbles and spectra, and the extractor that creates them.

The extraction can be done in worker processes,
and the resulting records then sent to the main process for writing.
This is faster than sending the objects themselves,
since the records contain only the selected fields instead of all the arrays of the objects.

In addition to the built-in tables of the models, bubbles and spectra of PTtools,
the objects of other :py:class:`~pttools.utils.fields.Extractable` classes, e.g. the spectra of other libraries,
can be exported to tables of their own by setting :py:attr:`~pttools.utils.fields.Extractable.TABLE`.
These tables have no parents, and all the objects of such a table must be of the same class.
"""

from collections.abc import Mapping
import dataclasses
import enum
import typing as tp

from pttools.bubble.bubble import Bubble
from pttools.models.base import BaseModel
from pttools.omgw0.spectrum import Spectrum
from pttools.ssm.spectrum import SSMSpectrum
from pttools.utils.fields import Extractable, Field, Fields, FieldSpec, Preset, extract

__all__ = [
    "Extractor",
    "Record",
    "Table",
    "TableName",
    "class_name",
    "find_class",
    "is_builtin_table",
    "object_id",
    "table_base_class",
    "table_fields",
    "table_of",
    "validate_table_name",
]


class Table(enum.StrEnum):
    """The tables of an exported file."""

    MODELS = "models"
    BUBBLES = "bubbles"
    #: Spectra that share the $y$ array
    SPECTRA_Y = "spectra_y"
    #: Spectra that have been given the frequencies $f$, and therefore share $f$ instead of $y$,
    #: see :py:attr:`pttools.omgw0.spectrum.Spectrum.f_given`
    SPECTRA_F = "spectra_f"


#: Name of a table: a built-in :py:class:`Table`,
#: or the :py:attr:`~pttools.utils.fields.Extractable.TABLE` of another class
type TableName = Table | str

#: Maximum length of the identifiers of the objects
_ID_MAX_LENGTH: int = 32

#: The names of the built-in tables
_BUILTIN_TABLES: frozenset[str] = frozenset(table.value for table in Table)


def is_builtin_table(table: TableName) -> bool:
    """Whether the table is one of the built-in tables of :py:class:`Table`."""
    return str(table) in _BUILTIN_TABLES


def validate_table_name(name: str) -> str:
    """Validate the name of a table of another class than the built-in ones.

    :param name: the name of the table
    :return: the name
    :raises ValueError: if the name is reserved for a built-in table, or is not a valid name for an HDF5 group
    """
    if is_builtin_table(name):
        raise ValueError(f"The table name \"{name}\" is reserved for the built-in table.")
    if not name or "/" in name or name in (".", ".."):
        raise ValueError(f"Invalid table name: \"{name}\". The name must be non-empty and must not contain \"/\".")
    return name


def class_name(cls: type) -> str:
    """Fully qualified name of a class, e.g. ``pttools.models.bag.BagModel``."""
    return f"{cls.__module__}.{cls.__qualname__}"


def find_class[T](base: type[T], name: str) -> type[T]:
    """Find a class by its fully qualified name among the given base class and its subclasses.

    Only the classes that have already been imported can be found.
    This is safer than importing a module by name, since the names may come from an untrusted file.

    :param base: the base class
    :param name: fully qualified name of the class, see :py:func:`class_name`
    :return: the class
    :raises ValueError: if the class is not found
    """
    stack: list[type] = [base]
    seen: set[type] = set()
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        if class_name(cls) == name:
            return tp.cast(type[T], cls)
        stack.extend(cls.__subclasses__())
    raise ValueError(
        f"Could not find the class \"{name}\" among the subclasses of {class_name(base)}. "
        "If it's a custom class, please import it before loading the data."
    )


def object_id(obj: object) -> str:
    """The unique identifier of an object, which is used for deduplication when exporting.

    The identifiers are stored as ASCII strings of at most 32 characters,
    which is the length of the ``uuid.uuid4().hex`` identifiers of the PTtools objects.

    :param obj: the object or its record
    :return: the ``id`` attribute of the object
    :raises ValueError: if the object has no ``id`` attribute, or it's empty, too long or not ASCII
    """
    id_ = getattr(obj, "id", None)
    if not isinstance(id_, str) or not id_ or len(id_) > _ID_MAX_LENGTH or not id_.isascii():
        raise ValueError(
            f"The object of {class_name(type(obj))} must have a unique \"id\" attribute, "
            f"which is a non-empty ASCII string of at most {_ID_MAX_LENGTH} characters. Got: {id_!r}"
        )
    return id_


def table_base_class(table: TableName) -> type[Extractable]:
    """The base class of the objects of the given table.

    For the tables of other classes, this is :py:class:`~pttools.utils.fields.Extractable`.
    """
    if not is_builtin_table(table):
        return Extractable
    return {
        Table.MODELS: BaseModel,
        Table.BUBBLES: Bubble,
        Table.SPECTRA_Y: SSMSpectrum,
        Table.SPECTRA_F: Spectrum,
    }[Table(table)]


def table_fields(cls: type[Extractable], table: TableName) -> Fields:
    """The fields of the objects of the given class in the given table.

    :param cls: the class of the objects
    :param table: the table
    :return: :py:attr:`pttools.omgw0.spectrum.Spectrum.FIELDS_F` for :py:attr:`Table.SPECTRA_F`,
        :py:attr:`pttools.omgw0.spectrum.Spectrum.FIELDS_Y` for :py:attr:`Table.SPECTRA_Y`,
        and the :py:attr:`~pttools.utils.fields.Extractable.FIELDS` of the class otherwise,
        e.g. for :py:class:`pttools.ssm.spectrum.SSMSpectrum`, which supports only the $y$ array,
        and for the classes of the other tables
    """
    if table == Table.SPECTRA_F:
        if not issubclass(cls, Spectrum):
            raise TypeError(f"Only spectra of the type {class_name(Spectrum)} can be in the table {table}.")
        return cls.FIELDS_F
    if table == Table.SPECTRA_Y and issubclass(cls, Spectrum):
        return cls.FIELDS_Y
    return cls.FIELDS


def table_of(obj: object) -> TableName:
    """The table to which the given object belongs.

    :param obj: a model, a bubble, a spectrum, or an object of another class that has a table,
        see :py:attr:`pttools.utils.fields.Extractable.TABLE`
    :return: the table
    :raises TypeError: if the object cannot be exported
    """
    if isinstance(obj, Spectrum) and obj.f_given:
        return Table.SPECTRA_F
    if isinstance(obj, SSMSpectrum):
        return Table.SPECTRA_Y
    if isinstance(obj, Bubble):
        return Table.BUBBLES
    if isinstance(obj, BaseModel):
        return Table.MODELS
    if isinstance(obj, Extractable) and obj.TABLE is not None:
        return validate_table_name(obj.TABLE)
    raise TypeError(
        f"Cannot export objects of type {type(obj)}. "
        "Supported: models, bubbles, spectra and other Extractable classes that have a TABLE."
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Record:
    """The extracted data of a model, bubble or spectrum.

    :param table: the table of the object
    :param id: unique identifier of the object, which is used for deduplication
    :param cls: fully qualified name of the class of the object
    :param data: the values of the extracted fields
    :param parent: the record of the model of a bubble, or of the bubble of a spectrum
    """

    table: TableName
    id: str
    cls: str
    data: dict[str, tp.Any]
    parent: "Record | None" = None


def _as_tuple(spec: FieldSpec) -> tuple[Preset | str | Field, ...]:
    """Convert a field specification to a tuple of its items, so that it can be extended and stored."""
    if isinstance(spec, (str, Field)):
        return (spec,)
    return tuple(spec)


class Extractor:
    """Extracts the selected fields of models, bubbles and spectra to :py:class:`Record` objects.

    The extractor can be pickled and sent to worker processes,
    as long as the custom fields of the specifications, if any, can be pickled.

    :param model_fields: the fields of the models, see :py:data:`pttools.utils.fields.FieldSpec`
    :param bubble_fields: the fields of the bubbles
    :param spectrum_fields: the fields of the spectra
    :param importable: whether to include the fields of the :py:attr:`~pttools.utils.fields.Preset.INIT` preset,
        which are needed for recreating the objects with :py:class:`pttools.export.importer.Importer`
    :param other_fields: the fields of the tables of other classes by the name of the table,
        see :py:attr:`pttools.utils.fields.Extractable.TABLE`.
        The tables that are not given here have the fields of the :py:attr:`~pttools.utils.fields.Preset.MINIMAL`
        preset.
    """

    def __init__(
            self,
            model_fields: FieldSpec = Preset.MINIMAL,
            bubble_fields: FieldSpec = Preset.MINIMAL,
            spectrum_fields: FieldSpec = Preset.MINIMAL,
            importable: bool = True,
            other_fields: Mapping[str, FieldSpec] | None = None) -> None:
        """Store the field specifications. The fields are resolved and cached when they're first needed."""
        self.specs: dict[Table, tuple[Preset | str | Field, ...]] = {
            Table.MODELS: _as_tuple(model_fields),
            Table.BUBBLES: _as_tuple(bubble_fields),
            Table.SPECTRA_Y: _as_tuple(spectrum_fields),
            # The spectra that share f have the same field specification, but different field definitions.
            Table.SPECTRA_F: _as_tuple(spectrum_fields),
        }
        #: The field specifications of the tables of other classes
        self.other_specs: dict[str, tuple[Preset | str | Field, ...]] = {
            validate_table_name(name): _as_tuple(spec) for name, spec in (other_fields or {}).items()
        }
        self.importable: bool = importable
        self._cache: dict[tuple[type, TableName], tuple[Field, ...]] = {}

    def __getstate__(self) -> dict[str, tp.Any]:
        """Get the state for pickling without the cache, which may contain fields that cannot be pickled."""
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state

    def fields(self, cls: type[Extractable], table: TableName | None = None) -> tuple[Field, ...]:
        """The selected fields for the given class.

        :param cls: the class of the objects
        :param table: the table of the objects. If None, it's determined from the class.
        :return: the selected fields
        :raises TypeError: if the table is not given, and the class has no table
        """
        if table is None:
            table = next((tbl for tbl in Table if issubclass(cls, table_base_class(tbl))), None)
            if table is None:
                if cls.TABLE is None:
                    raise TypeError(f"The class {class_name(cls)} has no table.")
                table = validate_table_name(cls.TABLE)
        key = (cls, table)
        if key in self._cache:
            return self._cache[key]
        spec = self.spec(table) + ((Preset.INIT,) if self.importable else ())
        fields = table_fields(cls, table).select(spec, cls=cls)
        self._cache[key] = fields
        return fields

    def spec(self, table: TableName) -> tuple[Preset | str | Field, ...]:
        """The field specification of the given table, without the :py:attr:`~Preset.INIT` preset."""
        if is_builtin_table(table):
            return self.specs[Table(table)]
        return self.other_specs.get(str(table), (Preset.MINIMAL,))

    def extract(self, obj: Extractable) -> Record:
        """Extract the selected fields of an object and of its parents.

        :param obj: a model, a bubble, a spectrum, or an object of another class that has a table
        :return: the record of the object, which contains the records of its bubble and model as parents
        """
        table = table_of(obj)
        parent: Record | None = None
        if isinstance(obj, SSMSpectrum):
            parent = self.extract(obj.bubble)
        elif isinstance(obj, Bubble):
            parent = self.extract(obj.model)
        return Record(
            table=table,
            id=object_id(obj),
            cls=class_name(type(obj)),
            data=extract(obj, self.fields(type(obj), table)),
            parent=parent
        )
