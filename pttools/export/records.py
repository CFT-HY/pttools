"""Records of the data extracted from models, bubbles and spectra, and the extractor that creates them.

The extraction can be done in worker processes,
and the resulting records then sent to the main process for writing.
This is faster than sending the objects themselves,
since the records contain only the selected fields instead of all the arrays of the objects.
"""

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
    "class_name",
    "find_class",
    "table_base_class",
    "table_fields",
    "table_of",
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


def table_base_class(table: Table) -> type[Extractable]:
    """The base class of the objects of the given table."""
    return {
        Table.MODELS: BaseModel,
        Table.BUBBLES: Bubble,
        Table.SPECTRA_Y: SSMSpectrum,
        Table.SPECTRA_F: Spectrum,
    }[table]


def table_fields(cls: type[Extractable], table: Table) -> Fields:
    """The fields of the objects of the given class in the given table.

    :param cls: the class of the objects
    :param table: the table
    :return: :py:attr:`pttools.omgw0.spectrum.Spectrum.FIELDS_F` for :py:attr:`Table.SPECTRA_F`,
        :py:attr:`pttools.omgw0.spectrum.Spectrum.FIELDS_Y` for :py:attr:`Table.SPECTRA_Y`,
        and the :py:attr:`~pttools.utils.fields.Extractable.FIELDS` of the class otherwise,
        e.g. for :py:class:`pttools.ssm.spectrum.SSMSpectrum`, which supports only the $y$ array
    """
    if table == Table.SPECTRA_F:
        if not issubclass(cls, Spectrum):
            raise TypeError(f"Only spectra of the type {class_name(Spectrum)} can be in the table {table}.")
        return cls.FIELDS_F
    if table == Table.SPECTRA_Y and issubclass(cls, Spectrum):
        return cls.FIELDS_Y
    return cls.FIELDS


def table_of(obj: object) -> Table:
    """The table to which the given object belongs."""
    if isinstance(obj, Spectrum) and obj.f_given:
        return Table.SPECTRA_F
    if isinstance(obj, SSMSpectrum):
        return Table.SPECTRA_Y
    if isinstance(obj, Bubble):
        return Table.BUBBLES
    if isinstance(obj, BaseModel):
        return Table.MODELS
    raise TypeError(f"Cannot export objects of type {type(obj)}. Supported: models, bubbles and spectra.")


@dataclasses.dataclass(frozen=True, slots=True)
class Record:
    """The extracted data of a model, bubble or spectrum.

    :param table: the table of the object
    :param id: unique identifier of the object, which is used for deduplication
    :param cls: fully qualified name of the class of the object
    :param data: the values of the extracted fields
    :param parent: the record of the model of a bubble, or of the bubble of a spectrum
    """

    table: Table
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
    """

    def __init__(
            self,
            model_fields: FieldSpec = Preset.MINIMAL,
            bubble_fields: FieldSpec = Preset.MINIMAL,
            spectrum_fields: FieldSpec = Preset.MINIMAL,
            importable: bool = True) -> None:
        self.specs: dict[Table, tuple[Preset | str | Field, ...]] = {
            Table.MODELS: _as_tuple(model_fields),
            Table.BUBBLES: _as_tuple(bubble_fields),
            Table.SPECTRA_Y: _as_tuple(spectrum_fields),
            # The spectra that share f have the same field specification, but different field definitions.
            Table.SPECTRA_F: _as_tuple(spectrum_fields),
        }
        self.importable: bool = importable
        self._cache: dict[tuple[type, Table], tuple[Field, ...]] = {}

    def __getstate__(self) -> dict[str, tp.Any]:
        """Get the state for pickling without the cache, which may contain fields that cannot be pickled."""
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state

    def fields(self, cls: type[Extractable], table: Table | None = None) -> tuple[Field, ...]:
        """The selected fields for the given class.

        :param cls: the class of the objects
        :param table: the table of the objects. If None, it's determined from the class.
        :return: the selected fields
        """
        if table is None:
            table = next(tbl for tbl in Table if issubclass(cls, table_base_class(tbl)))
        key = (cls, table)
        if key in self._cache:
            return self._cache[key]
        spec = self.specs[table] + ((Preset.INIT,) if self.importable else ())
        fields = table_fields(cls, table).select(spec, cls=cls)
        self._cache[key] = fields
        return fields

    def extract(self, obj: BaseModel | Bubble | SSMSpectrum) -> Record:
        """Extract the selected fields of an object and of its parents.

        :param obj: a model, a bubble or a spectrum
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
            id=obj.id,
            cls=class_name(type(obj)),
            data=extract(obj, self.fields(type(obj), table)),
            parent=parent
        )
