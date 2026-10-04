r"""Exporter for writing models, bubbles and spectra to an HDF5 file.

File layout
-----------
The file has a group for each :py:class:`~pttools.export.records.Table`.
Each group has a dataset for each field, so that the values of a field can be read at once.

.. code-block:: text

    /                           attrs: format, format_version, pttools_version, created
    /models/id                  (N_models,)            unique identifiers
    /models/class               (N_models,)            fully qualified class names
    /models/params              (N_models,)            the fields of each model as a JSON string
    /bubbles/id                 (N_bubbles,)
    /bubbles/model              (N_bubbles,)           row index in /models
    /bubbles/v_wall             (N_bubbles,)           a scalar field
    /bubbles/v                  (sum of lengths,)      a ragged field: the profiles of all bubbles concatenated
    /bubbles/xi_offsets         (N_bubbles + 1,)       the profile of bubble i is v[xi_offsets[i]:xi_offsets[i+1]]
    /spectra_y/id               (N_spectra_y,)         spectra that have been given the y array
    /spectra_y/bubble           (N_spectra_y,)         row index in /bubbles
    /spectra_y/r_star           (N_spectra_y,)         a scalar field
    /spectra_y/omgw0_h2         (N_spectra_y, n_y)     an array field
    /spectra_y/y                (n_y,)                 the grid of these spectra
    /spectra_f/id               (N_spectra_f,)         spectra that have been given the frequencies f instead of y
    /spectra_f/bubble           (N_spectra_f,)         row index in /bubbles
    /spectra_f/omgw0_h2         (N_spectra_f, n_f)     an array field
    /spectra_f/f                (n_f,)                 the grid of these spectra

The spectra that have been given the frequencies $f$ (see :py:attr:`pttools.omgw0.spectrum.Spectrum.f_given`)
are stored in a separate group, where $f$ is the grid instead of $y$.
Therefore, a file can contain both a set of spectra with the same $y$ and a set of spectra with the same $f$,
and the arrays of these sets can have different lengths.
The bubbles and models are shared by both sets.

The objects of other :py:class:`~pttools.utils.fields.Extractable` classes,
such as the spectra of other libraries that use the same file format,
are stored in groups named by their :py:attr:`~pttools.utils.fields.Extractable.TABLE`.
These groups have the same layout as the groups of the spectra, but without the parent column.

.. code-block:: text

    /TABLE/id                   (N_TABLE,)
    /TABLE/omgw0_h2             (N_TABLE, n_f)         an array field
    /TABLE/f                    (n_f,)                 a grid field

The model parameters are stored as JSON, since different model classes have different parameters.
The groups have the attributes ``n_rows`` (the number of committed rows) and ``fields`` (the names of the fields).
The groups of the bubbles and spectra also have the attribute ``class``,
as all the bubbles of a file, and all the spectra of each group, must be of the same class.
The datasets have the attributes ``kind``, ``type``, ``axis`` and ``description``.

The rows are buffered in memory and written in batches.
A batch is committed by updating the ``n_rows`` attributes after all its data has been written,
so if the writing is interrupted, the partially written rows are discarded when the file is opened again.

Integrity
---------
The numerical datasets are protected by the Fletcher-32 checksum filter of HDF5,
which verifies each chunk automatically when it's read.
In addition, a SHA-256 checksum of the entire file is written to ``FILE.sha256`` when the exporter is closed.
It can be verified with :py:func:`pttools.export.checksum.verify_checksum` or ``sha256sum --check FILE.sha256``.
"""

from collections.abc import Iterable
import dataclasses
import datetime
import enum
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import typing as tp

import h5py
import numpy as np

from pttools.bubble.bubble import Bubble
from pttools.export.checksum import checksum_path, write_checksum
from pttools.export.records import (
    Extractor,
    Record,
    Table,
    TableName,
    class_name,
    find_class,
    is_builtin_table,
    object_id,
    table_base_class,
    table_of,
    validate_table_name,
)
from pttools.ssm.spectrum import SSMSpectrum
from pttools.utils.fields import Extractable, Field, FieldShape, FieldSpec, FieldType, Preset, extract

__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "ExportFormatError",
    "Exporter",
    "offsets_name",
    "parent_column",
]

logger: logging.Logger = logging.getLogger(__name__)

#: Value of the ``format`` attribute of the exported files
FORMAT_NAME: str = "pttools"
#: Version of the file format. This should be incremented when the format is changed in an incompatible way.
FORMAT_VERSION: int = 1

#: Approximate size of the chunks of the datasets in bytes.
#: The chunks should be roughly in the range of 10 kiB to 1 MiB.
CHUNK_BYTES: int = 128 * 1024

#: Names of the datasets that are not fields
_RESERVED_NAMES: frozenset[str] = frozenset({"id", "class", "params", "model", "bubble"})

#: Value of the ``kind`` attribute of the offsets datasets of the ragged fields
_OFFSETS_KIND: str = "offsets"

_PARENT_COLUMNS: dict[Table, str] = {
    Table.BUBBLES: "model",
    Table.SPECTRA_Y: "bubble",
    Table.SPECTRA_F: "bubble",
}


class ExportFormatError(ValueError):
    """The file is not a valid PTtools export file, or its contents do not match the exporter settings."""


def offsets_name(axis: str) -> str:
    """Name of the dataset that contains the offsets of the ragged fields of the given axis."""
    return f"{axis}_offsets"


def parent_column(table: TableName) -> str | None:
    """Name of the dataset that contains the row indices of the parents, if any.

    The tables of other classes than the built-in ones have no parents.
    """
    return _PARENT_COLUMNS.get(Table(table)) if is_builtin_table(table) else None


def _pttools_version() -> str:
    """Version of the installed PTtools package, or "unknown" if it's not installed as a package."""
    try:
        return importlib.metadata.version("pttools-gw")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _json_default(value: tp.Any) -> tp.Any:
    """Convert the values that the json module cannot serialize, such as Numpy arrays and dates.

    This is used as the ``default`` argument of :py:func:`json.dumps`.

    :param value: the value to convert
    :return: a value that can be serialized
    :raises TypeError: if the value cannot be converted
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, enum.Enum):
        return value.value
    raise TypeError(f"Cannot convert {type(value)} to JSON.")


def _encode_str(value: tp.Any) -> str:
    """Convert a value to a string for storing it in a string field.

    None is converted to an empty string, enums to their values, dates to ISO 8601,
    and lists and tuples to one item per line.

    :param value: the value to convert
    :return: the string
    """
    if value is None:
        return ""
    if isinstance(value, enum.Enum):
        return str(value.value)
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value)
    return str(value)


def _encode_scalar(field: Field, value: tp.Any) -> tp.Any:
    """Convert the value of a scalar field to the Python type corresponding to the type of the field.

    For float fields, None is converted to NaN.

    :param field: the field
    :param value: the value to convert
    :return: the converted value
    """
    match field.type:
        case FieldType.FLOAT:
            return np.nan if value is None else float(value)
        case FieldType.INT:
            return int(value)
        case FieldType.BOOL:
            return bool(value)
        case FieldType.STR:
            return _encode_str(value)
    raise ValueError(f"Unknown field type: {field.type}")


def _numpy_dtype(field_type: FieldType) -> tp.Any:
    """The Numpy dtype used for storing the values of the given field type in HDF5."""
    return {
        FieldType.BOOL: np.bool_,
        FieldType.INT: np.int64,
        FieldType.FLOAT: np.float64,
        FieldType.STR: h5py.string_dtype(),
    }[field_type]


#: The columns of the models table, where the fields of each model are stored as JSON
_MODEL_COLUMNS: tuple[Field, ...] = (
    Field("class", type=FieldType.STR, description="fully qualified name of the model class"),
    Field("params", type=FieldType.STR, description="the exported fields of the model as a JSON object"),
)


def _set_attrs(dset: h5py.Dataset, kind: str, field_type: str, axis: str, description: str) -> None:
    """Set the metadata attributes of a dataset.

    The values are converted to plain strings, since h5py does not support StrEnum attributes.

    :param dset: the dataset
    :param kind: the kind of the dataset, i.e. the shape of the field or "offsets"
    :param field_type: the type of the field
    :param axis: the axis of an array field
    :param description: the description of the field
    """
    dset.attrs["kind"] = str(kind)
    dset.attrs["type"] = str(field_type)
    dset.attrs["axis"] = str(axis)
    dset.attrs["description"] = str(description)


@dataclasses.dataclass(slots=True)
class _Row:
    """A buffered row of a table.

    :param id: unique identifier of the object
    :param parent: row index of the parent object, or -1 if there is none
    :param data: the encoded values of the fields
    """

    id: str
    parent: int
    data: dict[str, tp.Any]


class _TableWriter:
    """Writer for a single table, i.e. an HDF5 group.

    :param group: the HDF5 group of the table
    :param table: the table
    :param compression: HDF5 compression filter, e.g. "gzip", "lzf" or None
    :param compression_opts: compression level for gzip
    """

    def __init__(
            self,
            group: h5py.Group,
            table: TableName,
            compression: str | None,
            compression_opts: int | None) -> None:
        self.group: h5py.Group = group
        self.table: TableName = table
        self.parent_column: str | None = parent_column(table)
        self.compression: str | None = compression
        self.compression_opts: int | None = compression_opts

        #: Number of committed rows
        self.n_rows: int = int(group.attrs.get("n_rows", 0))
        #: Number of rows written to the datasets but not yet committed
        self.n_written: int = self.n_rows
        self.cls: str | None = group.attrs.get("class")
        self.field_names: tuple[str, ...] | None = \
            tuple(group.attrs["fields"]) if "fields" in group.attrs else None
        self.fields: tuple[Field, ...] | None = None
        self.buffer: list[_Row] = []
        self.index: dict[str, int] = {}
        #: Values of the grid fields
        self.grids: dict[str, np.ndarray] = {}
        #: Lengths of the array fields
        self.array_lengths: dict[str, int] = {}
        #: Total lengths of the ragged axes
        self.ragged_sizes: dict[str, int] = {}

        if "id" in group:
            self._truncate()
            ids = group["id"][:self.n_rows]
            self.index = {id_.decode("ascii"): i for i, id_ in enumerate(ids)}
        for name, dset in group.items():
            kind = dset.attrs.get("kind")
            if kind == FieldShape.GRID:
                self.grids[name] = dset[()]
            elif kind == FieldShape.ARRAY:
                self.array_lengths[name] = dset.shape[1]

    @property
    def n_total(self) -> int:
        """Number of rows including the buffered ones."""
        return self.n_rows + len(self.buffer)

    def _truncate(self) -> None:
        """Discard the rows that were written but not committed, e.g. due to a crash."""
        n = self.n_rows
        offsets: dict[str, int] = {}
        for dset in self.group.values():
            if dset.attrs.get("kind") == _OFFSETS_KIND:
                if dset.shape[0] > n + 1:
                    dset.resize((n + 1,))
                offsets[dset.attrs["axis"]] = int(dset[n])
        self.ragged_sizes = offsets
        for dset in self.group.values():
            kind = dset.attrs.get("kind")
            if kind == FieldShape.RAGGED:
                size = offsets[dset.attrs["axis"]]
                if dset.shape[0] > size:
                    dset.resize((size,))
            elif kind in (FieldShape.SCALAR, FieldShape.ARRAY) and dset.shape[0] > n:
                dset.resize((n, *dset.shape[1:]))

    def set_schema(self, cls: str, fields: tuple[Field, ...]) -> None:
        """Set the fields of the table, or verify that they match those of the existing data."""
        # The models table stores the class of each model separately.
        if self.table != Table.MODELS and self.cls is not None and cls != self.cls:
            raise ExportFormatError(
                f"All the {self.table} of a file must be of the same class. Got: {cls}, expected: {self.cls}")
        if self.fields is not None:
            return
        names = tuple(field.name for field in fields)
        reserved = _RESERVED_NAMES.intersection(names) if self.table != Table.MODELS else set()
        if reserved:
            raise ExportFormatError(f"These field names are reserved: {reserved}")
        if self.field_names is None:
            self.group.attrs["fields"] = np.array(names, dtype=h5py.string_dtype())
            if self.table != Table.MODELS:
                self.group.attrs["class"] = cls
                self.cls = cls
            self.field_names = names
        elif set(names) != set(self.field_names):
            raise ExportFormatError(
                f"The fields of the {self.table} do not match those of the existing file. "
                f"Missing from the file: {sorted(set(names) - set(self.field_names))}, "
                f"extra in the file: {sorted(set(self.field_names) - set(names))}"
            )
        self.fields = fields

    def append(self, id_: str, cls: str, data: dict[str, tp.Any], parent: int, fields: tuple[Field, ...]) -> int:
        """Encode and buffer a row.

        :return: the row index
        """
        self.set_schema(cls, fields)
        encoded: dict[str, tp.Any] = {}
        ragged_lengths: dict[str, int] = {}
        for field in fields:
            value = data[field.name]
            if field.shape == FieldShape.SCALAR:
                encoded[field.name] = _encode_scalar(field, value)
                continue
            arr = np.asarray(value, dtype=_numpy_dtype(field.type))
            if arr.ndim != 1:
                raise ValueError(f"The array field \"{field.name}\" must be 1D. Got shape: {arr.shape}")
            if field.shape == FieldShape.GRID:
                grid = self.grids.setdefault(field.name, arr)
                if not (grid.shape == arr.shape and np.array_equal(grid, arr, equal_nan=True)):
                    raise ValueError(
                        f"The grid \"{field.name}\" must be the same for all the {self.table} of a file. "
                        "Please use a separate file for each grid."
                    )
            elif field.shape == FieldShape.ARRAY:
                length = self.array_lengths.setdefault(field.name, arr.size)
                if arr.size != length:
                    raise ValueError(
                        f"The array \"{field.name}\" must have the same length for all the {self.table} of a file. "
                        f"Got: {arr.size}, expected: {length}"
                    )
                encoded[field.name] = arr
            else:
                length = ragged_lengths.setdefault(field.axis, arr.size)
                if arr.size != length:
                    raise ValueError(
                        f"The ragged fields of the axis \"{field.axis}\" must have the same length for each object. "
                        f"Got: {arr.size} for \"{field.name}\", expected: {length}"
                    )
                encoded[field.name] = arr
        index = self.n_total
        self.buffer.append(_Row(id=id_, parent=parent, data=encoded))
        self.index[id_] = index
        return index

    def _create_dataset(
            self,
            name: str,
            dtype: tp.Any,
            row_shape: tuple[int, ...] = (),
            kind: str = FieldShape.SCALAR,
            field_type: str = "",
            axis: str = "",
            description: str = "") -> h5py.Dataset:
        """Create an empty resizable dataset with chunking, compression and checksums.

        The chunk size is chosen so that each chunk has approximately :py:data:`CHUNK_BYTES` bytes.
        The compression and the checksums are not applied to variable-length strings.

        :param name: name of the dataset
        :param dtype: data type of the dataset
        :param row_shape: shape of a single row, e.g. ``(n_y,)`` for an array field
        :param kind: the kind of the dataset, see :py:func:`_set_attrs`
        :param field_type: the type of the field
        :param axis: the axis of an array field
        :param description: the description of the field
        :return: the dataset
        """
        # Variable-length strings are stored as objects
        is_str = np.dtype(dtype).kind == "O"
        row_bytes = (16 if is_str else np.dtype(dtype).itemsize) * int(np.prod(row_shape))
        chunk_rows = int(np.clip(CHUNK_BYTES // max(row_bytes, 1), 1, 2**16))
        kwargs: dict[str, tp.Any] = {}
        if not is_str:
            kwargs = {
                "compression": self.compression,
                "compression_opts": self.compression_opts if self.compression == "gzip" else None,
                "shuffle": self.compression is not None and np.dtype(dtype).itemsize > 1,
                "fletcher32": True,
            }
        dset = self.group.create_dataset(
            name,
            shape=(0, *row_shape),
            maxshape=(None, *row_shape),
            chunks=(chunk_rows, *row_shape),
            dtype=dtype,
            **kwargs
        )
        _set_attrs(dset, kind=kind, field_type=field_type, axis=axis, description=description)
        return dset

    def _append_rows(self, name: str, values: np.ndarray, **create_kwargs: tp.Any) -> None:
        """Append rows to a dataset, and create the dataset if it does not exist yet.

        :param name: name of the dataset
        :param values: the rows to append, with the rows along the first axis
        :param create_kwargs: arguments for :py:meth:`_create_dataset`
        """
        dset = self.group[name] if name in self.group else \
            self._create_dataset(name, values.dtype if values.dtype.kind != "O" else h5py.string_dtype(),
                                 values.shape[1:], **create_kwargs)
        start = dset.shape[0]
        dset.resize((start + values.shape[0], *dset.shape[1:]))
        dset[start:] = values

    def write(self) -> None:
        """Write the buffered rows to the datasets without committing them."""
        if not self.buffer:
            return
        if self.fields is None:
            raise RuntimeError("The fields have not been set.")
        rows = self.buffer
        self._append_rows(
            "id", np.array([row.id for row in rows], dtype="S32"),
            field_type="id", description="unique identifier")
        if self.parent_column is not None:
            self._append_rows(
                self.parent_column, np.array([row.parent for row in rows], dtype=np.int64),
                field_type=FieldType.INT, description=f"row index in /{self.parent_table()}")

        offsets_written: set[str] = set()
        for field in self.fields:
            common = {
                "kind": field.shape, "field_type": field.type, "axis": field.axis, "description": field.description
            }
            # The grids are stored only once, and are therefore not in the rows.
            if field.shape == FieldShape.GRID:
                if field.name not in self.group:
                    dset = self.group.create_dataset(field.name, data=self.grids[field.name])
                    _set_attrs(dset, **common)
                continue
            values = [row.data[field.name] for row in rows]
            if field.shape == FieldShape.SCALAR:
                arr = np.array(values, dtype=object if field.type == FieldType.STR else _numpy_dtype(field.type))
                self._append_rows(field.name, arr, **common)
            elif field.shape == FieldShape.ARRAY:
                self._append_rows(field.name, np.stack(values), **common)
            else:
                self._append_rows(field.name, np.concatenate(values), **common)
                if field.axis not in offsets_written:
                    self._append_offsets(field.axis, np.array([value.size for value in values], dtype=np.int64))
                    offsets_written.add(field.axis)
        self.n_written += len(rows)
        self.buffer = []

    def _append_offsets(self, axis: str, lengths: np.ndarray) -> None:
        """Append the end offsets of new rows to the offsets dataset of a ragged axis.

        The offsets dataset is created with the initial offset 0 if it does not exist yet.

        :param axis: the ragged axis
        :param lengths: the lengths of the arrays of the new rows
        """
        name = offsets_name(axis)
        if name not in self.group:
            dset = self._create_dataset(
                name, np.int64, field_type=FieldType.INT, kind=_OFFSETS_KIND, axis=axis,
                description=f"offsets of the ragged fields of the axis {axis}")
            dset.resize((1,))
            dset[0] = 0
        size = self.ragged_sizes.get(axis, 0)
        offsets = size + np.cumsum(lengths)
        self._append_rows(name, offsets)
        self.ragged_sizes[axis] = int(offsets[-1])

    def parent_table(self) -> str:
        """The table of the parents of the rows of this table."""
        return Table.MODELS if self.table == Table.BUBBLES else Table.BUBBLES

    def commit(self) -> None:
        """Commit the written rows by updating the row count."""
        if self.n_written != self.n_rows:
            self.n_rows = self.n_written
            self.group.attrs["n_rows"] = self.n_rows


class Exporter:
    """Exporter for writing models, bubbles and spectra to an HDF5 file.

    The bubbles and models of the spectra are written automatically, and each object is written only once.
    The objects are identified by their ``id`` attributes, which are preserved when the objects are pickled.
    Therefore, the deduplication works also for objects that have been sent between processes.

    For parallel computation, the fields can be extracted in the worker processes with :py:attr:`extractor`,
    and the resulting records added to the exporter in the main process.
    This avoids sending the large arrays of the objects between the processes.

    .. code-block:: python

        from concurrent.futures import ProcessPoolExecutor
        import functools

        from pttools.bubble import Bubble
        from pttools.export import Exporter, Extractor, Record
        from pttools.models import BagModel, Model
        from pttools.omgw0 import Spectrum

        def compute(params: tuple[float, float, float], model: Model, extractor: Extractor) -> Record:
            v_wall, alpha_n, r_star = params
            bubble = Bubble(model, v_wall=v_wall, alpha_n=alpha_n)
            return extractor.extract(Spectrum(bubble, r_star=r_star))

        if __name__ == "__main__":
            model = BagModel(alpha_n_min=0.01)
            params = [(v_wall, 0.1, r_star) for v_wall in (0.3, 0.5, 0.7) for r_star in (0.1, 0.2)]
            with Exporter("spectra.h5") as exporter, ProcessPoolExecutor() as executor:
                # The extractor and the model are pickled and sent to the worker processes.
                worker = functools.partial(compute, model=model, extractor=exporter.extractor)
                exporter.add_many(executor.map(worker, params))

    :param path: path of the HDF5 file
    :param mode: "x" to create a new file and fail if it exists,
        "w" to create a new file and overwrite any existing one,
        "a" to append to an existing file or to create a new one.
        When appending, the fields must be the same as in the existing file.
    :param model_fields: the fields of the models, see :py:data:`pttools.utils.fields.FieldSpec`
    :param bubble_fields: the fields of the bubbles
    :param spectrum_fields: the fields of the spectra
    :param importable: whether to include the fields of the :py:attr:`~pttools.utils.fields.Preset.INIT` preset,
        which are needed for recreating the objects with :py:class:`pttools.export.importer.Importer`
    :param other_fields: the fields of the tables of other classes by the name of the table,
        see :py:attr:`pttools.utils.fields.Extractable.TABLE`.
        The tables that are not given here have the fields of the :py:attr:`~pttools.utils.fields.Preset.MINIMAL`
        preset.
    :param compression: HDF5 compression filter, e.g. "gzip", "lzf" or None
    :param compression_opts: compression level for gzip
    :param buffer_size: number of rows of a table to buffer in memory before writing
    :param checksum: whether to write a SHA-256 checksum file when the exporter is closed
    """

    def __init__(
            self,
            path: str | os.PathLike[str],
            mode: tp.Literal["x", "w", "a"] = "x",
            model_fields: FieldSpec = Preset.MINIMAL,
            bubble_fields: FieldSpec = Preset.MINIMAL,
            spectrum_fields: FieldSpec = Preset.MINIMAL,
            importable: bool = True,
            compression: str | None = "gzip",
            compression_opts: int | None = 4,
            buffer_size: int = 256,
            checksum: bool = True,
            other_fields: tp.Mapping[str, FieldSpec] | None = None) -> None:
        """Open the HDF5 file and initialize its tables.

        Any existing checksum file is removed, as it would become invalid when the file is modified.

        :raises ValueError: if the mode or the buffer size is invalid
        :raises FileExistsError: if the mode is "x" and the file already exists
        :raises pttools.export.exporter.ExportFormatError: if the file to be appended to is not a PTtools export file
            or has an incompatible format version
        """
        h5py_modes = {"x": "w-", "w": "w", "a": "a"}
        if mode not in h5py_modes:
            raise ValueError(f"Invalid mode: {mode}. Should be one of {list(h5py_modes)}.")
        if buffer_size < 1:
            raise ValueError(f"buffer_size must be positive. Got: {buffer_size}")

        self.path: Path = Path(path)
        #: The extractor, which can be sent to worker processes for extracting the fields there
        self.extractor: Extractor = Extractor(
            model_fields=model_fields,
            bubble_fields=bubble_fields,
            spectrum_fields=spectrum_fields,
            importable=importable,
            other_fields=other_fields
        )
        self.buffer_size: int = buffer_size
        self.compression: str | None = compression
        self.compression_opts: int | None = compression_opts
        self.checksum: bool = checksum

        if mode == "x" and self.path.exists():
            raise FileExistsError(f"The file already exists: {self.path}. Use mode=\"a\" to append to it.")
        # The checksum would become invalid when the file is modified.
        checksum_path(self.path).unlink(missing_ok=True)

        self._file: h5py.File = h5py.File(self.path, h5py_modes[mode])
        self._closed: bool = False
        #: Whether writing has failed, after which the file may contain partially written rows
        self._failed: bool = False
        self._classes: dict[str, type] = {}
        try:
            self._init_file()
            # The built-in tables come first, so that the parents are written before their children.
            self._tables: dict[TableName, _TableWriter] = {table: self._create_writer(table) for table in Table}
            # The tables of other classes are created when they're needed, except those that are already in the file.
            for name in self._file:
                if not is_builtin_table(name):
                    self._tables[validate_table_name(name)] = self._create_writer(name)
        except Exception:
            self._file.close()
            self._closed = True
            raise

    def _init_file(self) -> None:
        """Write the format attributes to a new file, or validate those of an existing file.

        :raises ExportFormatError: if the file is not a PTtools export file or has an incompatible format version
        """
        attrs = self._file.attrs
        if "format" not in attrs:
            if len(self._file):
                raise ExportFormatError(f"The file is not a PTtools export file: {self.path}")
            attrs["format"] = FORMAT_NAME
            attrs["format_version"] = FORMAT_VERSION
            attrs["pttools_version"] = _pttools_version()
            attrs["created"] = datetime.datetime.now(datetime.UTC).isoformat()
            attrs["h5py_version"] = h5py.version.version
            attrs["hdf5_version"] = h5py.version.hdf5_version
            return
        if attrs["format"] != FORMAT_NAME:
            raise ExportFormatError(f"The file is not a PTtools export file: {self.path}")
        if attrs["format_version"] != FORMAT_VERSION:
            raise ExportFormatError(
                f"Cannot append to a file of format version {attrs['format_version']}. "
                f"This version of PTtools writes version {FORMAT_VERSION}."
            )

    def _create_writer(self, table: TableName) -> _TableWriter:
        """Create the writer of a table, and the HDF5 group of the table if it does not exist yet."""
        return _TableWriter(
            self._file.require_group(str(table)), table,
            compression=self.compression, compression_opts=self.compression_opts
        )

    def _row_index(self, table: TableName, id_: str) -> int | None:
        """Row index of an object that has already been added, or None if it has not been added.

        This does not create the table, so that no empty tables are left in the file
        if adding the object fails before its data is buffered.
        """
        writer = self._tables.get(table)
        return None if writer is None else writer.index.get(id_)

    def _writer(self, table: TableName) -> _TableWriter:
        """The writer of the given table, which is created if it does not exist yet."""
        writer = self._tables.get(table)
        if writer is None:
            writer = self._tables[table] = self._create_writer(validate_table_name(str(table)))
        return writer

    # -----
    # Context manager
    # -----

    def __enter__(self) -> tp.Self:
        """Enter the context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Close the exporter when exiting the context manager."""
        self.close()

    # -----
    # Properties
    # -----

    @property
    def closed(self) -> bool:
        """Whether the exporter has been closed."""
        return self._closed

    @property
    def n_models(self) -> int:
        """Number of models, including the buffered ones."""
        return self._tables[Table.MODELS].n_total

    @property
    def n_bubbles(self) -> int:
        """Number of bubbles, including the buffered ones."""
        return self._tables[Table.BUBBLES].n_total

    @property
    def n_spectra_y(self) -> int:
        """Number of spectra that share $y$, including the buffered ones."""
        return self._tables[Table.SPECTRA_Y].n_total

    @property
    def n_spectra_f(self) -> int:
        """Number of spectra that share $f$, including the buffered ones."""
        return self._tables[Table.SPECTRA_F].n_total

    @property
    def tables(self) -> tuple[str, ...]:
        """Names of the tables, including the built-in tables and the tables of other classes."""
        return tuple(str(table) for table in self._tables)

    def n_rows(self, table: TableName) -> int:
        """Number of rows in the given table, including the buffered ones.

        :param table: the table
        :return: the number of rows, or 0 if the table does not exist
        """
        writer = self._tables.get(table)
        return 0 if writer is None else writer.n_total

    # -----
    # Adding data
    # -----

    def add(self, obj: Extractable | Record) -> int:
        """Add a model, a bubble, a spectrum, an object of another class that has a table, or a record to the file.

        The bubble and the model of a spectrum, and the model of a bubble, are added automatically.
        Objects that have already been added are skipped.

        :param obj: the object to add.
            Objects of other classes than the models, bubbles and spectra of PTtools must have a table,
            see :py:attr:`pttools.utils.fields.Extractable.TABLE`.
        :return: the row index of the object in its table
        """
        if self._closed:
            raise RuntimeError("The exporter has been closed.")
        if self._failed:
            raise RuntimeError("Writing to the file has failed, and therefore no more data can be added.")
        index = self._add_record(obj) if isinstance(obj, Record) else self._add_object(obj)
        if any(len(table.buffer) >= self.buffer_size for table in self._tables.values()):
            self.flush()
        return index

    def add_many(self, objs: Iterable[Extractable | Record]) -> list[int]:
        """Add multiple objects or records to the file, see :py:meth:`add`.

        :param objs: the objects to add
        :return: the row indices of the objects in their tables
        """
        return [self.add(obj) for obj in objs]

    def _add_object(self, obj: Extractable) -> int:
        """Add an object and its parents, unless they have already been added.

        The fields are extracted only for the objects that have not been added yet.

        :param obj: the object to add
        :return: the row index of the object in its table
        """
        table = table_of(obj)
        obj_id = object_id(obj)
        index = self._row_index(table, obj_id)
        if index is not None:
            return index
        parent = -1
        if isinstance(obj, SSMSpectrum):
            parent = self._add_object(obj.bubble)
        elif isinstance(obj, Bubble):
            parent = self._add_object(obj.model)
        fields = self.extractor.fields(type(obj), table)
        data = extract(obj, fields)
        return self._append(self._writer(table), obj_id, class_name(type(obj)), data, parent, fields)

    def _add_record(self, record: Record) -> int:
        """Add a record and its parent records, unless they have already been added.

        :param record: the record to add
        :return: the row index of the object in its table
        :raises ValueError: if the record has no parent when it should, or has one when it should not,
            if its class does not belong to its table, or if its fields do not match
        """
        index = self._row_index(record.table, object_id(record))
        if index is not None:
            return index
        parent = -1
        if parent_column(record.table) is not None:
            if record.parent is None:
                raise ValueError(f"The record of {record.cls} must have a parent record.")
            parent = self._add_record(record.parent)
        elif record.parent is not None:
            raise ValueError(f"The record of {record.cls} in the table \"{record.table}\" cannot have a parent record.")
        cls = self._classes.get(record.cls)
        if cls is None:
            cls = find_class(table_base_class(record.table), record.cls)
            self._classes[record.cls] = cls
        if not is_builtin_table(record.table) and record.table != cls.TABLE:
            raise ValueError(
                f"The record of {record.cls} is for the table \"{record.table}\", "
                f"but the table of the class is \"{cls.TABLE}\"."
            )
        fields = self.extractor.fields(cls, record.table)
        if set(record.data) != {field.name for field in fields}:
            raise ValueError(
                f"The fields of the record of {record.cls} do not match those of the exporter. "
                "Please create the records with Exporter.extractor."
            )
        return self._append(self._writer(record.table), record.id, record.cls, record.data, parent, fields)

    def _append(
            self,
            writer: _TableWriter,
            id_: str,
            cls: str,
            data: dict[str, tp.Any],
            parent: int,
            fields: tuple[Field, ...]) -> int:
        """Append the extracted data of an object to the buffer of its table.

        The fields of the models are converted to a JSON string.

        :param writer: the writer of the table
        :param id_: unique identifier of the object
        :param cls: fully qualified name of the class of the object
        :param data: the values of the fields
        :param parent: row index of the parent object, or -1 if there is none
        :param fields: the fields
        :return: the row index of the object in its table
        """
        if writer.table == Table.MODELS:
            # Different model classes have different fields, and therefore they are stored as JSON.
            data = {"class": cls, "params": json.dumps(data, default=_json_default)}
            fields = _MODEL_COLUMNS
        return writer.append(id_, cls, data, parent, fields)

    # -----
    # Writing
    # -----

    def flush(self) -> None:
        """Write the buffered rows to the file and commit them."""
        if self._closed or self._failed:
            return
        try:
            # The parents are written before the children, so that the children never refer to missing rows.
            # The built-in tables are the first in the dictionary, and are therefore written in the order of Table.
            for writer in self._tables.values():
                writer.write()
            for writer in self._tables.values():
                writer.commit()
            self._file.flush()
        except Exception:
            # Retrying could misalign the datasets.
            # The rows that were written but not committed are discarded when the file is opened again.
            self._failed = True
            raise

    def close(self) -> None:
        """Write the buffered rows, close the file and write the checksum file.

        If writing has failed, the buffered rows are discarded, and no checksum file is written.
        """
        if self._closed:
            return
        try:
            self.flush()
        finally:
            self._file.close()
            self._closed = True
        if self._failed:
            return
        logger.info(
            "Exported %d spectra with the same y, %d spectra with the same f, %d bubbles and %d models to %s",
            self.n_spectra_y, self.n_spectra_f, self.n_bubbles, self.n_models, self.path)
        for table, writer in self._tables.items():
            if not is_builtin_table(table):
                logger.info("Exported %d rows of the table \"%s\" to %s", writer.n_total, table, self.path)
        if self.checksum:
            write_checksum(self.path)
