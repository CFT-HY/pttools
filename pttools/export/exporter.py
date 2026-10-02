r"""Exporter for writing models, bubbles and spectra to an HDF5 file.

File layout
-----------
The file has a group for each :py:class:`~pttools.export.records.Table`.
Each group has a dataset for each field, so that the values of a field can be read at once,
e.g. for training machine learning models.

.. code-block:: text

    /                         attrs: format, format_version, pttools_version, created
    /models/id                (N_models,)        unique identifiers
    /models/class             (N_models,)        fully qualified class names
    /models/params            (N_models,)        the fields of each model as a JSON string
    /bubbles/id               (N_bubbles,)
    /bubbles/model            (N_bubbles,)       row index in /models
    /bubbles/v_wall           (N_bubbles,)       a scalar field
    /bubbles/v                (sum of lengths,)  a ragged field: the profiles of all bubbles concatenated
    /bubbles/xi_offsets       (N_bubbles + 1,)   the profile of bubble i is v[xi_offsets[i]:xi_offsets[i+1]]
    /spectra/id               (N_spectra,)
    /spectra/bubble           (N_spectra,)       row index in /bubbles
    /spectra/r_star           (N_spectra,)       a scalar field
    /spectra/omgw0_h2         (N_spectra, n_y)   an array field
    /spectra/y                (n_y,)             a grid field, which is the same for all spectra

The model parameters are stored as JSON, since different model classes have different parameters.
The groups have the attributes ``n_rows`` (the number of committed rows) and ``fields`` (the names of the fields).
The bubbles and spectra groups also have the attribute ``class``,
as all the bubbles and all the spectra of a file must be of the same class.
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
from pttools.export.records import Extractor, Record, Table, class_name, find_class, table_base_class, table_of
from pttools.models.base import BaseModel
from pttools.ssm.spectrum import SSMSpectrum
from pttools.utils.fields import Field, FieldShape, FieldSpec, FieldType, Preset, extract

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
    Table.SPECTRA: "bubble",
}


class ExportFormatError(ValueError):
    """The file is not a valid PTtools export file, or its contents do not match the exporter settings."""


def offsets_name(axis: str) -> str:
    """Name of the dataset that contains the offsets of the ragged fields of the given axis."""
    return f"{axis}_offsets"


def parent_column(table: Table) -> str | None:
    """Name of the dataset that contains the row indices of the parents, if any."""
    return _PARENT_COLUMNS.get(table)


def _pttools_version() -> str:
    try:
        return importlib.metadata.version("pttools-gw")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _json_default(value: tp.Any) -> tp.Any:
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
    # The values are converted to plain strings, since h5py does not support StrEnum attributes.
    dset.attrs["kind"] = str(kind)
    dset.attrs["type"] = str(field_type)
    dset.attrs["axis"] = str(axis)
    dset.attrs["description"] = str(description)


@dataclasses.dataclass(slots=True)
class _Row:
    id: str
    parent: int
    data: dict[str, tp.Any]


class _TableWriter:
    """Writer for a single table, i.e. an HDF5 group."""

    def __init__(self, group: h5py.Group, table: Table, compression: str | None, compression_opts: int | None) -> None:
        self.group: h5py.Group = group
        self.table: Table = table
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

        def compute(params):
            bubble = Bubble(model, v_wall=params[0], alpha_n=params[1])
            return extractor.extract(Spectrum(bubble, r_star=params[2]))

        with Exporter("spectra.h5") as exporter:
            extractor = exporter.extractor
            with ProcessPoolExecutor() as executor:
                exporter.add_many(executor.map(compute, params))

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
            checksum: bool = True) -> None:
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
            importable=importable
        )
        self.buffer_size: int = buffer_size
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
            self._tables: dict[Table, _TableWriter] = {
                table: _TableWriter(
                    self._file.require_group(table.value), table,
                    compression=compression, compression_opts=compression_opts
                )
                for table in Table
            }
        except Exception:
            self._file.close()
            self._closed = True
            raise

    def _init_file(self) -> None:
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

    # -----
    # Context manager
    # -----

    def __enter__(self) -> tp.Self:
        return self

    def __exit__(self, *args: object) -> None:
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
    def n_spectra(self) -> int:
        """Number of spectra, including the buffered ones."""
        return self._tables[Table.SPECTRA].n_total

    # -----
    # Adding data
    # -----

    def add(self, obj: BaseModel | Bubble | SSMSpectrum | Record) -> int:
        """Add a model, a bubble, a spectrum or a record to the file.

        The bubble and the model of a spectrum, and the model of a bubble, are added automatically.
        Objects that have already been added are skipped.

        :param obj: the object to add
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

    def add_many(self, objs: Iterable[BaseModel | Bubble | SSMSpectrum | Record]) -> list[int]:
        """Add multiple objects or records to the file, see :py:meth:`add`.

        :param objs: the objects to add
        :return: the row indices of the objects in their tables
        """
        return [self.add(obj) for obj in objs]

    def _add_object(self, obj: BaseModel | Bubble | SSMSpectrum) -> int:
        table = table_of(obj)
        writer = self._tables[table]
        if obj.id in writer.index:
            return writer.index[obj.id]
        parent = -1
        if isinstance(obj, SSMSpectrum):
            parent = self._add_object(obj.bubble)
        elif isinstance(obj, Bubble):
            parent = self._add_object(obj.model)
        fields = self.extractor.fields(type(obj), table)
        return self._append(writer, obj.id, class_name(type(obj)), extract(obj, fields), parent, fields)

    def _add_record(self, record: Record) -> int:
        writer = self._tables[record.table]
        if record.id in writer.index:
            return writer.index[record.id]
        parent = -1
        if record.table != Table.MODELS:
            if record.parent is None:
                raise ValueError(f"The record of {record.cls} must have a parent record.")
            parent = self._add_record(record.parent)
        cls = self._classes.get(record.cls)
        if cls is None:
            cls = find_class(table_base_class(record.table), record.cls)
            self._classes[record.cls] = cls
        fields = self.extractor.fields(cls, record.table)
        if set(record.data) != {field.name for field in fields}:
            raise ValueError(
                f"The fields of the record of {record.cls} do not match those of the exporter. "
                "Please create the records with Exporter.extractor."
            )
        return self._append(writer, record.id, record.cls, record.data, parent, fields)

    def _append(
            self,
            writer: _TableWriter,
            id_: str,
            cls: str,
            data: dict[str, tp.Any],
            parent: int,
            fields: tuple[Field, ...]) -> int:
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
            for table in Table:
                self._tables[table].write()
            for table in Table:
                self._tables[table].commit()
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
            "Exported %d spectra, %d bubbles and %d models to %s",
            self.n_spectra, self.n_bubbles, self.n_models, self.path)
        if self.checksum:
            write_checksum(self.path)
