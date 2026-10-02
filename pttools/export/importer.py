"""Importer for reading the files written by :py:class:`pttools.export.exporter.Exporter`.

The importer can read the exported fields as Numpy arrays, e.g. for training machine learning models,
and recreate the :py:class:`~pttools.models.model.Model`, :py:class:`~pttools.bubble.bubble.Bubble`
and :py:class:`~pttools.ssm.spectrum.SSMSpectrum` objects,
if the file contains the fields of the :py:attr:`~pttools.utils.fields.Preset.INIT` preset.
The recreated objects are computed again from their parameters,
and therefore their results match the stored ones only if the PTtools version is the same.
"""

from collections.abc import Iterable, Mapping, Sequence
import inspect
import json
import os
from pathlib import Path
import typing as tp

import h5py
import numpy as np

from pttools.bubble.bubble import Bubble
from pttools.export.checksum import verify_checksum
from pttools.export.exporter import FORMAT_NAME, FORMAT_VERSION, ExportFormatError, offsets_name, parent_column
from pttools.export.records import Table, find_class
from pttools.models.model import Model
from pttools.ssm.spectrum import SSMSpectrum
from pttools.utils.fields import Extractable, FieldShape, Preset

__all__ = [
    "Importer",
    "Rows",
]

#: Row selection: a row index, a slice, a sequence of row indices, or None for all the rows
type Rows = int | slice | Sequence[int] | np.ndarray | None


def _to_python(value: tp.Any) -> tp.Any:
    """Convert Numpy scalars and bytes to Python objects."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _init_kwargs(cls: type[Extractable], values: Mapping[str, tp.Any], what: str) -> dict[str, tp.Any]:
    """Get the constructor arguments for recreating an object from the stored fields.

    :param cls: the class of the object
    :param values: the stored values of the fields
    :param what: description of the object for the error messages
    :return: the constructor arguments
    """
    init_fields = cls.FIELDS.preset(Preset.INIT)
    missing = [field.name for field in init_fields if field.name not in values]
    if missing:
        raise ValueError(
            f"The file does not contain the parameters that are needed for recreating the {what}: {missing}. "
            "Please export the data with importable=True."
        )
    params = inspect.signature(cls).parameters
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
    kwargs: dict[str, tp.Any] = {}
    for field in init_fields:
        if field.name in params or accepts_kwargs:
            value = _to_python(values[field.name])
            kwargs[field.name] = value if field.decode is None else field.decode(value)
    return kwargs


class Importer:
    """Importer for reading the files written by :py:class:`pttools.export.exporter.Exporter`.

    .. code-block:: python

        with Importer("spectra.h5", verify=True) as importer:
            # Arrays for machine learning
            params = importer.read_scalars(Table.SPECTRA)
            omgw0_h2 = importer.read(Table.SPECTRA, "omgw0_h2")
            y = importer.read(Table.SPECTRA, "y")
            # Recreating the objects
            spectra = importer.load_spectra([0, 1])

    The recreated objects are cached, so that the spectra of the same bubble share the same bubble object,
    and the bubbles of the same model share the same model object.

    :param path: path of the HDF5 file
    :param verify: whether to verify the file against its SHA-256 checksum file before opening it
    :raises pttools.export.checksum.ChecksumError: if the verification fails
    :raises FileNotFoundError: if the verification is requested but the checksum file does not exist
    """

    def __init__(self, path: str | os.PathLike[str], verify: bool = False) -> None:
        self.path: Path = Path(path)
        if verify:
            verify_checksum(self.path, raise_error=True)
        self._file: h5py.File = h5py.File(self.path, "r")
        try:
            if self._file.attrs.get("format") != FORMAT_NAME:
                raise ExportFormatError(f"The file is not a PTtools export file: {self.path}")
            if self._file.attrs["format_version"] > FORMAT_VERSION:
                raise ExportFormatError(
                    f"The file has the format version {self._file.attrs['format_version']}, "
                    f"but this version of PTtools supports only versions up to {FORMAT_VERSION}. "
                    "Please update PTtools."
                )
        except Exception:
            self._file.close()
            raise
        self._models: dict[int, Model] = {}
        self._bubbles: dict[int, Bubble] = {}

    # -----
    # Context manager
    # -----

    def __enter__(self) -> tp.Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the file."""
        self._file.close()

    # -----
    # Metadata
    # -----

    @property
    def attrs(self) -> dict[str, tp.Any]:
        """Attributes of the file, e.g. the PTtools version used for creating it."""
        return {key: _to_python(value) for key, value in self._file.attrs.items()}

    def _group(self, table: Table | str) -> h5py.Group:
        return self._file[Table(table).value]

    def n_rows(self, table: Table | str) -> int:
        """Number of rows in the given table."""
        return int(self._group(table).attrs.get("n_rows", 0))

    @property
    def n_models(self) -> int:
        """Number of models."""
        return self.n_rows(Table.MODELS)

    @property
    def n_bubbles(self) -> int:
        """Number of bubbles."""
        return self.n_rows(Table.BUBBLES)

    @property
    def n_spectra(self) -> int:
        """Number of spectra."""
        return self.n_rows(Table.SPECTRA)

    def class_name(self, table: Table | str) -> str | None:
        """Fully qualified name of the class of the bubbles or spectra."""
        cls = self._group(table).attrs.get("class")
        return None if cls is None else str(cls)

    def fields(self, table: Table | str) -> tuple[str, ...]:
        """Names of the fields of the given table."""
        group = self._group(table)
        return tuple(str(name) for name in group.attrs["fields"]) if "fields" in group.attrs else ()

    def field_info(self, table: Table | str, name: str) -> dict[str, str]:
        """Metadata of a field, i.e. its kind, type, axis and description."""
        return {key: str(_to_python(value)) for key, value in self._group(table)[name].attrs.items()}

    # -----
    # Reading data
    # -----

    def _indices(self, rows: Rows, n: int) -> np.ndarray:
        if rows is None:
            return np.arange(n)
        if isinstance(rows, slice):
            return np.arange(n)[rows]
        indices = np.atleast_1d(np.asarray(rows, dtype=np.int64))
        indices = np.where(indices < 0, indices + n, indices)
        if indices.size and (indices.min() < 0 or indices.max() >= n):
            raise IndexError(f"Row indices out of range for {n} rows: {rows}")
        return indices

    @staticmethod
    def _read_dataset(dset: h5py.Dataset, rows: Rows, n: int) -> np.ndarray:
        """Read the given rows of a dataset efficiently."""
        if isinstance(rows, (int, np.integer)):
            index = int(rows) + n if rows < 0 else int(rows)
            if not 0 <= index < n:
                raise IndexError(f"Row index {rows} is out of range for {n} rows.")
            return dset[index]
        if rows is None:
            return dset[:n]
        if isinstance(rows, slice):
            start, stop, step = rows.indices(n)
            if step > 0:
                return dset[start:stop][::step]
            return dset[:n][rows]
        indices = np.asarray(rows, dtype=np.int64)
        indices = np.where(indices < 0, indices + n, indices)
        if indices.size == 0:
            return dset[0:0]
        if indices.min() < 0 or indices.max() >= n:
            raise IndexError(f"Row indices out of range for {n} rows.")
        # HDF5 requires the indices to be unique and increasing.
        unique, inverse = np.unique(indices, return_inverse=True)
        return dset[unique][inverse]

    def read(self, table: Table | str, name: str, rows: Rows = None) -> tp.Any:
        """Read a field or a structural column of a table.

        - Scalar fields are returned as 1D arrays (or as a single value for a single row).
        - Array fields are returned as 2D arrays of the shape (rows, length).
        - Grid fields are returned as 1D arrays regardless of the rows.
        - Ragged fields are returned as lists of 1D arrays (or as a single array for a single row).
        - Strings are returned as arrays of Python strings.

        The structural columns are ``id``, the parent row index ``model`` or ``bubble``,
        and for models ``class`` and ``params``.

        :param table: the table
        :param name: name of the field
        :param rows: the rows to read, see :py:data:`Rows`
        :return: the values
        """
        group = self._group(table)
        if name not in group:
            raise KeyError(f"The table \"{table}\" has no field \"{name}\". Available: {self.fields(table)}")
        dset = group[name]
        n = self.n_rows(table)
        kind = dset.attrs.get("kind")
        if kind == FieldShape.GRID:
            return dset[()]
        if kind == FieldShape.RAGGED:
            return self._read_ragged(dset, group[offsets_name(dset.attrs["axis"])], rows, n)
        values = self._read_dataset(dset, rows, n)
        if h5py.check_string_dtype(dset.dtype) is None:
            return values
        if isinstance(values, (bytes, str)):
            return _to_python(values)
        if dset.dtype.kind == "O":
            return np.array([_to_python(value) for value in np.atleast_1d(values)], dtype=object)
        return np.char.decode(values, "ascii")

    def _read_ragged(
            self, dset: h5py.Dataset, offsets: h5py.Dataset, rows: Rows, n: int) -> np.ndarray | list[np.ndarray]:
        if isinstance(rows, (int, np.integer)):
            index = self._indices(rows, n)[0]
            return dset[offsets[index]:offsets[index + 1]]
        indices = self._indices(rows, n)
        if indices.size == 0:
            return []
        offs = offsets[:n + 1]
        # Read the covered range at once and split it, since reading many small slices is slow.
        start = offs[indices.min()]
        data = dset[start:offs[indices.max() + 1]]
        return [data[offs[i] - start:offs[i + 1] - start] for i in indices]

    def read_scalars(
            self,
            table: Table | str,
            names: Iterable[str] | None = None,
            rows: Rows = None) -> dict[str, np.ndarray]:
        """Read the scalar fields of a table.

        The result can be converted to a Pandas DataFrame with ``pandas.DataFrame(result)``.

        :param table: the table
        :param names: names of the fields. If None, all the scalar fields are read.
        :param rows: the rows to read, see :py:data:`Rows`
        :return: the values of the fields
        """
        group = self._group(table)
        if names is None:
            names = [name for name in self.fields(table) if group[name].attrs.get("kind") == FieldShape.SCALAR]
        return {name: self.read(table, name, rows) for name in names}

    def model_params(self, index: int) -> dict[str, tp.Any]:
        """The stored fields of a model."""
        return json.loads(self.read(Table.MODELS, "params", index))

    def parent_indices(self, table: Table | str, rows: Rows = None) -> np.ndarray:
        """Row indices of the models of bubbles or of the bubbles of spectra."""
        column = parent_column(Table(table))
        if column is None:
            raise ValueError("Models have no parents.")
        return self.read(table, column, rows)

    # -----
    # Recreating objects
    # -----

    def _row_values(self, table: Table, cls: type[Extractable], index: int) -> dict[str, tp.Any]:
        """Read the stored fields that are needed for recreating an object."""
        available = set(self.fields(table))
        return {
            field.name: self.read(table, field.name, index)
            for field in cls.FIELDS.preset(Preset.INIT) if field.name in available
        }

    def load_model(self, index: int) -> Model:
        """Recreate a model.

        :param index: row index of the model
        :return: the model
        """
        index = int(self._indices(index, self.n_models)[0])
        if index in self._models:
            return self._models[index]
        cls = find_class(Model, self.read(Table.MODELS, "class", index))
        model = cls(**_init_kwargs(cls, self.model_params(index), f"model {index} ({cls.__name__})"))
        self._models[index] = model
        return model

    def load_bubble(self, index: int, solve: bool = True, **kwargs: tp.Any) -> Bubble:
        """Recreate a bubble.

        :param index: row index of the bubble
        :param solve: whether to solve the bubble
        :param kwargs: additional arguments for the constructor of the bubble, e.g. ``allow_invalid``
        :return: the bubble
        """
        index = int(self._indices(index, self.n_bubbles)[0])
        bubble = self._bubbles.get(index)
        if bubble is None:
            cls_name = self.class_name(Table.BUBBLES)
            if cls_name is None:
                raise ValueError("The file contains no bubbles.")
            cls = find_class(Bubble, cls_name)
            model = self.load_model(int(self.read(Table.BUBBLES, "model", index)))
            init = _init_kwargs(cls, self._row_values(Table.BUBBLES, cls, index), f"bubble {index}")
            bubble = cls(model, **init, solve=False, **kwargs)
            self._bubbles[index] = bubble
        if solve and not bubble.solved:
            bubble.solve()
        return bubble

    def load_spectrum(self, index: int, compute: bool = True, **kwargs: tp.Any) -> SSMSpectrum:
        """Recreate a spectrum.

        :param index: row index of the spectrum
        :param compute: whether to compute the spectrum
        :param kwargs: additional arguments for the constructor of the spectrum, e.g. ``parallel``
        :return: the spectrum
        """
        index = int(self._indices(index, self.n_spectra)[0])
        cls_name = self.class_name(Table.SPECTRA)
        if cls_name is None:
            raise ValueError("The file contains no spectra.")
        cls = find_class(SSMSpectrum, cls_name)
        bubble = self.load_bubble(int(self.read(Table.SPECTRA, "bubble", index)), solve=False)
        init = _init_kwargs(cls, self._row_values(Table.SPECTRA, cls, index), f"spectrum {index}")
        # r_star is computed from beta_tilde, and giving both would be an error.
        if init.get("beta_tilde") is not None:
            init["r_star"] = None
        return cls(bubble, **init, compute=compute, **kwargs)

    def load_bubbles(self, rows: Rows = None, solve: bool = True, **kwargs: tp.Any) -> list[Bubble]:
        """Recreate multiple bubbles, see :py:meth:`load_bubble`."""
        return [self.load_bubble(int(i), solve=solve, **kwargs) for i in self._indices(rows, self.n_bubbles)]

    def load_models(self, rows: Rows = None) -> list[Model]:
        """Recreate multiple models, see :py:meth:`load_model`."""
        return [self.load_model(int(i)) for i in self._indices(rows, self.n_models)]

    def load_spectra(self, rows: Rows = None, compute: bool = True, **kwargs: tp.Any) -> list[SSMSpectrum]:
        """Recreate multiple spectra, see :py:meth:`load_spectrum`."""
        return [self.load_spectrum(int(i), compute=compute, **kwargs) for i in self._indices(rows, self.n_spectra)]
