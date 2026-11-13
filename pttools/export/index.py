r"""Index for finding the models, bubbles and spectra of an exported file by their parameter values.

The datasets computed on a parameter grid, e.g. by :py:func:`pttools.analysis.dataset.compute_dataset`,
contain a spectrum for each combination of the parameter values.
:py:class:`DatasetIndex` finds the distinct values of each parameter, and the rows that have the given values.
It's used by the interactive viewer of :py:mod:`pttools.plot.dataset`.

The rows are found hierarchically: the model by the parameters of the models,
the bubble by its model and the parameters of the bubbles,
and the spectrum by its bubble and the parameters of the spectra.
This way only the parent row indices of the spectra have to be kept in memory,
and the parameters of the spectra are read from the file only for the spectra of the selected bubble.
"""

from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import functools
import itertools
import logging
import math
import typing as tp

import numpy as np

from pttools.export.importer import Importer, _to_python
from pttools.export.records import Table, find_class
from pttools.models.base import BaseModel
from pttools.utils.fields import FieldShape, describe

__all__ = [
    "DEFAULT_BUBBLE_PARAMS",
    "DEFAULT_SPECTRUM_PARAMS",
    "Axis",
    "DatasetIndex",
    "Selection",
    "Value",
    "detect_axes",
]

logger: logging.Logger = logging.getLogger(__name__)

#: Number of rows that are read at once when going through a whole column of a table
CHUNK_ROWS: int = 1_000_000

#: The parameters that identify a bubble among the bubbles of its model
DEFAULT_BUBBLE_PARAMS: tuple[str, ...] = ("alpha_n", "v_wall")

#: The parameters that identify a spectrum among the spectra of its bubble
DEFAULT_SPECTRUM_PARAMS: tuple[str, ...] = ("beta_tilde", "T_star", "g_star", "nuc_type")

#: Maximum number of model parameters that :py:func:`detect_axes` combines
MAX_DETECTED_AXES: int = 4

#: Number of bubbles for which the parameters of their spectra are cached
SPECTRA_CACHE_SIZE: int = 32

#: Value of a parameter
type Value = float | int | str | bool


@dataclasses.dataclass(frozen=True, slots=True)
class Axis:
    r"""A parameter of a dataset and its distinct values.

    :param name: name of the parameter, i.e. of its field
    :param table: the table of the parameter
    :param values: the distinct values of the parameter in increasing order
    :param description: description of the parameter, e.g. ``"$v_\text{wall}$, wall speed"``
    """

    name: str
    table: Table
    values: tuple[Value, ...]
    description: str = ""

    @property
    def numeric(self) -> bool:
        """Whether the values are numbers, as opposed to e.g. the names of the nucleation types."""
        return all(_is_number(value) for value in self.values)


@dataclasses.dataclass(frozen=True, slots=True)
class Selection:
    """The rows of a combination of parameter values.

    :param model: row index of the model, or None if there is no model with the values
    :param bubble: row index of the bubble, or None if there is no bubble with the values
    :param spectrum: row index of the spectrum in the table of the index,
        or None if there is no spectrum with the values
    """

    model: int | None = None
    bubble: int | None = None
    spectrum: int | None = None


class DatasetIndex:
    r"""Index for finding the models, bubbles and spectra of an exported file by their parameter values.

    .. code-block:: python

        with Importer("dataset.h5") as importer:
            index = DatasetIndex(importer)
            values = {axis.name: axis.values[0] for axis in index.axes}
            selection = index.lookup(values)
            if selection.spectrum is not None:
                spectrum = importer.import_spectrum(selection.spectrum, table=index.table)

    Some combinations of the values may not have a bubble or a spectrum,
    e.g. as the spectra are not computed for the bubbles that have no fluid profile.
    The values are compared exactly, and therefore they should be taken from the :py:attr:`axes`.
    If several rows have the same values, the first one is used.

    :param importer: the importer of the file, which has to stay open while the index is used
    :param table: the table of the spectra. Defaults to
        :py:attr:`Table.SPECTRA_F <pttools.export.records.Table.SPECTRA_F>` if it has spectra,
        and to :py:attr:`Table.SPECTRA_Y <pttools.export.records.Table.SPECTRA_Y>` otherwise.
    :param model_params: names of the parameters that identify the models.
        If None, they are detected with :py:func:`detect_axes`.
    :param bubble_params: names of the fields that identify a bubble among the bubbles of its model.
        The fields that are not in the file are ignored.
    :param spectrum_params: names of the fields that identify a spectrum among the spectra of its bubble.
        The fields that are not in the file or have no values are ignored.
    :raises ValueError: if the file has no bubbles, or the models don't have the given parameters
    """

    def __init__(
            self,
            importer: Importer,
            table: Table | None = None,
            model_params: Iterable[str] | None = None,
            bubble_params: Iterable[str] = DEFAULT_BUBBLE_PARAMS,
            spectrum_params: Iterable[str] = DEFAULT_SPECTRUM_PARAMS) -> None:
        """Read the parameters of the models and bubbles, and the parents and the parameter values of the spectra."""
        if importer.n_bubbles == 0:
            raise ValueError(f"The file has no bubbles: {importer.path}")
        self.importer: Importer = importer
        if table is None:
            table = Table.SPECTRA_F if importer.n_spectra_f or not importer.n_spectra_y else Table.SPECTRA_Y
        #: The table of the spectra
        self.table: Table = Table(table)

        # Models
        model_rows = [importer.model_params(i) for i in range(importer.n_models)]
        model_names = detect_axes(model_rows) if model_params is None else tuple(model_params)
        missing = sorted({name for name in model_names for row in model_rows if name not in row})
        if missing:
            raise ValueError(f"Some of the models don't have the parameters: {missing}")
        descriptions = self.model_descriptions()
        #: The parameters of the models
        self.model_axes: tuple[Axis, ...] = tuple(
            Axis(name, Table.MODELS, _distinct(row[name] for row in model_rows), descriptions.get(name, ""))
            for name in model_names
        )
        self._models: dict[tuple[Value, ...], int] = _first_rows(
            (tuple(row[name] for name in model_names) for row in model_rows), "models")

        # Bubbles
        bubble_names = self._scalar_fields(Table.BUBBLES, bubble_params)
        bubble_cols = {name: _to_list(importer.read(Table.BUBBLES, name)) for name in bubble_names}
        #: The parameters of the bubbles
        self.bubble_axes: tuple[Axis, ...] = tuple(
            Axis(name, Table.BUBBLES, _distinct(bubble_cols[name]), self._description(Table.BUBBLES, name))
            for name in bubble_names
        )
        self._bubble_models: list[int] = _to_list(importer.parent_indices(Table.BUBBLES))
        self._bubbles: dict[tuple[Value, ...], int] = _first_rows(
            zip(self._bubble_models, *bubble_cols.values(), strict=True), "bubbles")

        # Spectra
        n_spectra = importer.n_rows(self.table)
        spectrum_axes: list[Axis] = []
        for name in self._scalar_fields(self.table, spectrum_params) if n_spectra else ():
            values = self._distinct_values(name)
            if values:
                spectrum_axes.append(Axis(name, self.table, values, self._description(self.table, name)))
            else:
                logger.info("The field \"%s\" of the table \"%s\" has no values. Ignoring it.", name, self.table)
        #: The parameters of the spectra
        self.spectrum_axes: tuple[Axis, ...] = tuple(spectrum_axes)
        #: The bubbles of the spectra in increasing order
        self._spectrum_bubbles: np.ndarray = self._read_parents(n_spectra)
        #: The rows of the spectra in the order of their bubbles, or None if they are already in that order
        self._spectrum_order: np.ndarray | None = None
        if n_spectra and np.any(self._spectrum_bubbles[1:] < self._spectrum_bubbles[:-1]):
            self._spectrum_order = np.argsort(self._spectrum_bubbles, kind="stable")
            self._spectrum_bubbles = self._spectrum_bubbles[self._spectrum_order]
        self._spectra_by_params = functools.lru_cache(maxsize=SPECTRA_CACHE_SIZE)(self._read_spectra_by_params)

    @property
    def axes(self) -> tuple[Axis, ...]:
        """The parameters of the models, bubbles and spectra."""
        return self.model_axes + self.bubble_axes + self.spectrum_axes

    def default_values(self) -> dict[str, Value]:
        """The parameter values of the first bubble that has not failed and has spectra, and of its first spectrum.

        If there is no such bubble, the first bubble that has spectra is used, and then the first bubble.

        :return: the values by the names of the parameters
        """
        importer = self.importer
        failed = importer.read(Table.BUBBLES, "failed") if "failed" in importer.fields(Table.BUBBLES) \
            else np.zeros(importer.n_bubbles, dtype=np.bool_)
        has_spectra = np.zeros(importer.n_bubbles, dtype=np.bool_)
        has_spectra[np.unique(self._spectrum_bubbles)] = True
        for candidates in (has_spectra & ~failed, has_spectra):
            if candidates.any():
                bubble = int(np.argmax(candidates))
                return self.values_of(bubble, int(self.spectra_of(bubble)[0]))
        return self.values_of(0)

    def lookup(self, values: Mapping[str, Value]) -> Selection:
        """Find the model, the bubble and the spectrum that have the given parameter values.

        :param values: the values of all the parameters of :py:attr:`axes` by their names
        :return: the rows, of which those that were not found are None
        """
        model = self._models.get(tuple(values[axis.name] for axis in self.model_axes))
        if model is None:
            return Selection()
        bubble = self._bubbles.get((model, *(values[axis.name] for axis in self.bubble_axes)))
        if bubble is None:
            return Selection(model=model)
        spectrum = self._spectra_by_params(bubble).get(tuple(values[axis.name] for axis in self.spectrum_axes))
        return Selection(model=model, bubble=bubble, spectrum=spectrum)

    def model_descriptions(self) -> dict[str, str]:
        """The descriptions of the parameters of the models by their names.

        These are taken from the fields of the classes of the models, as they are not stored in the file.
        """
        descriptions: dict[str, str] = {}
        for cls_name in dict.fromkeys(_to_list(self.importer.read(Table.MODELS, "class"))):
            try:
                cls = find_class(BaseModel, cls_name)
            except ValueError:
                continue
            for name, field in cls.FIELDS.items():
                descriptions.setdefault(name, describe(field, cls).description)
        return descriptions

    def spectra_of(self, bubble: int) -> np.ndarray:
        """Row indices of the spectra of a bubble in increasing order.

        :param bubble: row index of the bubble
        :return: row indices of the spectra in the table of the index
        """
        start, stop = np.searchsorted(self._spectrum_bubbles, [bubble, bubble + 1])
        if self._spectrum_order is None:
            return np.arange(start, stop)
        return self._spectrum_order[start:stop]

    def values_of(self, bubble: int, spectrum: int | None = None) -> dict[str, Value]:
        """The parameter values of a bubble and a spectrum.

        :param bubble: row index of the bubble
        :param spectrum: row index of the spectrum in the table of the index.
            If None, the first values of the parameters of the spectra are used.
        :return: the values by the names of the parameters
        """
        importer = self.importer
        model_params = importer.model_params(self._bubble_models[bubble])
        values: dict[str, Value] = {axis.name: model_params[axis.name] for axis in self.model_axes}
        values.update({axis.name: _to_python(importer.read(Table.BUBBLES, axis.name, bubble))
                       for axis in self.bubble_axes})
        for axis in self.spectrum_axes:
            values[axis.name] = axis.values[0] if spectrum is None \
                else _to_python(importer.read(self.table, axis.name, spectrum))
        return values

    def _description(self, table: Table, name: str) -> str:
        """The stored description of a field."""
        return self.importer.field_info(table, name).get("description", "")

    def _distinct_values(self, name: str) -> tuple[Value, ...]:
        """The distinct values of a field of the spectra, without NaN.

        The column is read in chunks, so that it does not have to fit in the memory at once.
        """
        n_rows = self.importer.n_rows(self.table)
        values: set[Value] = set()
        for start in range(0, n_rows, CHUNK_ROWS):
            chunk = self.importer.read(self.table, name, slice(start, min(start + CHUNK_ROWS, n_rows)))
            values.update(_to_list(np.unique(chunk)))
        return _distinct(values)

    def _read_parents(self, n_spectra: int) -> np.ndarray:
        """Read the row indices of the bubbles of the spectra in chunks to an array of the smallest sufficient type."""
        dtype = np.int32 if self.importer.n_bubbles <= np.iinfo(np.int32).max else np.int64
        parents = np.empty(n_spectra, dtype=dtype)
        for start in range(0, n_spectra, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, n_spectra)
            parents[start:stop] = self.importer.parent_indices(self.table, slice(start, stop))
        return parents

    def _read_spectra_by_params(self, bubble: int) -> dict[tuple[Value, ...], int]:
        """Read the parameters of the spectra of a bubble.

        :param bubble: row index of the bubble
        :return: the row indices of the spectra by their parameter values
        """
        rows = self.spectra_of(bubble)
        if rows.size == 0:
            return {}
        # A contiguous range is much faster to read than individual rows.
        contiguous = rows[-1] - rows[0] + 1 == rows.size
        selection = slice(int(rows[0]), int(rows[-1]) + 1) if contiguous else rows
        cols = [_to_list(self.importer.read(self.table, axis.name, selection)) for axis in self.spectrum_axes]
        if not cols:
            return {(): int(rows[0])}
        return _first_rows(zip(*cols, strict=True), f"spectra of the bubble {bubble}", rows=rows.tolist())

    def _scalar_fields(self, table: Table, names: Iterable[str]) -> tuple[str, ...]:
        """Those of the given fields that are scalar fields of the table."""
        fields = self.importer.fields(table)
        return tuple(
            name for name in names
            if name in fields and self.importer.field_info(table, name).get("kind") == FieldShape.SCALAR
        )


def detect_axes(rows: Sequence[Mapping[str, tp.Any]], max_axes: int = MAX_DETECTED_AXES) -> tuple[str, ...]:
    """Detect the parameters that identify the rows, e.g. the models of a dataset.

    The models of a dataset have also parameters that depend on the varied ones,
    e.g. $a_s$ and $V_s$ that are adjusted by the model for each pair of sound speeds.
    The values of the varied parameters repeat on the grid, whereas the dependent ones have more distinct values.
    Therefore, of the sets of numeric parameters whose values are different for each row,
    the one with the smallest total number of distinct values is selected.
    Of equally good sets, the one with the parameters that are later in the rows is selected,
    as the parameters that are specific to the class of the models come after the general ones.

    This is a heuristic: if only one parameter is varied, a parameter that depends on it identifies the rows
    equally well. The parameters can then be given explicitly to :py:class:`DatasetIndex`.

    :param rows: the parameters of each row by their names
    :param max_axes: maximum number of parameters in the set
    :return: names of the parameters in the order of the rows. Empty if there is at most one row.
        If no set identifies the rows, all the numeric parameters that have several distinct values.
    """
    if len(rows) <= 1:
        return ()
    counts: dict[str, int] = {}
    for name in rows[0]:
        if all(name in row and _is_number(row[name]) and math.isfinite(row[name]) for row in rows):
            count = len({row[name] for row in rows})
            if count > 1:
                counts[name] = count
    best: tuple[int, tuple[str, ...]] | None = None
    # The later parameters are tried first, so that they are selected of equally good sets.
    names_reversed = tuple(reversed(counts))
    for size in range(1, min(max_axes, len(counts)) + 1):
        for names in itertools.combinations(names_reversed, size):
            total = sum(counts[name] for name in names)
            if (best is None or total < best[0]) \
                    and len({tuple(row[name] for name in names) for row in rows}) == len(rows):
                best = (total, names)
    if best is None:
        return tuple(counts)
    return tuple(reversed(best[1]))


def _distinct(values: Iterable[Value]) -> tuple[Value, ...]:
    """The distinct values in increasing order, without NaN."""
    return tuple(sorted({value for value in values if value == value}))  # noqa: PLR0124


def _first_rows(
        keys: Iterable[tuple[Value, ...]],
        what: str,
        rows: Sequence[int] | None = None) -> dict[tuple[Value, ...], int]:
    """Map the keys to the row indices of their first occurrences.

    :param keys: the keys of the rows
    :param what: description of the rows for the log message of duplicate keys
    :param rows: the row indices. Defaults to the indices of the keys.
    :return: the row indices by the keys
    """
    result: dict[tuple[Value, ...], int] = {}
    n_duplicates = 0
    for i, key in enumerate(keys):
        if key in result:
            n_duplicates += 1
        else:
            result[key] = i if rows is None else rows[i]
    if n_duplicates:
        logger.warning(
            "%d of the %s have the same parameter values as another one. Using the first of each.", n_duplicates, what)
    return result


def _is_number(value: tp.Any) -> bool:
    """Whether the value is an integer or a float, but not a boolean."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _to_list(values: tp.Any) -> list[tp.Any]:
    """Convert an array to a list of Python objects."""
    return np.asarray(values).tolist()
