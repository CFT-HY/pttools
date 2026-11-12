r"""Bubbles imported from a file without solving them again.

The bubbles are created by :py:meth:`pttools.export.importer.Importer.import_bubble`.
"""

from __future__ import annotations

from collections.abc import Mapping
import typing as tp
import uuid

from pttools.bubble.bubble.nucleated import BaseNucleatedBubble
from pttools.bubble.export import BUBBLE_FIELDS
from pttools.export.imported import Imported
import pttools.type_hints as th

if th.TYPE_CHECKING_OR_DOCS:
    from pttools.models.model import Model
    from pttools.utils.fields import Fields

__all__ = ["ImportedBubble"]


class ImportedBubble(Imported, BaseNucleatedBubble):
    r"""A bubble whose solution is imported from a file without solving it again.

    The stored fields are available as attributes, see :py:class:`pttools.export.imported.Imported`,
    and the quantities of :py:class:`pttools.bubble.bubble.nucleated.BaseNucleatedBubble`,
    such as $\kappa$, $\omega$ and $\bar{U}_f$, are computed from the stored fluid profiles and parameters
    unless their values are stored.
    The fields that are needed for this are included in the
    :py:attr:`~pttools.utils.fields.Preset.MINIMAL` preset.
    Unlike :py:class:`pttools.bubble.bubble.Bubble`, the bubble cannot be solved again.

    :param model: the equation of state, e.g. from :py:meth:`pttools.export.importer.Importer.load_model`
    :param data: the stored values of the fields by their names
    :param fields: the fields of the class of the stored bubble
    :param source: description of the source of the data for the error messages, e.g. ``"bubble 3 of data.h5"``
    """

    def __init__(
            self,
            model: Model,
            data: Mapping[str, tp.Any],
            fields: Fields = BUBBLE_FIELDS,
            source: str = "the file") -> None:
        """Set the attributes from the stored data."""
        self.model: Model = model
        self.id: str = str(data.get("id") or uuid.uuid4().hex)
        self._import_fields(data, fields, source)
        # The notes are stored one per line.
        notes = self.__dict__.get("notes", "")
        self.notes: list[str] = notes.splitlines() if isinstance(notes, str) else list(notes)
        # A bubble whose profiles were exported had been solved, unless the solver crashed.
        self.solved: bool = "v" in data and not self.__dict__.get("solver_crashed", False)
        self.solving_attempted: bool = True
        labels = self.default_labels(model, self.v_wall, self.alpha_n)
        self.label_latex: str = labels[0]
        self.label_unicode: str = labels[1]
