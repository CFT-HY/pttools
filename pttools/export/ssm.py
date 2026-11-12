r"""Sound Shell Model spectra imported from a file without computing them again.

The spectra are created by :py:meth:`pttools.export.importer.Importer.import_spectrum`.
"""

from __future__ import annotations

from collections.abc import Mapping
import typing as tp
import uuid

from pttools.export.imported import Imported
from pttools.ssm.export import SSM_SPECTRUM_FIELDS
from pttools.ssm.spectrum_base_ssm import BaseSSMSpectrum
import pttools.type_hints as th

if th.TYPE_CHECKING_OR_DOCS:
    from pttools.export.bubble import ImportedBubble
    from pttools.utils.fields import Fields

__all__ = ["ImportedSSMSpectrum"]


class ImportedSSMSpectrum(Imported, BaseSSMSpectrum):
    r"""A Sound Shell Model spectrum imported from a file without computing it again.

    The stored fields are available as attributes, see :py:class:`pttools.export.imported.Imported`,
    and the quantities of :py:class:`pttools.ssm.spectrum_base_ssm.BaseSSMSpectrum` are computed from them,
    e.g. $\mathcal{H}_* \eta_\text{sh}$ from $r_*$ and $\bar{U}_f$ of the bubble.
    The spectra that need the spectral densities, e.g. :py:attr:`pow_U_par`,
    are available only if the spectral densities are stored,
    e.g. with the :py:attr:`~pttools.utils.fields.Preset.FULL` preset.
    The spectrum cannot be computed again or reused with ``with_params()``.

    :param bubble: the bubble of the spectrum, e.g. from :py:meth:`pttools.export.importer.Importer.import_bubble`
    :param data: the stored values of the fields by their names
    :param fields: the fields of the class of the stored spectrum
    :param source: description of the source of the data for the error messages, e.g. ``"spectrum 3 of data.h5"``
    """

    def __init__(
            self,
            bubble: ImportedBubble,
            data: Mapping[str, tp.Any],
            fields: Fields = SSM_SPECTRUM_FIELDS,
            source: str = "the file") -> None:
        """Set the attributes from the stored data."""
        self._bubble: ImportedBubble = bubble
        self.id: str = str(data.get("id") or uuid.uuid4().hex)
        # The overrides of the equation of state parameters are not exported, and the bubble is used instead.
        self._nu_gdh2024: float | None = None
        self._omega_barotropic: float | None = None
        self._import_fields(data, fields, source)
        if "label_latex" not in data or "label_unicode" not in data:
            labels = self.default_labels(bubble, self.r_star)
            self.label_latex: str = labels[0]
            self.label_unicode: str = labels[1]

    @property
    @tp.override
    def bubble(self) -> ImportedBubble:
        """The bubble of the spectrum."""
        return self._bubble
