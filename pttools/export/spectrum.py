r"""Spectra with the GW power spectrum today imported from a file without computing them again.

The spectra are created by :py:meth:`pttools.export.importer.Importer.import_spectrum`.
"""

from __future__ import annotations

from collections.abc import Mapping
import typing as tp

import numpy as np

from pttools.export.ssm import ImportedSSMSpectrum
from pttools.omgw0.base_ssm import BaseSSMOmgw0
from pttools.omgw0.export import SPECTRUM_FIELDS
import pttools.type_hints as th
from pttools.utils.fields import FieldShape

if th.TYPE_CHECKING_OR_DOCS:
    from pttools.export.bubble import ImportedBubble
    from pttools.utils.fields import Fields

__all__ = ["ImportedSpectrum"]


class ImportedSpectrum(BaseSSMOmgw0, ImportedSSMSpectrum):
    r"""A spectrum with the GW power spectrum today $\Omega_{\text{gw},0}$ imported from a file.

    This is the imported counterpart of :py:class:`pttools.omgw0.spectrum.Spectrum`,
    see :py:class:`pttools.export.ssm.ImportedSSMSpectrum`.
    The spectra that have been given the frequencies $f$ have the stored frequencies,
    and their $y$ array is computed from them unless it's stored.
    If $\mathcal{P}_\text{gw}$ is not stored, it's computed from the stored $\Omega_{\text{gw},0} h^2$
    with the default values of the constants of :py:meth:`F_gw0_h2`,
    so that $\Omega_{\text{gw},0} h^2$, the signal-to-noise ratio etc. can be computed with any constants.

    :param bubble: the bubble of the spectrum, e.g. from :py:meth:`pttools.export.importer.Importer.import_bubble`
    :param data: the stored values of the fields by their names
    :param fields: the fields of the class of the stored spectrum,
        e.g. :py:attr:`pttools.omgw0.spectrum.Spectrum.FIELDS_F` for the spectra that share $f$
    :param source: description of the source of the data for the error messages, e.g. ``"spectrum 3 of data.h5"``
    """

    def __init__(
            self,
            bubble: ImportedBubble,
            data: Mapping[str, tp.Any],
            fields: Fields = SPECTRUM_FIELDS,
            source: str = "the file") -> None:
        """Set the attributes from the stored data."""
        super().__init__(bubble, data, fields=fields, source=source)
        # The frequencies are stored as the grid of the spectra that have been given them,
        # and they are returned by f() as is, as for the computed spectra.
        f_given = "f" in data and fields["f"].shape == FieldShape.GRID
        self._f_given = np.asarray(data["f"]) if f_given else None
        if "y" not in self.__dict__:
            if self._f_given is None:
                # The y array is a field of all the spectra, and therefore it's missing only from incomplete data.
                return
            self.y: th.FloatArr1D = self.z_from_f(self._f_given)
        if "pow_gw" not in self.__dict__ and "omgw0_h2" in data:
            self.__dict__["pow_gw"] = np.asarray(data["omgw0_h2"]) / self.F_gw0_h2()
