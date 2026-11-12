r"""Base class of the SSM spectra with the GW power spectrum today $\Omega_{\text{gw},0}$."""

from __future__ import annotations

import abc
import functools
import typing as tp

from pttools.omgw0 import const
from pttools.omgw0.base import BaseOmgw0
from pttools.omgw0.export import SPECTRUM_FIELDS, SPECTRUM_FIELDS_F, SPECTRUM_FIELDS_Y
from pttools.omgw0.factors import H
from pttools.ssm.spectrum_base_ssm import BaseSSMSpectrum
import pttools.type_hints as th
from pttools.utils.decorators import cached_property

if th.TYPE_CHECKING_OR_DOCS:

    from pttools.utils.fields import Fields


__all__ = ["BaseSSMOmgw0"]


class BaseSSMOmgw0(BaseOmgw0, BaseSSMSpectrum, abc.ABC):
    r"""Common base class of the SSM spectra with the GW power spectrum today $\Omega_{\text{gw},0}$.

    This is shared by the computed :py:class:`Spectrum` and the imported
    :py:class:`pttools.export.spectrum.ImportedSpectrum`.
    """

    FIELDS: tp.ClassVar[Fields] = SPECTRUM_FIELDS
    #: The exportable fields of the spectra that have been given the $y$ array.
    #: For these, $y$ is shared by the spectra of a file.
    #: This is an alias of :py:attr:`FIELDS` for consistency with :py:attr:`FIELDS_F`.
    #: Subclasses that change :py:attr:`FIELDS` should also change this and :py:attr:`FIELDS_F`.
    FIELDS_Y: tp.ClassVar[Fields] = SPECTRUM_FIELDS_Y
    #: The exportable fields of the spectra that have been given the frequencies $f$ instead of $y$.
    #: For these, $f$ is shared by the spectra of a file instead of $y$.
    FIELDS_F: tp.ClassVar[Fields] = SPECTRUM_FIELDS_F

    # The frequencies given to the constructor, which are returned by f() as is, or None if y was given
    _f_given: th.FloatArr1D | None

    @property
    def f_given(self) -> bool:
        """Whether the frequencies $f$ were given as an argument instead of the $y$ array."""
        return self._f_given is not None

    @cached_property
    @tp.override
    def _f_y(self) -> th.FloatArr1D:
        # The frequencies given as an argument are returned as is, instead of converting them back from y.
        if self._f_given is not None:
            return self._f_given
        return self.f(self.y)

    @cached_property
    @tp.override
    def H_star(self) -> float:
        return float(H(self.e_star))

    @cached_property
    @tp.override
    def R_star(self) -> float:
        return self.r_star / self.H_star

    @functools.cached_property
    def R_star_m(self) -> th.FloatOrArr:
        r"""Mean bubble separation $R_*$, in meters, presuming that $T$ is in GeV."""
        return self.R_star * const.GEV_IN_J * const.PLANCK_LENGTH
