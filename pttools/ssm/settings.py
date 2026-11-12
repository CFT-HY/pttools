r"""Accuracy settings of the bubbles and spectra.

See :ref:`sphx_glr_auto_examples_props_n_points.py` for the effect of the settings on the accuracy of the spectra.
"""

import dataclasses
import typing as tp

from pttools.bubble.const import DEFAULT_N_XI
from pttools.ssm.const import DEFAULT_N_T, DEFAULT_N_Z_LOOKUP

__all__ = ["SSMSettings"]


@dataclasses.dataclass(frozen=True, slots=True)
class SSMSettings:
    r"""Accuracy settings of the bubbles and spectra, i.e. the sizes of their internal arrays.

    According to :ref:`sphx_glr_auto_examples_props_n_points.py`,
    the accuracy and the computation time of the spectra are determined mostly by ``n_z_lookup``.

    :param n_z_lookup: number of points of the lookup arrays of the spectra
    :param nT: number of points of the bubble lifetime distribution integration
    :param nx_P_tilde_gw: number of points of the integration of $\tilde{P}_\text{gw}$,
        or None for the same as ``n_z_lookup``
    :param n_xi: number of points of the fluid profiles of the bubbles, see :py:class:`pttools.bubble.bubble.Bubble`
    """

    n_z_lookup: int = DEFAULT_N_Z_LOOKUP
    nT: int = DEFAULT_N_T  # noqa: N815
    nx_P_tilde_gw: int | None = None  # noqa: N815
    n_xi: int = DEFAULT_N_XI

    def spectrum_kwargs(self) -> dict[str, tp.Any]:
        """The accuracy settings as arguments of the spectra.

        The spectra are :py:class:`pttools.ssm.spectrum.SSMSpectrum` and :py:class:`pttools.omgw0.spectrum.Spectrum`.
        """
        return {"n_z_lookup": self.n_z_lookup, "nx_P_tilde_gw": self.nx_P_tilde_gw, "nT": self.nT}
