r"""Accuracy settings of the bubbles and spectra.

See :ref:`sphx_glr_auto_examples_props_n_points.py` for the effect of the settings on the accuracy of the spectra.
"""

import dataclasses
import typing as tp


@dataclasses.dataclass(frozen=True, slots=True)
class Accuracy:
    r"""Accuracy settings of the bubbles and spectra.

    :param n_xi: number of points of the fluid profiles of the bubbles, see :py:class:`pttools.bubble.bubble.Bubble`
    :param n_z_lookup: number of points of the lookup arrays of the spectra
    :param nx_P_tilde_gw: number of points of the integration of $\tilde{P}_\text{gw}$,
        or None for the same as ``n_z_lookup``
    :param nT: number of points of the bubble lifetime distribution integration
    """

    n_xi: int
    n_z_lookup: int
    nx_P_tilde_gw: int | None  # noqa: N815
    nT: int  # noqa: N815

    def spectrum_kwargs(self) -> dict[str, tp.Any]:
        """The accuracy settings as arguments of the spectra.

        The spectra are :py:class:`pttools.ssm.spectrum.SSMSpectrum` and :py:class:`pttools.omgw0.spectrum.Spectrum`.
        """
        return {"n_z_lookup": self.n_z_lookup, "nx_P_tilde_gw": self.nx_P_tilde_gw, "nT": self.nT}
