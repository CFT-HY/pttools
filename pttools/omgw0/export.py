r"""Exportable fields of the gravitational wave spectra today.

These extend the fields of :py:mod:`pttools.ssm.export`.
In the :py:attr:`~pttools.utils.fields.Preset.MINIMAL` preset,
$\mathcal{P}_\text{gw}$ is replaced with $\Omega_{\text{gw},0} h^2$.
"""

from collections.abc import Set

from pttools.ssm.export import SSM_SPECTRUM_FIELDS, Y_AXIS
from pttools.utils.fields import (
    PRESETS_ALL,
    PRESETS_FULL,
    PRESETS_FULL_INIT,
    PRESETS_MINIMAL,
    Field,
    Fields,
    FieldShape,
    Preset,
)

__all__ = [
    "SPECTRUM_FIELDS",
]


def _array(name: str, description: str, presets: Set[Preset] = frozenset(), call: bool = False) -> Field:
    """Create a field for an array along the axis $y$.

    :param name: name of the field, which is also the name of the attribute
    :param description: description of the field
    :param presets: the presets that include the field
    :param call: whether the attribute is a method that should be called without arguments
    :return: the field
    """
    return Field(name, call=call, shape=FieldShape.ARRAY, axis=Y_AXIS, presets=presets, description=description)


#: Fields of :py:class:`pttools.omgw0.spectrum.Spectrum`
SPECTRUM_FIELDS: Fields = Fields(
    SSM_SPECTRUM_FIELDS,
    # Replace pow_gw with omgw0_h2 in the minimal preset
    _array("pow_gw", r"$\mathcal{P}_\text{gw}(y)$, GW power spectrum at the time of production"),
    # Input parameters
    Field("g_star", presets=PRESETS_ALL, description="$g_*$, degrees of freedom for pressure at GW production"),
    Field(
        "gs_star", presets=PRESETS_FULL_INIT,
        description="$g_{s,*}$, degrees of freedom for entropy at GW production"),
    Field("T_star", presets=PRESETS_ALL, description="$T_*$, temperature at GW production"),
    # Computed values
    Field("F_gw0", call=True, presets=PRESETS_FULL, description=r"$F_{\text{gw},0}$, power attenuation"),
    Field(
        "ge_star", presets=PRESETS_FULL,
        description="$g_{e,*}$, degrees of freedom for energy density at GW production"),
    Field("e_star", presets=PRESETS_FULL, description="$e_*$, energy density at GW production"),
    Field("f_max", presets=PRESETS_FULL, description=r"$f_\text{max}$, maximum frequency today"),
    Field("f_min", presets=PRESETS_FULL, description=r"$f_\text{min}$, minimum frequency today"),
    Field(
        "f_star0", presets=PRESETS_FULL,
        description=r"$f_{\ast,0}$, frequency today corresponding to the Hubble rate"),
    Field("H_star", presets=PRESETS_FULL, description="$H_*$, Hubble rate at GW production"),
    Field(
        "omgw0_peak_f", getter="omgw0_peak", call=True, index=0, presets=PRESETS_FULL,
        description=r"$f_\text{peak}$, frequency of the peak of $\Omega_{\text{gw},0}$"),
    Field(
        "omgw0_peak", call=True, index=1, presets=PRESETS_FULL,
        description=r"$\Omega_{\text{gw},0,\text{peak}}$, peak of $\Omega_{\text{gw},0}$"),
    Field(
        "omgw0_total", call=True, presets=PRESETS_FULL,
        description=r"$\Omega_{\text{gw},0,\text{total}}$, $\Omega_{\text{gw},0}$ integrated over all frequencies"),
    Field("R_star", presets=PRESETS_FULL, description="$R_*$, mean bubble separation"),
    Field("R_star_m", presets=PRESETS_FULL, description="$R_*$, mean bubble separation in meters"),
    Field("snr", call=True, index=0, presets=PRESETS_FULL, description="signal-to-noise ratio for LISA"),
    Field(
        "snr_ins", call=True, index=0, presets=PRESETS_FULL,
        description="signal-to-noise ratio for LISA with only the instrument noise"),
    # Computed arrays
    _array(
        "omgw0_h2", r"$\Omega_{\text{gw},0} h^2 (y)$, GW power spectrum today", PRESETS_MINIMAL, call=True),
    _array("omgw0", r"$\Omega_{\text{gw},0} (y)$, GW power spectrum today", call=True),
    _array("f", "$f(y)$, frequency today", call=True),
)
