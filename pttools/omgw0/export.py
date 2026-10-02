r"""Exportable fields of the gravitational wave spectra today.

These extend the fields of :py:mod:`pttools.ssm.export`.
In the :py:attr:`~pttools.utils.fields.Preset.MINIMAL` preset,
$\mathcal{P}_\text{gw}$ is replaced with $\Omega_{\text{gw},0} h^2$.
"""

import typing as tp

from pttools.ssm.export import SSM_SPECTRUM_FIELDS, Y_AXIS
from pttools.utils.fields import Field, Fields, FieldShape, Preset

if tp.TYPE_CHECKING:
    from pttools.omgw0.spectrum import Spectrum

__all__ = [
    "SPECTRUM_FIELDS",
]

_MINIMAL_INIT = {Preset.MINIMAL, Preset.FULL, Preset.INIT}
_FULL = {Preset.FULL}
_FULL_INIT = {Preset.FULL, Preset.INIT}


def _F_gw0(spectrum: "Spectrum") -> float:
    return spectrum.F_gw0()


def _f(spectrum: "Spectrum") -> tp.Any:
    return spectrum.f()


def _f_max(spectrum: "Spectrum") -> float:
    return spectrum.f().max()


def _f_min(spectrum: "Spectrum") -> float:
    return spectrum.f().min()


def _omgw0(spectrum: "Spectrum") -> tp.Any:
    return spectrum.omgw0()


def _omgw0_h2(spectrum: "Spectrum") -> tp.Any:
    return spectrum.omgw0_h2()


def _omgw0_peak_f(spectrum: "Spectrum") -> float:
    return spectrum.omgw0_peak()[0]


def _omgw0_peak(spectrum: "Spectrum") -> float:
    return spectrum.omgw0_peak()[1]


def _omgw0_total(spectrum: "Spectrum") -> float:
    return spectrum.omgw0_total()


def _snr(spectrum: "Spectrum") -> float:
    return spectrum.snr()[0]


def _snr_ins(spectrum: "Spectrum") -> float:
    return spectrum.snr_ins()[0]


def _array(name: str, getter: tp.Any, description: str, presets: tp.Iterable[Preset]) -> Field:
    return Field(
        name, getter=getter, shape=FieldShape.ARRAY, axis=Y_AXIS, presets=frozenset(presets), description=description)


#: Fields of :py:class:`pttools.omgw0.spectrum.Spectrum`
SPECTRUM_FIELDS: Fields = Fields(
    SSM_SPECTRUM_FIELDS,
    # Replace pow_gw with omgw0_h2 in the minimal preset
    _array("pow_gw", None, r"$\mathcal{P}_\text{gw}(y)$, GW power spectrum at the time of production", ()),
    # Input parameters
    Field("g_star", presets=_MINIMAL_INIT, description="$g_*$, degrees of freedom for pressure at GW production"),
    Field(
        "gs_star", presets=_FULL_INIT,
        description="$g_{s,*}$, degrees of freedom for entropy at GW production"),
    Field("T_star", presets=_MINIMAL_INIT, description="$T_*$, temperature at GW production"),
    # Computed values
    Field("F_gw0", getter=_F_gw0, presets=_FULL, description=r"$F_{\text{gw},0}$, power attenuation"),
    Field("ge_star", presets=_FULL, description="$g_{e,*}$, degrees of freedom for energy density at GW production"),
    Field("e_star", presets=_FULL, description="$e_*$, energy density at GW production"),
    Field("f_max", getter=_f_max, presets=_FULL, description=r"$f_\text{max}$, maximum frequency today"),
    Field("f_min", getter=_f_min, presets=_FULL, description=r"$f_\text{min}$, minimum frequency today"),
    Field("f_star0", presets=_FULL, description="$f_{*,0}$, frequency today corresponding to the Hubble rate"),
    Field("H_star", presets=_FULL, description="$H_*$, Hubble rate at GW production"),
    Field(
        "omgw0_peak_f", getter=_omgw0_peak_f, presets=_FULL,
        description=r"$f_\text{peak}$, frequency of the peak of $\Omega_{\text{gw},0}$"),
    Field(
        "omgw0_peak", getter=_omgw0_peak, presets=_FULL,
        description=r"$\Omega_{\text{gw},0,\text{peak}}$, peak of $\Omega_{\text{gw},0}$"),
    Field(
        "omgw0_total", getter=_omgw0_total, presets=_FULL,
        description=r"$\Omega_{\text{gw},0,\text{total}}$, $\Omega_{\text{gw},0}$ integrated over all frequencies"),
    Field("R_star", presets=_FULL, description="$R_*$, mean bubble separation"),
    Field("R_star_m", presets=_FULL, description="$R_*$, mean bubble separation in meters"),
    Field("snr", getter=_snr, presets=_FULL, description="signal-to-noise ratio for LISA"),
    Field(
        "snr_ins", getter=_snr_ins, presets=_FULL,
        description="signal-to-noise ratio for LISA with only the instrument noise"),
    # Computed arrays
    _array(
        "omgw0_h2", _omgw0_h2,
        r"$\Omega_{\text{gw},0} h^2 (y)$, GW power spectrum today", {Preset.MINIMAL}),
    _array("omgw0", _omgw0, r"$\Omega_{\text{gw},0} (y)$, GW power spectrum today", ()),
    _array("f", _f, "$f(y)$, frequency today", ()),
)
