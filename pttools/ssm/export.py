r"""Exportable fields of the Sound Shell Model spectra.

The fields of the :py:attr:`~pttools.utils.fields.Preset.INIT` preset are named after the constructor parameters
of :py:class:`pttools.ssm.spectrum.SSMSpectrum`, so that a spectrum can be recreated from them.
The parameter $y = kR_*$ is a :py:attr:`~pttools.utils.fields.FieldShape.GRID` field,
which is stored only once per file.
"""

import typing as tp

import numpy as np

from pttools.bubble.export import csb2_Tn, css2_Tn
from pttools.ssm.nucleation import NucType
from pttools.utils.fields import Field, Fields, FieldShape, FieldType, Preset, decode_optional, decode_optional_int

if tp.TYPE_CHECKING:
    from pttools.ssm.spectrum import SSMSpectrum
    from pttools.ssm.suppression import Suppression, SuppressionMethod

__all__ = [
    "SSM_SPECTRUM_FIELDS",
    "Y_AXIS",
    "decode_suppression",
    "decode_suppression_method",
]

#: Name of the axis of the spectra, $y = kR_*$
Y_AXIS: str = "y"

_MINIMAL = {Preset.MINIMAL}
_MINIMAL_FULL = {Preset.MINIMAL, Preset.FULL}
_MINIMAL_INIT = {Preset.MINIMAL, Preset.FULL, Preset.INIT}
_FULL = {Preset.FULL}
_FULL_INIT = {Preset.FULL, Preset.INIT}


# The suppression module is imported within the functions,
# since importing it here would result in a circular import with pttools.ssm.spectrum.

def decode_suppression(name: str) -> "Suppression":
    """Get a built-in suppression dataset by its name.

    :param name: name of the suppression dataset
    :return: the suppression dataset
    :raises ValueError: if there is no built-in suppression dataset with the given name
    """
    from pttools.ssm.suppression import SUPPRESSIONS  # noqa: PLC0415
    for suppression in SUPPRESSIONS:
        if suppression.name == name:
            return suppression
    raise ValueError(
        f"Unknown suppression dataset: \"{name}\". Available: {', '.join(sup.name for sup in SUPPRESSIONS)}")


def decode_suppression_method(value: str) -> "SuppressionMethod":
    """Get a suppression method by its value."""
    from pttools.ssm.suppression import SuppressionMethod  # noqa: PLC0415
    return SuppressionMethod(value)


def _array(
        name: str,
        description: str,
        presets: tp.Iterable[Preset],
        axis: str = Y_AXIS,
        getter: tp.Callable[[tp.Any], tp.Any] | None = None) -> Field:
    return Field(
        name, getter=getter, shape=FieldShape.ARRAY, axis=axis, presets=frozenset(presets), description=description)


def _ragged(name: str, description: str, axis: str) -> Field:
    return Field(name, shape=FieldShape.RAGGED, axis=axis, presets=frozenset(_FULL), description=description)


def _spec_den_gw_low(spectrum: "SSMSpectrum") -> tp.Any:
    # In the bag model the low-k spectral density is independent of z, and is therefore computed as a scalar.
    return np.broadcast_to(spectrum.spec_den_gw_low, spectrum.y.shape)


def _bubble_css2_Tn(spectrum: "SSMSpectrum") -> float:
    return css2_Tn(spectrum.bubble)


def _bubble_csb2_Tn(spectrum: "SSMSpectrum") -> float:
    return csb2_Tn(spectrum.bubble)


#: Fields of :py:class:`pttools.ssm.spectrum.SSMSpectrum`
SSM_SPECTRUM_FIELDS: Fields = Fields(
    # Bubble parameters, which are copied here for convenience
    Field("v_wall", getter="bubble.v_wall", presets=_MINIMAL, description=r"$v_\text{wall}$, wall speed"),
    Field("alpha_n", getter="bubble.alpha_n", presets=_MINIMAL, description=r"$\alpha_n$, transition strength"),
    # Input parameters
    Field(
        "beta_tilde", presets=_MINIMAL_INIT, decode=decode_optional,
        description=r"$\tilde{\beta} = \beta / H_*$, nucleation rate parameter (NaN if $r_*$ was given instead)"),
    Field("r_star", presets=_MINIMAL_INIT, description="$r_*$, Hubble-scaled mean bubble spacing"),
    Field(
        "a_star_a_r_ratio", presets=_FULL_INIT,
        description="$a_* / a_r$, ratio of the scale factors at the time of GW production and at radiation domination"),
    Field(
        "low_k", type=FieldType.BOOL, presets=_FULL_INIT,
        description="whether the low $k$ approximation of Giombi et al. (2024) is used"),
    Field("N_sh", presets=_FULL_INIT, description=r"$N_\text{sh}$, number of shock formation times"),
    Field("nuc_type", type=FieldType.STR, presets=_FULL_INIT, decode=NucType, description="nucleation type"),
    Field("nT", type=FieldType.INT, presets=_FULL_INIT, description="number of points in the $t$ array"),
    Field(
        "nx_P_tilde_gw", presets=_FULL_INIT, decode=decode_optional_int,
        description=r"number of points in the $\tilde{P}_\text{gw}$ integration (NaN for the default)"),
    Field("n_z_lookup", type=FieldType.INT, presets=_FULL_INIT, description="number of points in the lookup arrays"),
    Field(
        "z_st_thresh", presets=_FULL_INIT,
        description=r"$z_\text{st,thresh}$, $z$ above which an approximation of the sine transform is used"),
    Field("T_tilde_min", presets={Preset.INIT}, description=r"$\tilde{T}_\text{min}$"),
    Field("T_tilde_max", presets={Preset.INIT}, description=r"$\tilde{T}_\text{max}$"),
    Field(
        "suppression", getter="suppression.name", type=FieldType.STR, presets={Preset.INIT},
        decode=decode_suppression, description="name of the suppression dataset"),
    Field(
        "suppression_method", type=FieldType.STR, presets={Preset.INIT}, decode=decode_suppression_method,
        description="suppression method"),
    # Computed arrays
    # The lengths of the lookup arrays depend on the parameters, and therefore they are ragged.
    _ragged("a2", r"$|A(z)|^2$", axis="qT_lookup"),
    _ragged("a2_lookup", r"$|A_\text{lookup}(z)|^2$", axis="a2_lookup"),
    _array("spec_den_gw_ssm", r"$\tilde{P}_\text{gw,ssm}$", _FULL),
    _array("spec_den_gw_expanded", r"$\tilde{P}_\text{gw,expanded}$", _FULL),
    _array("spec_den_gw_int", r"$\tilde{P}_\text{gw,int}$", _FULL),
    _array("spec_den_gw_low", r"$\tilde{P}_\text{gw,low}$", _FULL, getter=_spec_den_gw_low),
    _array("spec_den_v", r"$\tilde{P}_v(z)$", _FULL),
    _ragged("spec_den_v_lookup", r"$\tilde{P}_v(z_\text{lookup})$", axis="z_lookup"),
    _ragged("T_tilde", r"$\tilde{T}$", axis="T_tilde"),
    Field(
        "y", shape=FieldShape.GRID, axis=Y_AXIS, presets=_MINIMAL_INIT,
        description="$y = kR_*$, wavenumber scaled by the mean bubble spacing"),
    _ragged("z_lookup", r"$z_\text{lookup}$", axis="z_lookup"),
    _array("pow_gw", r"$\mathcal{P}_\text{gw}(y)$, GW power spectrum", _MINIMAL),
    _array("pow_v", r"$\mathcal{P}_v(y)$, velocity power spectrum", ()),
    _array("spec_den_gw", r"$\tilde{P}_\text{gw}(y)$, spectral density of the GW power", ()),
    # Computed values
    Field("cs2", presets=_MINIMAL_FULL, description=r"$c_s^2(T_\text{gw})$, speed of sound squared"),
    Field(
        "css2_Tn", getter=_bubble_css2_Tn, presets=_MINIMAL,
        description="$c_{s,s}^2(T_n)$, speed of sound squared in the symmetric phase at the nucleation temperature"),
    Field(
        "csb2_Tn", getter=_bubble_csb2_Tn, presets=_MINIMAL,
        description="$c_{s,b}^2(T_n)$, speed of sound squared in the broken phase at the nucleation temperature"),
    Field("delta_tau_v", presets=_FULL, description=r"$\Delta \tau_\text{v}$, source duration"),
    Field("dilution_of_e", presets=_FULL, description="dilution of the energy density"),
    Field("H_star_eta_sh", presets=_FULL, description=r"$H_* \eta_\text{sh}$"),
    Field("H_star_eta_star", presets=_FULL, description=r"$\mathcal{H}_* \eta_*$"),
    Field("H_star_eta_v", presets=_FULL, description=r"$H_* \eta_\text{v}$"),
    Field("H_star_eta_v_old", presets=_FULL, description=r"$H_* \eta_\text{v}$ with the old definition"),
    Field("k_peak_eta_star", presets=_FULL, description=r"$k_\text{peak} \eta_*$"),
    Field("J", presets=_FULL, description="$J$, source lifetime factor"),
    Field("label_latex", type=FieldType.STR, presets=_FULL, description="LaTeX label"),
    Field("label_unicode", type=FieldType.STR, presets=_FULL, description="Unicode label"),
    Field("source_lifetime_factor", presets=_FULL, description="source lifetime factor"),
    Field("suppression_factor", presets=_FULL, description="suppression factor"),
    Field("tau_end", presets=_FULL, description=r"$\tau_\text{end}$, dimensionless end time of the source"),
    Field("tau_star", presets=_FULL, description=r"$\tau_*$, dimensionless start time of the source"),
    Field("ubarf2", presets=_FULL, description=r"$\bar{U}_f^2$, mean square fluid velocity"),
)
