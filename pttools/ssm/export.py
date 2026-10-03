r"""Exportable fields of the Sound Shell Model spectra.

The fields of the :py:attr:`~pttools.utils.fields.Preset.INIT` preset are named after the constructor parameters
of :py:class:`pttools.ssm.spectrum.SSMSpectrum`, so that a spectrum can be recreated from them.
The parameter $y = kR_*$ is a :py:attr:`~pttools.utils.fields.FieldShape.GRID` field,
which is stored only once per file.
"""

from collections.abc import Set
import typing as tp

import numpy as np

from pttools.ssm.nucleation import NucType
from pttools.ssm.suppression import SUPPRESSIONS, Suppression, SuppressionMethod
from pttools.utils.fields import (
    PRESETS_ALL,
    PRESETS_FULL,
    PRESETS_FULL_INIT,
    PRESETS_INIT,
    PRESETS_MINIMAL,
    PRESETS_MINIMAL_FULL,
    Field,
    Fields,
    FieldShape,
    FieldType,
    Preset,
    decode_optional,
    decode_optional_int,
)

if tp.TYPE_CHECKING:
    from pttools.ssm.spectrum import SSMSpectrum

__all__ = [
    "SSM_SPECTRUM_FIELDS",
    "Y_AXIS",
    "decode_suppression",
]

#: Name of the axis of the spectra, $y = kR_*$
Y_AXIS: str = "y"


def decode_suppression(name: str) -> Suppression:
    """Get a built-in suppression dataset by its name.

    :param name: name of the suppression dataset
    :return: the suppression dataset
    :raises ValueError: if there is no built-in suppression dataset with the given name
    """
    for suppression in SUPPRESSIONS:
        if suppression.name == name:
            return suppression
    raise ValueError(
        f"Unknown suppression dataset: \"{name}\". Available: {', '.join(sup.name for sup in SUPPRESSIONS)}")


def _array(
        name: str,
        description: str = "",
        presets: Set[Preset] = frozenset(),
        axis: str = Y_AXIS,
        getter: tp.Callable[[tp.Any], tp.Any] | None = None) -> Field:
    """Create a field for an array, whose length is the same for all the spectra of a file.

    :param name: name of the field
    :param description: description of the field.
        If empty, the first line of the docstring of the corresponding property is used.
    :param presets: the presets that include the field
    :param axis: name of the axis of the array
    :param getter: function that returns the array for a given spectrum
    :return: the field
    """
    return Field(name, getter=getter, shape=FieldShape.ARRAY, axis=axis, presets=presets, description=description)


def _ragged(name: str, description: str, axis: str) -> Field:
    """Create a field for a variable-length array of the :py:attr:`~pttools.utils.fields.Preset.FULL` preset.

    The length of the array can vary by spectrum.

    :param name: name of the field
    :param description: description of the field
    :param axis: name of the axis of the array. The arrays of the same axis must have the same length.
    :return: the field
    """
    return Field(name, shape=FieldShape.RAGGED, axis=axis, presets=PRESETS_FULL, description=description)


def _spec_den_gw_low(spectrum: "SSMSpectrum") -> np.ndarray:
    r"""$\tilde{P}_\text{gw,low}(y)$ as an array.

    In the bag model the low-$k$ spectral density is independent of $z$, and is therefore computed as a scalar.
    This broadcasts it to the shape of $y$, so that its shape is the same for all the spectra.
    """
    return np.broadcast_to(spectrum.spec_den_gw_low, spectrum.y.shape)


#: Fields of :py:class:`pttools.ssm.spectrum.SSMSpectrum`
SSM_SPECTRUM_FIELDS: Fields = Fields(
    # Bubble parameters, which are copied here for convenience
    Field("v_wall", getter="bubble.v_wall", presets=PRESETS_MINIMAL, description=r"$v_\text{wall}$, wall speed"),
    Field("alpha_n", getter="bubble.alpha_n", presets=PRESETS_MINIMAL, description=r"$\alpha_n$, transition strength"),
    # Input parameters
    Field(
        "beta_tilde", presets=PRESETS_ALL, decode=decode_optional,
        description=r"$\tilde{\beta} = \beta / H_*$, nucleation rate parameter, if it was given instead of $r_*$"),
    Field("r_star", presets=PRESETS_ALL, description="$r_*$, Hubble-scaled mean bubble spacing"),
    Field(
        "a_star_a_r_ratio", presets=PRESETS_FULL_INIT,
        description="$a_* / a_r$, ratio of the scale factors at the time of GW production and at radiation domination"),
    Field(
        "low_k", type=FieldType.BOOL, presets=PRESETS_FULL_INIT,
        description="whether the low $k$ approximation of Giombi et al. (2024) is used"),
    Field("N_sh", presets=PRESETS_FULL_INIT, description=r"$N_\text{sh}$, number of shock formation times"),
    Field("nuc_type", type=FieldType.STR, presets=PRESETS_FULL_INIT, decode=NucType, description="nucleation type"),
    Field("nT", type=FieldType.INT, presets=PRESETS_FULL_INIT, description="number of points in the $t$ array"),
    Field(
        "nx_P_tilde_gw", presets=PRESETS_FULL_INIT, decode=decode_optional_int,
        description=r"number of points in the $\tilde{P}_\text{gw}$ integration, if not the default"),
    Field(
        "n_z_lookup", type=FieldType.INT, presets=PRESETS_FULL_INIT,
        description="number of points in the lookup arrays"),
    Field(
        "z_st_thresh", presets=PRESETS_FULL_INIT,
        description=r"$z_\text{st,thresh}$, $z$ above which an approximation of the sine transform is used"),
    Field("T_tilde_min", presets=PRESETS_INIT, description=r"$\tilde{T}_\text{min}$"),
    Field("T_tilde_max", presets=PRESETS_INIT, description=r"$\tilde{T}_\text{max}$"),
    Field(
        "suppression", getter="suppression.name", type=FieldType.STR, presets=PRESETS_INIT,
        decode=decode_suppression, description="name of the suppression dataset"),
    Field(
        "suppression_method", type=FieldType.STR, presets=PRESETS_INIT, decode=SuppressionMethod,
        description="suppression method"),
    # Computed arrays
    # The lengths of the lookup arrays depend on the parameters, and therefore they are ragged.
    _ragged("a2", r"$|A(z)|^2$", axis="qT_lookup"),
    _ragged("a2_lookup", r"$|A_\text{lookup}(z)|^2$", axis="a2_lookup"),
    _array("spec_den_gw_ssm", r"$\tilde{P}_\text{gw,ssm}$", PRESETS_FULL),
    _array("spec_den_gw_expanded", r"$\tilde{P}_\text{gw,expanded}$", PRESETS_FULL),
    _array("spec_den_gw_int", r"$\tilde{P}_\text{gw,int}$", PRESETS_FULL),
    _array("spec_den_gw_low", r"$\tilde{P}_\text{gw,low}$", PRESETS_FULL, getter=_spec_den_gw_low),
    _array("spec_den_v", r"$\tilde{P}_v(z)$", PRESETS_FULL),
    _ragged("spec_den_v_lookup", r"$\tilde{P}_v({z}_\text{lookup})$", axis="z_lookup"),
    _ragged("T_tilde", r"$\tilde{T}$", axis="T_tilde"),
    Field(
        "y", shape=FieldShape.GRID, axis=Y_AXIS, presets=PRESETS_ALL,
        description="$y = kR_*$, wavenumber scaled by the mean bubble spacing"),
    _ragged("z_lookup", r"$z_\text{lookup}$", axis="z_lookup"),
    # The descriptions of the properties are taken from their docstrings.
    _array("pow_gw", presets=PRESETS_MINIMAL),
    _array("pow_v"),
    _array("spec_den_gw", r"$\tilde{P}_\text{gw}(y)$, spectral density of the GW power"),
    # Computed values
    Field("cs2", presets=PRESETS_MINIMAL_FULL, description=r"$c_s^2(T_{\text{gw}})$, speed of sound squared"),
    # The descriptions of the properties are taken from their docstrings.
    Field("css2_Tn", presets=PRESETS_MINIMAL),
    Field("csb2_Tn", presets=PRESETS_MINIMAL),
    Field("delta_tau_v", presets=PRESETS_FULL),
    Field("dilution_of_e", presets=PRESETS_FULL),
    Field("H_star_eta_sh", presets=PRESETS_FULL),
    Field("H_star_eta_star", presets=PRESETS_FULL),
    Field("H_star_eta_v", presets=PRESETS_FULL),
    Field("H_star_eta_v_old", presets=PRESETS_FULL),
    Field("k_peak_eta_star", presets=PRESETS_FULL),
    Field("J", presets=PRESETS_FULL),
    Field("label_latex", type=FieldType.STR, presets=PRESETS_FULL, description="LaTeX label"),
    Field("label_unicode", type=FieldType.STR, presets=PRESETS_FULL, description="Unicode label"),
    Field("source_lifetime_factor", presets=PRESETS_FULL),
    # The docstring of suppression_factor is copied from a method, and therefore it does not suit as a description.
    Field("suppression_factor", presets=PRESETS_FULL, description="suppression factor"),
    Field("tau_end", presets=PRESETS_FULL),
    Field("tau_star", presets=PRESETS_FULL),
    Field("ubarf2", presets=PRESETS_FULL, description=r"$\bar{U}_f^2$, mean square fluid velocity"),
)
