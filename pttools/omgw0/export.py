r"""Exportable fields of the gravitational wave spectra today.

These extend the fields of :py:mod:`pttools.ssm.export`.
In the :py:attr:`~pttools.utils.fields.Preset.MINIMAL` preset,
$\mathcal{P}_\text{gw}$ is replaced with $\Omega_{\text{gw},0} h^2$.

The spectra can be given either the $y$ array or the frequencies $f$.
For the former, the fields are :py:data:`SPECTRUM_Y_FIELDS`, where $y$ is shared by the spectra of a file.
These are the same as the general fields :py:data:`SPECTRUM_FIELDS`, which are also used e.g. for the JSON export.
For the latter, the fields are :py:data:`SPECTRUM_F_FIELDS`, where $f$ is shared instead,
and $y$ is an array of each spectrum, as it depends on the parameters of the spectrum.
"""

from collections.abc import Set
import dataclasses

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
    "F_AXIS",
    "SPECTRUM_FIELDS",
    "SPECTRUM_F_FIELDS",
    "SPECTRUM_Y_FIELDS",
]

#: Name of the axis of the spectra that have been given the frequencies $f$
F_AXIS: str = "f"


def _array(name: str, description: str = "", presets: Set[Preset] = frozenset(), call: bool = False) -> Field:
    """Create a field for an array along the axis $y$.

    :param name: name of the field, which is also the name of the attribute
    :param description: description of the field.
        If empty, the first line of the docstring of the corresponding property or method is used.
    :param presets: the presets that include the field
    :param call: whether the attribute is a method that should be called without arguments
    :return: the field
    """
    return Field(name, call=call, shape=FieldShape.ARRAY, axis=Y_AXIS, presets=presets, description=description)


#: Fields of :py:class:`pttools.omgw0.spectrum.Spectrum`.
#: The descriptions of the fields that have no description are taken from the docstrings
#: of the corresponding properties and methods.
SPECTRUM_FIELDS: Fields = Fields(
    SSM_SPECTRUM_FIELDS,
    # Replace pow_gw with omgw0_h2 in the minimal preset
    _array("pow_gw"),
    # Input parameters
    Field("g_star", presets=PRESETS_ALL, description="$g_*$, degrees of freedom for pressure at GW production"),
    Field(
        "gs_star", presets=PRESETS_FULL_INIT,
        description="$g_{s,*}$, degrees of freedom for entropy at GW production"),
    Field("T_star", presets=PRESETS_ALL, description="$T_*$, temperature at GW production"),
    # Computed values
    Field("F_gw0", call=True, presets=PRESETS_FULL),
    Field("ge_star", presets=PRESETS_FULL),
    Field("e_star", presets=PRESETS_FULL),
    Field("f_max", presets=PRESETS_FULL),
    Field("f_min", presets=PRESETS_FULL),
    Field("f_star0", presets=PRESETS_FULL),
    Field("H_star", presets=PRESETS_FULL),
    # The docstrings of the methods that return tuples describe the entire tuples,
    # and therefore the fields of their items have descriptions of their own.
    Field(
        "omgw0_peak_f", getter="omgw0_peak", call=True, index=0, presets=PRESETS_FULL,
        description=r"$f_\text{peak}$, frequency of the peak of $\Omega_{\text{gw},0}$"),
    Field(
        "omgw0_peak", call=True, index=1, presets=PRESETS_FULL,
        description=r"$\Omega_{\text{gw},0,\text{peak}}$, peak of $\Omega_{\text{gw},0}$"),
    Field("omgw0_total", call=True, presets=PRESETS_FULL),
    Field("R_star", presets=PRESETS_FULL),
    Field("R_star_m", presets=PRESETS_FULL),
    Field("snr", call=True, index=0, presets=PRESETS_FULL, description="signal-to-noise ratio for LISA"),
    Field(
        "snr_ins", call=True, index=0, presets=PRESETS_FULL,
        description="signal-to-noise ratio for LISA with only the instrument noise"),
    # Computed arrays
    _array("omgw0_h2", presets=PRESETS_MINIMAL, call=True),
    _array("omgw0", call=True),
    # The docstring of f() describes the conversion, and therefore it does not suit as a description.
    _array("f", "$f(y)$, frequency today", call=True),
)


def _on_f_grid(fields: Fields) -> Fields:
    """Convert the fields of the spectra that share $y$ to the fields of the spectra that share $f$.

    The arrays along the $y$ axis are moved to the $f$ axis,
    $f$ becomes the grid, which is also needed for recreating the spectra,
    and $y$ becomes an array of each spectrum, which is included only in the
    :py:attr:`~pttools.utils.fields.Preset.FULL` preset, as it can be computed from $f$.

    :param fields: the fields of the spectra that share $y$
    :return: the fields of the spectra that share $f$
    """
    converted = [
        dataclasses.replace(field, axis=F_AXIS) if field.shape == FieldShape.ARRAY and field.axis == Y_AXIS else field
        for field in fields.values()
    ]
    return Fields(
        *converted,
        Field("y", shape=FieldShape.ARRAY, axis=F_AXIS, presets=PRESETS_FULL, description=fields["y"].description),
        Field(
            "f", call=True, shape=FieldShape.GRID, axis=F_AXIS, presets=PRESETS_ALL,
            description="$f$, frequencies today"),
    )


#: Fields of :py:class:`pttools.omgw0.spectrum.Spectrum` for the spectra that have been given the $y$ array.
#: This is an alias of :py:data:`SPECTRUM_FIELDS` for consistency with :py:data:`SPECTRUM_F_FIELDS`.
SPECTRUM_Y_FIELDS: Fields = SPECTRUM_FIELDS

#: Fields of :py:class:`pttools.omgw0.spectrum.Spectrum` for the spectra that have been given the frequencies $f$
SPECTRUM_F_FIELDS: Fields = _on_f_grid(SPECTRUM_Y_FIELDS)
