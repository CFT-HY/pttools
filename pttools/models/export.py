r"""Exportable fields of the equation of state models.

The fields of the :py:attr:`~pttools.utils.fields.Preset.INIT` preset are named after the constructor parameters,
so that a model can be recreated from them.
The constructor of a subclass may not accept all of these, in which case only the accepted ones are used.
"""

import datetime
import typing as tp

from pttools.utils.fields import Field, Fields, FieldType, Preset

__all__ = [
    "ANALYTIC_MODEL_FIELDS",
    "BASE_MODEL_FIELDS",
    "CONST_CS_MODEL_FIELDS",
    "CONST_CS_THERMO_MODEL_FIELDS",
    "MODEL_FIELDS",
]

_MINIMAL = {Preset.MINIMAL, Preset.FULL, Preset.INIT}
_FULL = {Preset.FULL}
_FULL_INIT = {Preset.FULL, Preset.INIT}


def _now(_obj: tp.Any) -> datetime.datetime:
    return datetime.datetime.now()


#: Fields of :py:class:`pttools.models.base.BaseModel`
BASE_MODEL_FIELDS: Fields = Fields(
    Field("name", type=FieldType.STR, presets=_MINIMAL, description="name of the model"),
    Field("label_latex", type=FieldType.STR, presets=_FULL_INIT, description="LaTeX label"),
    Field("label_unicode", type=FieldType.STR, presets=_FULL_INIT, description="Unicode label"),
    Field("datetime", getter=_now, type=FieldType.STR, presets=_FULL, description="time of the export"),
    Field("T_min", presets=_FULL_INIT, description=r"$T_\text{min}$, minimum temperature"),
    Field("T_max", presets=_FULL_INIT, description=r"$T_\text{max}$, maximum temperature"),
    Field(
        "restrict_to_valid", type=FieldType.BOOL, presets=_FULL_INIT,
        description="whether invalid temperatures are converted to NaN"),
    Field(
        "silence_temp", type=FieldType.BOOL, presets=_FULL_INIT,
        description="whether invalid temperatures are not logged"),
    Field(
        "temperature_is_physical", type=FieldType.BOOL, presets=_FULL_INIT,
        description="whether the temperature is in physical units"),
)

#: Fields of :py:class:`pttools.models.model.Model`
MODEL_FIELDS: Fields = Fields(
    BASE_MODEL_FIELDS,
    Field("T_ref", presets=_FULL_INIT, description=r"$T_\text{ref}$, reference temperature"),
    Field("T_crit", presets=_FULL, description=r"$T_\text{crit}$, critical temperature"),
    Field("V_s", presets=_FULL_INIT, description=r"$V_s$, potential in the symmetric phase"),
    Field("V_b", presets=_FULL_INIT, description=r"$V_b$, potential in the broken phase"),
    Field("w_crit", presets=_FULL, description=r"$w_\text{crit}$, enthalpy at the critical temperature"),
    Field("w_min", presets=_FULL, description=r"$w_\text{min}$, minimum enthalpy"),
    Field("w_max", presets=_FULL, description=r"$w_\text{max}$, maximum enthalpy"),
    Field("w_min_s", presets=_FULL, description=r"$w_{\text{min},s}$, minimum enthalpy in the symmetric phase"),
    Field("w_min_b", presets=_FULL, description=r"$w_{\text{min},b}$, minimum enthalpy in the broken phase"),
    Field("w_max_s", presets=_FULL, description=r"$w_{\text{max},s}$, maximum enthalpy in the symmetric phase"),
    Field("w_max_b", presets=_FULL, description=r"$w_{\text{max},b}$, maximum enthalpy in the broken phase"),
    Field("alpha_n_min", presets=_FULL, description=r"$\alpha_{n,\text{min}}$, minimum transition strength"),
    Field(
        "w_at_alpha_n_min", presets=_FULL,
        description=r"$w(\alpha_{n,\text{min}})$, enthalpy at the minimum transition strength"),
)

#: Fields of :py:class:`pttools.models.analytic.AnalyticModel`
ANALYTIC_MODEL_FIELDS: Fields = Fields(
    MODEL_FIELDS,
    Field("a_s", presets=_FULL_INIT, description="$a_s$, prefactor of $p$ in the symmetric phase"),
    Field("a_b", presets=_FULL_INIT, description="$a_b$, prefactor of $p$ in the broken phase"),
)

#: Fields of :py:class:`pttools.models.const_cs.ConstCSModel`
CONST_CS_MODEL_FIELDS: Fields = Fields(
    ANALYTIC_MODEL_FIELDS,
    Field("css2", presets=_FULL_INIT, description="$c_{s,s}^2$, speed of sound squared in the symmetric phase"),
    Field("csb2", presets=_FULL_INIT, description="$c_{s,b}^2$, speed of sound squared in the broken phase"),
    Field("mu_s", presets=_FULL, description=r"$\mu_s$, exponent of $T$ in $p$ in the symmetric phase"),
    Field("mu_b", presets=_FULL, description=r"$\mu_b$, exponent of $T$ in $p$ in the broken phase"),
)

#: Fields of :py:class:`pttools.models.const_cs_thermo.ConstCSThermoModel`
CONST_CS_THERMO_MODEL_FIELDS: Fields = Fields(
    BASE_MODEL_FIELDS,
    Field("t_ref", presets=_FULL, description=r"$T_\text{ref}$, reference temperature"),
    Field("css2", presets=_FULL, description="$c_{s,s}^2$, speed of sound squared in the symmetric phase"),
    Field("csb2", presets=_FULL, description="$c_{s,b}^2$, speed of sound squared in the broken phase"),
    Field("mu_s", presets=_FULL, description=r"$\mu_s$, exponent of $T$ in $p$ in the symmetric phase"),
    Field("mu_b", presets=_FULL, description=r"$\mu_b$, exponent of $T$ in $p$ in the broken phase"),
)
