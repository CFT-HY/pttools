"""Base class for equation of state models and thermodynamics models."""

import abc
import logging
import os
import typing as tp
import uuid

import numpy as np

from pttools.models.export import BASE_MODEL_FIELDS
import pttools.type_hints as th
from pttools.type_hints import FloatOrArr
from pttools.utils.fields import Extractable, Fields, FieldSpec, Preset
from pttools.utils.json import export_json

logger: logging.Logger = logging.getLogger(__name__)


class BaseModel(Extractable, abc.ABC):
    """The base for both Model and ThermoModel.

    All temperatures must be in units of GeV for the frequency conversion in Spectrum to work.
    """

    DEFAULT_LABEL_LATEX: str
    DEFAULT_LABEL_UNICODE: str
    DEFAULT_NAME: str
    # Zero temperature would break many of the equations
    DEFAULT_T_MIN: float = 1e-3
    DEFAULT_T_MAX: float = np.inf
    #: The exportable fields of the model. User-created model classes should extend this.
    FIELDS: tp.ClassVar[Fields] = BASE_MODEL_FIELDS

    #: Whether the temperature is in proper physics units.
    #: This is None for models that determine it at run time.
    TEMPERATURE_IS_PHYSICAL: bool | None = None

    #: The unit of the temperature in GeV, if the temperature is in physical units, e.g. $10^{-3}$ for MeV.
    TEMPERATURE_UNIT_GEV: float = 1.

    #: String formatting for thermodynamical quantities
    THERMO_FORMAT: str = "6e"

    #: Relative tolerance for the temperature validation.
    #: This allows for the floating point rounding errors of the conversions between temperature and enthalpy,
    #: e.g. when $T(w(T_{\text{min}}))$ is slightly below $T_{\text{min}}$.
    TEMP_RTOL: float = 1e-12

    def __init__(
            self,
            # Basic info
            name: str | None = None,
            label_latex: str | None = None,
            label_unicode: str | None = None,
            # Numerical values
            T_min: float | None = None,
            T_max: float | None = None,
            # Booleans
            restrict_to_valid: bool = True,
            gen_cs2: bool = True,
            gen_cs2_neg: bool = True,
            temperature_is_physical: bool | None = None,
            temperature_unit_gev: float | None = None,
            silence_temp: bool = False):
        r"""Initialize the model and validate its parameters.

        :param name: name of the model, which should not contain spaces. Defaults to ``DEFAULT_NAME``.
        :param label_latex: LaTeX label of the model. Defaults to ``DEFAULT_LABEL_LATEX``.
        :param label_unicode: Unicode label of the model. Defaults to ``DEFAULT_LABEL_UNICODE``.
        :param T_min: $T_\text{min}$, minimum temperature at which the model is valid. Defaults to ``DEFAULT_T_MIN``.
        :param T_max: $T_\text{max}$, maximum temperature at which the model is valid. Defaults to ``DEFAULT_T_MAX``.
        :param restrict_to_valid: whether temperatures outside the validity range are converted to NaN
        :param gen_cs2: whether to generate the $c_s^2$ function, used internally for postponing its generation
        :param gen_cs2_neg: whether to generate the $-c_s^2$ function
        :param temperature_is_physical: whether the temperature is in physical units.
            Defaults to ``TEMPERATURE_IS_PHYSICAL``.
        :param temperature_unit_gev: the unit of the temperature in GeV, if the temperature is in physical units.
            Defaults to ``TEMPERATURE_UNIT_GEV``.
        :param silence_temp: whether to suppress the logging of temperatures outside the validity range
        :raises ValueError: if the name, labels or temperature limits are invalid
        """
        #: Unique identifier of the model, which is used for distinguishing the compiled functions of the models.
        #: Python's id() is not used, since its values are reused after the objects have been garbage collected,
        #: which would result in a model using the compiled functions of a previous model.
        self.id: str = uuid.uuid4().hex
        self.name: str = self.DEFAULT_NAME if name is None else name
        self.label_latex: str = self.DEFAULT_LABEL_LATEX if label_latex is None else label_latex
        self.label_unicode: str = self.DEFAULT_LABEL_UNICODE if label_unicode is None else label_unicode
        self.T_min: float = self.DEFAULT_T_MIN if T_min is None else T_min
        self.T_max: float = self.DEFAULT_T_MAX if T_max is None else T_max
        self.silence_temp: bool = silence_temp
        self.restrict_to_valid: bool = restrict_to_valid
        self.temperature_is_physical: bool | None = self.TEMPERATURE_IS_PHYSICAL \
            if temperature_is_physical is None else temperature_is_physical
        self.temperature_unit_gev: float = self.TEMPERATURE_UNIT_GEV \
            if temperature_unit_gev is None else temperature_unit_gev

        if self.name is None:
            raise ValueError("The model must have a name.")
        if " " in self.name:
            logger.warning(
                "Model names should not have spaces to ensure that the file names don't cause problems. "
                "Got: \"%s\".",
                self.name
            )
        if not (self.label_latex and self.label_unicode):
            raise ValueError("The model must have labels.")
        if "$" in self.label_unicode:
            logger.warning(
                "The Unicode label of a model should not contain \"$\". Got: \"%s\"",
                self.label_unicode
            )
        if self.T_min <= 0:
            raise ValueError(f"T_min should be larger than zero. Got: {self.T_min}")
        if self.T_max <= self.T_min:
            raise ValueError(f"T_max ({self.T_max}) should be higher than T_min ({self.T_min}).")

        if gen_cs2:
            self.cs2 = self.gen_cs2()
        if gen_cs2_neg:
            self.cs2_neg = self.gen_cs2_neg()

    # Concrete methods

    def export(
            self,
            path: str | os.PathLike[str] | None = None,
            fields: FieldSpec = Preset.FULL) -> dict[str, tp.Any]:
        """Export the model parameters to a dictionary, and optionally save them as a JSON file.

        :param path: path of the JSON file
        :param fields: the fields to export, see :py:data:`pttools.utils.fields.FieldSpec`
        :return: the exported data
        """
        data = self.extract(fields)
        if path is not None:
            export_json(data, path)
        return data

    def gen_cs2(self) -> th.CS2Fun:
        r"""This function generates a Numba-jitted $c_s^2$ function for the model."""
        raise NotImplementedError("This class does not have gen_cs2 defined")

    def gen_cs2_neg(self) -> th.CS2Fun:
        r"""This function generates a negative version of the Numba-jitted $c_s^2$ function.

        The negative version is used for finding the maximum of $c_s^2$ with a minimization algorithm.
        """
        raise NotImplementedError("This class does not have gen_cs2_neg defined")

    def info(self) -> str:
        """Get a string with information about the model."""
        data = self.export()
        max_key_length = max(len(key) for key in data) + 1
        return "\n".join(
            f"{key:<{max_key_length}}: {f'{value:{self.THERMO_FORMAT}}' if isinstance(value, float) else value}"
            for key, value in self.export().items()
        )

    def validate_temp[T: FloatOrArr](self, temp: T) -> T:
        """Validate that the given temperatures are in the validity range of the model.

        If invalid values are found, a copy of the array is created where those are set to np.nan.
        """
        t_min = self.T_min * (1 - self.TEMP_RTOL)
        t_max = self.T_max * (1 + self.TEMP_RTOL)
        if np.isscalar(temp):
            return tp.cast(T, self._validate_temp_scalar(tp.cast(float, temp), t_min, t_max))
        # np.isscalar() does not narrow the type for the type checker.
        return tp.cast(T, self._validate_temp_arr(tp.cast(th.FloatArr, temp), t_min, t_max))

    def _validate_temp_scalar(self, temp: float, t_min: float, t_max: float) -> float:
        if temp < t_min:
            if not self.silence_temp:
                logger.warning(
                    "The temperature %s is below the minimum temperature %s of the model \"%s\".",
                    temp, self.T_min, self.name
                )
            if self.restrict_to_valid:
                return np.nan
        elif temp > t_max:
            if not self.silence_temp:
                logger.warning(
                    "The temperature %s is above the maximum temperature %s of the model \"%s\".",
                    temp, self.T_max, self.name
                )
            if self.restrict_to_valid:
                return np.nan
        return temp

    def _validate_temp_arr(self, temp: th.FloatArr, t_min: float, t_max: float) -> th.FloatArr:
        below = temp < t_min
        above = temp > t_max
        has_below = np.any(below)
        has_above = np.any(above)
        if self.restrict_to_valid and (has_below or has_above):
            temp = np.copy(temp)
        if has_below:
            if not self.silence_temp:
                logger.warning(
                    "Some temperatures (%s and possibly above) "
                    "are below the minimum temperature %s of the model \"%s\".",
                    np.min(temp), self.T_min, self.name
                )
            if self.restrict_to_valid:
                temp[below] = np.nan
        if has_above:
            if not self.silence_temp:
                logger.warning(
                    "Some temperatures (%s and possibly below) "
                    "are above the maximum temperature %s of the model \"%s\".",
                    np.nanmax(temp), self.T_max, self.name
                )
            if self.restrict_to_valid:
                temp[above] = np.nan
        return temp

    # Abstract methods

    @abc.abstractmethod
    def cs2(self, *args: tp.Any, **kwargs: tp.Any) -> th.FloatOrArr:
        """Speed of sound squared $c_s^2$."""

    @abc.abstractmethod
    def cs2_neg(self, *args: tp.Any, **kwargs: tp.Any) -> th.FloatOrArr:
        """Speed of sound squared with a minus sign, $-c_s^2$. This is needed for finding the maximum of $c_s^2$."""
