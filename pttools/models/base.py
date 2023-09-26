"""Base class for equation of state models and thermodynamics models"""

import abc
import datetime
import logging
import typing as tp

import numpy as np

import pttools.type_hints as th

logger = logging.getLogger(__name__)


class BaseModel(abc.ABC):
    """The base for both Model and ThermoModel"""
    DEFAULT_LABEL_LATEX: str = None
    DEFAULT_LABEL_UNICODE: str = None
    DEFAULT_NAME: str = None
    # Zero temperature would break many of the equations
    DEFAULT_T_MIN: float = 1e-3
    DEFAULT_T_MAX: float = np.inf

    def __init__(
            self,
            name: str = None,
            t_min: float = None, t_max: float = None,
            restrict_to_valid: bool = True,
            label_latex: str = None,
            label_unicode: str = None,
            gen_cs2: bool = True,
            gen_cs2_neg: bool = True):
        self.name = self.DEFAULT_NAME if name is None else name
        self.label_latex = self.DEFAULT_LABEL_LATEX if label_latex is None else label_latex
        self.label_unicode = self.DEFAULT_LABEL_UNICODE if label_unicode is None else label_unicode
        self.t_min = self.DEFAULT_T_MIN if t_min is None else t_min
        self.t_max = self.DEFAULT_T_MAX if t_max is None else t_max
        self.restrict_to_valid = restrict_to_valid

        if self.name is None:
            raise ValueError("The model must have a name.")
        if " " in self.name:
            logger.warning(
                "Model names should not have spaces to ensure that the file names don't cause problems. "
                f"Got: \"{self.name}\".")
        if not (self.label_latex and self.label_unicode):
            raise ValueError("The model must have labels.")
        if "$" in self.label_unicode:
            logger.warning(f"The Unicode label of a model should not contain \"$\". Got: \"{self.label_unicode}\"")
        if self.t_min <= 0:
            raise ValueError(f"T_min should be larger than zero. Got: {self.t_min}")
        if self.t_max <= self.t_min:
            raise ValueError(f"T_max ({self.t_max}) should be higher than T_min ({self.t_min}).")

        if gen_cs2:
            self.cs2 = self.gen_cs2()
        if gen_cs2_neg:
            self.cs2_neg = self.gen_cs2_neg()

    # Concrete methods

    def export(self) -> tp.Dict[str, any]:
        """User-created model classes should extend this"""
        return {
            "name": self.name,
            "label_latex": self.label_latex,
            "label_unicode": self.label_unicode,
            "datetime": datetime.datetime.now(),
            "t_min": self.t_min,
            "t_max": self.t_max,
            "restrict_to_valid": self.restrict_to_valid
        }

    def gen_cs2(self) -> th.CS2Fun:
        r"""This function should generate a Numba-jitted $c_s^2$ function for the model."""
        raise NotImplementedError("This class does not have gen_cs2 defined")

    def gen_cs2_neg(self) -> th.CS2Fun:
        r"""This function should generate a negative version of
        the Numba-jitted $c_s^2$ function to be used for maximisation.
        """
        raise NotImplementedError("This class does not have gen_cs2_neg defined")

    def validate_temp(self, temp: th.FloatOrArr) -> th.FloatOrArr:
        """Validate that the given temperatures are in the validity range of the model.

        If invalid values are found, a copy of the array is created where those are set to np.nan.
        """
        if np.isscalar(temp):
            if temp < self.t_min:
                logger.warning(
                    f"The temperature {temp} "
                    f"is below the minimum temperature {self.t_min} of the model \"{self.name}\"."
                )
                if self.restrict_to_valid:
                    return np.nan
            elif temp > self.t_max:
                logger.warning(
                    f"The temperature {temp} "
                    f"is above the maximum temperature {self.t_max} of the model \"{self.name}\"."
                )
                if self.restrict_to_valid:
                    return np.nan
        else:
            below = temp < self.t_min
            above = temp > self.t_max
            has_below = np.any(below)
            has_above = np.any(above)
            if self.restrict_to_valid and (has_below or has_above):
                temp = np.copy(temp)
            if has_below:
                logger.warning(
                    f"Some temperatures ({np.min(temp)} and possibly above) "
                    f"are below the minimum temperature {self.t_min} of the model \"{self.name}\"."
                )
                if self.restrict_to_valid:
                    temp[below] = np.nan
            if has_above:
                # raise ValueError(temp)
                logger.warning(
                    f"Some temperatures ({np.max(temp)} and possibly above) "
                    f"are above the maximum temperature {self.t_max} of the model \"{self.name}\"."
                )
                if self.restrict_to_valid:
                    temp[above] = np.nan
        return temp

    # Abstract methods

    @abc.abstractmethod
    def cs2(self, *args, **kwargs) -> th.FloatOrArr:
        pass

    @abc.abstractmethod
    def cs2_neg(self, *args, **kwargs) -> th.FloatOrArr:
        pass
