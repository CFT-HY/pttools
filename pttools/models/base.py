import abc
import logging

import numpy as np

from pttools.bubble.bag import CS2Fun
import pttools.type_hints as th

logger = logging.getLogger(__name__)


class BaseModel(abc.ABC):
    """The base for both Model and ThermoModel"""
    DEFAULT_NAME: str = None
    DEFAULT_T_MIN: float = 0
    DEFAULT_T_MAX: float = np.inf

    def __init__(
            self,
            name: str = None,
            t_min: float = None, t_max: float = None,
            restrict_to_valid: bool = True,
            gen_cs2: bool = True):
        self.name = self.DEFAULT_NAME if name is None else name
        self.t_min = self.DEFAULT_T_MIN if t_min is None else t_min
        self.t_max = self.DEFAULT_T_MAX if t_max is None else t_max
        self.restrict_to_valid = restrict_to_valid

        if self.name is None:
            raise ValueError("The model must have a name")
        if self.t_max <= self.t_min:
            raise ValueError(f"T_max ({self.t_max}) should be higher than T_min ({self.t_min})")

        self.cs2: CS2Fun = self.gen_cs2() if gen_cs2 else None

    # Concrete methods

    def validate_temp(self, temp: th.FloatOrArr) -> th.FloatOrArr:
        """Validate that the given temperatures are in the validity range of the model.

        If invalid values are found, a copy of the array is created where those are set to np.nan.
        """
        if np.isscalar(temp):
            if temp < self.t_min:
                logger.warning(
                    f"The temperature {temp} "
                    f"is below the minimum temperature {self.t_min} of the model \"{self.name}\""
                )
                if self.restrict_to_valid:
                    return np.nan
            elif temp > self.t_max:
                logger.warning(
                    f"The temperature {temp} "
                    f"is above the maximum temperature {self.t_max} of the model \"{self.name}\""
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
                    f"are below the minimum temperature {self.t_min} of the model \"{self.name}\""
                )
                if self.restrict_to_valid:
                    temp[below] = np.nan
            if has_above:
                logger.warning(
                    f"Some temperatures ({np.max(temp)} and possibly above) "
                    f"are below the minimum temperature {self.t_max} of the model \"{self.name}\""
                )
                if self.restrict_to_valid:
                    temp[above] = np.nan
        return temp

    # Abstract methods

    @abc.abstractmethod
    def cs2(self, *args, **kwargs) -> th.FloatOrArr:
        pass

    @abc.abstractmethod
    def gen_cs2(self) -> CS2Fun:
        r"""This function should generate a Numba-jitted $c_s^2$ function for the model."""
