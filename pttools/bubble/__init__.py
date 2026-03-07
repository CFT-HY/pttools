r"""
This module contains the simulation framework for computing the fluid velocity profile
as a function of the radius of a self-similar bubble in a relativistic fluid.

Finds and analyses self-similar functions $v$ (radial fluid velocity)
and $w$ (fluid enthalpy) as functions of the scaled radius $\xi = r/t$.
Main inputs are the wall speed $v_w$ and the global transition strength parameter $\alpha_n$.
"""

from .alpha import *
from .approx import *
from .bag import *
from .boundary import *
from .bubble import *
from .chapman_jouguet import *
from .check import *
from .const import *
from .fluid import *
from .fluid_bag import *
from .fluid_reference import *
# from .gksvdv import *
from .integrate import *
# from .physical_params import *
from .phase import *
from .props import *
from .quantities import *
from .relativity import *
from .shock import *
from .solution_type import *
from .solution_type_bag import *
from .trim import *
from .v_minus import *
from .v_plus import *
