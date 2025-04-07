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
from .bubble_quantities import *
from .chapman_jouguet import *
from .check import *
from .const import *
from .fluid import *
from .fluid_bag import *
from .giese import *
from .integrate import *
# from .physical_params import *
from .props import *
from .quantities import *
from .relativity import *
from .shock import *
from .transition import *
from .trim import *
