r"""
Functions for calculating fluid velocity profile around expanding Higgs-phase bubble.

Finds, analyses and plots self-similar functions $v$ (radial fluid velocity)
and $w$ (fluid enthalpy) as functions of the scaled radius $\xi = r/t$.
Main inputs are wall speed $v_w$ and global transition strength parameter $\alpha_n$.
"""

from .alpha import *
from .approx import *
from .bag import *
from .boundary import *
from .chapman_jouguet import *
from .check import *
from .const import *
from .fluid import *
# from .physical_params import *
from .props import *
from .quantities import *
from .relativity import *
from .transition import *
