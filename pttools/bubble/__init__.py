r"""
Functions for calculating fluid profile around expanding Higgs-phase bubble.

Finds, analyses and plots self-similar functions $v$ (radial fluid velocity)
and $w$ (fluid enthalpy) as functions of the scaled variable $\xi = r/t$.
Main inputs are wall speed $v_w$ and global transition strength parameter
$\alpha_n$.

See Espinosa et al 2010, Hindmarsh & Hijazi 2019.

Authors: Mark Hindmarsh 2015-20, with Mudhahir Al-Ajmi, and contributions from:
Danny Bail (Sussex MPhys RP projects 2016-18); Jacky Lindsay and Mike Soughton (MPhys project 2017-18)

Changes planned at 06.20:
^^^^^^^^^^^^^^^^^^^^^^^^^
- allow general equation of state (so integrate with $V, T$ together instead of $v, w$ separately)
  Idea to introduce eos as a class. Need a new interface which uses eos variables rather than alpha.
- Include bubble nucleation calculations of beta (from $V(T,\phi)$)
- Now comments are docstrings, think about sphinx
- Complete checks for physical ($v_\text{wall}, \alpha_n$)

Changes 06.20:
^^^^^^^^^^^^^^
- Small improvements to docstrints.
- Start introducing checks for physical ($v_\text{wall}, \alpha_n$): check_wall_speed, check_physical_parameters

"""

from .alpha import *
from .approx import *
from .bag import *
from .boundary import *
from .check import *
from .const import *
from .fluid import *
from .plot import *
from .props import *
from .quantities import *
from .relativity import *
from .transition import *
