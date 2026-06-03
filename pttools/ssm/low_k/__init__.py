r"""Low-wavenumber tail of the GW spectrum

This module is based on the analytic approximations introduced in
:giombi_2024_cs:`\ ` and
:giombi_2026:`\ `.
This assumes a barotropic equation of state (EoS) with $p(e) = \omega e$,
where $\omega$ is a constant EoS parameter.

For $\mathcal{P}_\text{gw}$:
low frequencies $k^3$,
intermediate frequencies $k^1$
before peak $k^9$
high frequencies $k^{-3}$.
"""

from .analytical import *
from .const import *
from .integration import *
from .intersection import *
from .join import *
from .kernel import *
