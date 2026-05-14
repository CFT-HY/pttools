"""Constants for the low-k module"""

import math


IV_ANALYTICAL: float = 1 / (32 * math.pi ** 2)
r"""Source contribution $\mathcal{I}_v$ (analytical approximation)
$$\mathcal{I}_v = \frac{1}{32 \pi^2}$$
:giombi_2024_cs:`\ ` p. 11
:giombi_2026:`\ ` p. 22
"""

JV_ANALYTICAL: float = 1 / (128 * math.pi ** 4)
r"""$\mathcal{J}_v$ (analytical approximation)
$$\mathcal{J}_v = \frac{\mathcal{I}_v}{2\pi^2} = \frac{1}{128 \pi^4}$$
:giombi_2026:`\ ` p. 22
"""
