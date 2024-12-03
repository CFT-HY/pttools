"""
ConstCSModel testing
====================

Simple testing for the constant sound speed model
"""

import matplotlib.pyplot as plt

from pttools.bubble import Bubble
from pttools.models import ConstCSModel
from pttools.omgw0 import Spectrum

# Specify the equation of state
const_cs = ConstCSModel(
    a_s=1.5, a_b=1, css2=1/3, csb2=1/3-0.01, V_s=1
)

# Create a bubble and solve its fluid profile
bubble = Bubble(const_cs, v_wall=0.5, alpha_n=0.2)
bubble.plot()

# Compute gravitational wave spectrum for the bubble
spectrum = Spectrum(bubble)
spectrum.plot_multi()

plt.show()
