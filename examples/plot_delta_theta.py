r"""
Delta-Theta
===========

$\Delta \theta$ surfaces
"""

import os.path

import numpy as np

from examples.utils import FIG_DIR
from pttools.analysis.plot_delta_theta import DeltaThetaPlot3D
from pttools.models.bag import BagModel
from pttools.models.const_cs import ConstCSModel

bag = BagModel(a_s=1.1, a_b=1, V_s=1)
css = 1/np.sqrt(3) - 0.01
csb = 1/np.sqrt(3) - 0.02
const_cs = ConstCSModel(a_s=1.5, a_b=1, css2=css**2, csb2=csb**2, V_s=1)

plot = DeltaThetaPlot3D()
plot.add(bag)
plot.add(const_cs)

plot.save(os.path.join(FIG_DIR, "plot_delta_theta"))
if __name__ == "__main__" and "__file__" in globals():
    plot.show()
plot.fig()
