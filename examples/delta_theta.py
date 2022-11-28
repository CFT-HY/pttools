import os.path

import numpy as np

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

fig = plot.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_delta_theta"))
if "__file__" in globals():
    fig = plot.create_fig()
    fig.show()
fig
