r"""
Delta-Theta
===========

Plot $\Delta \theta$ surfaces as a function of $w_+$ and $w_-$.
"""

import numpy as np

from examples.utils import FIG_DIR
from pttools.analysis.plot_delta_theta import DeltaThetaPlot3D
from pttools.models.bag import BagModel
from pttools.models.const_cs import ConstCSModel


def main() -> DeltaThetaPlot3D:
    r"""Plot $\Delta \theta$ surfaces as a function of $w_+$ and $w_-$."""
    bag = BagModel(a_s=1.1, a_b=1, V_s=1)
    css = 1/np.sqrt(3) - 0.01
    csb = 1/np.sqrt(3) - 0.02
    const_cs = ConstCSModel(a_s=1.5, a_b=1, css2=css**2, csb2=csb**2, V_s=1)

    plot = DeltaThetaPlot3D()
    plot.add(bag)
    plot.add(const_cs)
    return plot


if __name__ == "__main__":
    plot: DeltaThetaPlot3D = main()
    plot.save(FIG_DIR / "plot_delta_theta")
    # Sphinx-Gallery runs the examples without __file__, and the figure is then shown by the expression below.
    if "__file__" in globals():
        plot.show()

# Sphinx-Gallery shows the figure that is the value of the last expression of the example.
# The expression has to be at the top level, and is therefore conditional
# instead of being within the block above, where plot is defined.
plot.fig() if __name__ == "__main__" else None  # pyrefly: ignore[unbound-name]
