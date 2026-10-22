"""
Comparison of BagModel and ConstCSModel
=======================================
"""

import numpy as np

from examples.utils import save_and_show_fig
from pttools import models
from pttools.plot.models import ModelsPlot
from pttools.utils.time import log_duration


@log_duration
def main() -> ModelsPlot:
    """Comparison of BagModel and ConstCSModel"""
    model_bag = models.BagModel.cached(a_s=1.1, a_b=1, V_s=1)
    # model_const_cs_like_bag = models.ConstCSModel(a_s=1.1, a_b=1, css2=1/3, csb2=1/3, V_s=1)
    model_thermo_bag = models.FullModel.cached(
        thermo=models.ConstCSThermoModel.cached(a_s=1.1, a_b=1, V_s=1, css2=1/3, csb2=1/3)
    )
    model_sm = models.FullModel.cached(thermo=models.StandardModel.cached(), allow_invalid=True)

    plot = ModelsPlot(temp=np.logspace(1, 2.5, 100))
    plot.add(model_bag, phase=models.Phase.SYMMETRIC)
    plot.add(model_thermo_bag, phase=models.Phase.SYMMETRIC, ls="--")
    plot.add(model_sm, phase=models.Phase.SYMMETRIC, ls=":")
    plot.process()
    return plot


if __name__ == "__main__":
    _plot = main()
    save_and_show_fig(_plot.fig, "model_comparison")
