r"""
Chapman-Jouguet speed
=====================

Plot $v_{CJ} (\alpha_n)$
"""

import numpy as np

from examples.utils import save_and_show_fig
from pttools import models
from pttools.plot.chapman_jouguet import ChapmanJouguetPlot
from pttools.utils.time import log_duration


@log_duration
def main() -> ChapmanJouguetPlot:
    r"""Plot $v_{CJ} (\alpha_n)$"""
    plot = ChapmanJouguetPlot(alpha_n=np.linspace(0.15, 0.3, 100))

    plot.add(models.BagModel.cached(a_s=1.1, a_b=1, V_s=1), analytical=False, label="Bag model (analytical)")
    plot.add(models.BagModel.cached(a_s=1.1, a_b=1, V_s=1), ls="--")
    plot.add(
        models.ConstCSModel.cached(a_s=1.1, a_b=1, css2=1/3, csb2=1/3, V_s=1),
        label="Constant $c_s$ model with bag coeff.",
        ls=":"
    )
    # Todo: adjust the parameters so that the models have valid critical temperatures
    plot.add(
        models.ConstCSModel.cached(a_s=1.1, a_b=1, css2=0.25, csb2=1/3, V_s=1),
        label="Constant $c_s$ model", ls="--"
    )
    plot.add(
        models.ConstCSModel.cached(a_s=1.1, a_b=1, css2=1/3, csb2=0.25, V_s=1, allow_invalid=True),
        label="Constant $c_s$ model v2", ls="-"
    )
    plot.add(
        models.FullModel.cached(
            thermo=models.ConstCSThermoModel.cached(a_s=1.1, a_b=1, css2=1/3, csb2=0.25, V_s=1, allow_invalid=True),
            allow_invalid=True
        ),
        ls=":"
    )
    plot.add(
        models.FullModel.cached(thermo=models.StandardModel.cached(V_s=10), allow_invalid=True),
        label="Standard Model with V"
    )
    plot.process()
    return plot


if __name__ == "__main__":
    _plot = main()
    save_and_show_fig(_plot.fig, "chapman_jouguet")
