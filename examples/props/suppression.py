r"""
Suppression factor
==================
Suppression factor $\Sigma({v}_\text{wall}, \alpha_n)$

:gowling_2021:`\ ` fig. 10

The absolute and relative differences between the suppression datasets
:py:data:`pttools.ssm.suppression.suppression.WITH_HYBRIDS` and :py:data:`pttools.ssm.suppression.suppression.GH21`
are plotted within the convex hull of their data points.
"""

import numpy as np
from pandas import DataFrame

from examples.utils import save_and_show_figs
from pttools.plot.suppression import SuppressionPlot
from pttools.plot.suppression_comparison import SuppressionComparisonPlot
from pttools.plot.vw_alpha import VwAlphaPlot
from pttools.ssm.suppression import GH21, SUPPRESSIONS, WITH_HYBRIDS
from pttools.utils.time import log_duration


@log_duration
def main() -> dict[str, VwAlphaPlot]:
    r"""Plot the suppression factor $\Sigma({v}_\text{wall}, \alpha_n)$ and the differences between datasets"""
    df = DataFrame({
        "name": [supp.name for supp in SUPPRESSIONS],
        "min_supp": [supp.suppressions.min() for supp in SUPPRESSIONS],
        "max_supp": [supp.suppressions.max() for supp in SUPPRESSIONS]
    })
    min_supp = df.min_supp.min()
    max_supp = df.max_supp.max()
    print(df)
    print(f"Minimum of all suppressions: {min_supp}")
    print(f"Maximum of all suppressions: {max_supp}")

    step = 0.25
    levels = np.arange(0, max_supp + step, step=step)
    plots: dict[str, VwAlphaPlot] = {}
    plots |= {
        f"suppression_{suppression.name}": SuppressionPlot(
            suppression, levels=levels
        ) for suppression in SUPPRESSIONS
    }
    plots |= {
        f"suppression_{suppression.name}_cropped": SuppressionPlot(
            suppression, v_wall_min=0.4, v_wall_max=0.9, alpha_n_max=0.5, alpha_n_max_lines=False, levels=levels
        ) for suppression in SUPPRESSIONS
    }
    plots |= {
        f"suppression_diff_{'rel' if relative else 'abs'}_{WITH_HYBRIDS.name}_{GH21.name}": SuppressionComparisonPlot(
            WITH_HYBRIDS, GH21, relative=relative
        ) for relative in (False, True)
    }
    return plots


if __name__ == "__main__":
    save_and_show_figs({name: plot.fig for (name, plot) in main().items()})
