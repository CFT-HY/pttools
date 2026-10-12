r"""
Suppression factor
==================
Suppression factor $\Sigma({v}_\text{wall}, \alpha_n)$

:gowling_2021:`\ ` fig. 10
"""

import numpy as np
from pandas import DataFrame

from examples.utils import save_and_show_figs
from pttools.analysis.suppression import SuppressionPlot
from pttools.ssm.suppression import SUPPRESSIONS


def main() -> dict[str, SuppressionPlot]:
    r"""Plot the suppression factor $\Sigma({v}_\text{wall}, \alpha_n)$"""
    df = DataFrame({
        "name": [supp.name for supp in SUPPRESSIONS],
        "max_supp": [supp.suppressions.max() for supp in SUPPRESSIONS]
    })
    max_supp = df.max_supp.max()
    print(df)
    print(f"Maximum of all suppressions: {max_supp}")

    step = 0.25
    levels = np.arange(0, max_supp + step, step=step)
    return {
        f"suppression_{suppression.name}": SuppressionPlot(
            suppression, levels=levels
        ) for suppression in SUPPRESSIONS
    } | {
        f"suppression_{suppression.name}_cropped": SuppressionPlot(
            suppression, v_wall_min=0.4, v_wall_max=0.9, alpha_n_max=0.5, alpha_n_max_lines=False, levels=levels
        ) for suppression in SUPPRESSIONS
    }


if __name__ == "__main__":
    save_and_show_figs({name: plot.fig for (name, plot) in main().items()})
