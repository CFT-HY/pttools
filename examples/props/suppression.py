r"""
Suppression factor $\Sigma({v}_\text{wall}, \alpha_n)$
======================================================

:gowling_2021:`\ ` fig. 10
"""

from examples.utils import save_and_show_figs
from pttools.analysis.suppression import SuppressionPlot
from pttools.ssm.suppression import WITH_HYBRIDS, NO_HYBRIDS, NO_HYBRIDS_EXT


def main() -> tuple[SuppressionPlot, SuppressionPlot, SuppressionPlot, SuppressionPlot]:
    r"""Plot the suppression factor $\Sigma({v}_\text{wall}, \alpha_n)$"""
    return \
        SuppressionPlot(WITH_HYBRIDS), SuppressionPlot(NO_HYBRIDS), SuppressionPlot(NO_HYBRIDS_EXT), \
        SuppressionPlot(
            NO_HYBRIDS_EXT,
            v_wall_min=0.4, v_wall_max=0.9, alpha_n_max=0.5,
            title="", alpha_n_max_lines=False
        )


if __name__ == "__main__":
    _fig_with_hybrids, _fig_no_hybrids, _fig_no_hybrids_ext, _fig_no_hybrids_ext2 = main()
    save_and_show_figs({
        "suppression_with_hybrids": _fig_no_hybrids.fig,
        "suppression_no_hybrids": _fig_no_hybrids.fig,
        "suppression_no_hybrids_ext": _fig_no_hybrids_ext.fig,
        "suppression_no_hybrids_ext_cropped": _fig_no_hybrids_ext2.fig
    })
