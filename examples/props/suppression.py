import matplotlib.pyplot as plt

from examples import utils
from pttools.analysis.suppression import SuppressionPlot
from pttools.omgw0.suppression import WITH_HYBRIDS, NO_HYBRIDS, NO_HYBRIDS_EXT


def main():
    return \
        SuppressionPlot(WITH_HYBRIDS), SuppressionPlot(NO_HYBRIDS), SuppressionPlot(NO_HYBRIDS_EXT), \
        SuppressionPlot(
            NO_HYBRIDS_EXT,
            v_wall_min=0.4, v_wall_max=0.9, alpha_n_max=0.5,
            title="", alpha_n_max_lines=False
        )


if __name__ == "__main__":
    fig_with_hybrids, fig_no_hybrids, fig_no_hybrids_ext, fig_no_hybrids_ext2 = main()
    utils.save(fig_with_hybrids.fig, "suppression_with_hybrids")
    utils.save(fig_no_hybrids.fig, "suppression_no_hybrids")
    utils.save(fig_no_hybrids_ext.fig, "suppression_no_hybrids_ext")
    utils.save(fig_no_hybrids_ext2.fig, "suppression_no_hybrids_ext_cropped")
    plt.show()
