import matplotlib as mpl
import matplotlib.pyplot as plt


def setup_plotting(font: str = "serif", font_size: int = 20, usetex: bool = True):
    """Set up plotting

    LaTeX can cause problems if system not configured correctly
    """
    plt.rc('text', usetex=usetex)
    plt.rc('font', family=font)

    mpl.rcParams.update({'font.size': font_size})
    mpl.rcParams.update({'lines.linewidth': 1.5})
    mpl.rcParams.update({'axes.linewidth': 2.0})
    mpl.rcParams.update({'axes.labelsize': font_size})
    mpl.rcParams.update({'xtick.labelsize': font_size})
    mpl.rcParams.update({'ytick.labelsize': font_size})
    # but make legend smaller
    mpl.rcParams.update({'legend.fontsize': 14})
