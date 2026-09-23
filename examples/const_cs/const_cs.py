"""
ConstCSModel
============

Plot various quantities for the constant sound speed model
"""

from math import sqrt

from matplotlib.figure import Figure

from examples.utils import save_and_show_figs
from pttools.analysis.plot_model import ModelPlot
from pttools.bubble import Bubble
from pttools.models import ConstCSModel
from pttools.omgw0 import Spectrum
from pttools.ssm import NucType


def main() -> tuple[ModelPlot, Figure, Figure]:
    """Plot various quantities for the constant sound speed model"""
    csb = 1 / sqrt(3) - 0.01
    const_cs = ConstCSModel(a_s=1.5, a_b=1, css2=1/3, csb2=csb**2, V_s=1)

    model_plot = ModelPlot(const_cs, t_log=False, y_log=False)

    bubble = Bubble(const_cs, v_wall=0.5, alpha_n=0.2)
    bubble_fig = bubble.plot()

    spectrum = Spectrum(bubble, r_star=0.1, nuc_type=NucType.EXPONENTIAL)
    spectrum_fig, _axs = spectrum.plot_multi()

    return model_plot, bubble_fig, spectrum_fig


if __name__ == "__main__":
    _model_plot, _bubble_fig, _spectrum_fig = main()
    save_and_show_figs({
        "const_cs": _model_plot.fig,
        "const_cs_bubble": _bubble_fig,
        "const_cs_spectrum": _spectrum_fig
    })
