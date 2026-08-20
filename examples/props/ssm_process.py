"""
Sound Shell Model process
=========================

Plot the intermediate steps of the Sound Shell Model.
"""

from matplotlib.figure import Figure
import numpy as np

from examples.utils import save_and_show_fig
from pttools.bubble import Bubble
from pttools.models import ConstCSModel
from pttools.ssm import NucType
from pttools.omgw0 import DEFAULT_T_STAR, Spectrum
from pttools.utils import as_latex


def main() -> Figure:
    model = ConstCSModel(css2=1/4, csb2=1/4, alpha_n_min=0.1)

    # Create and simulate the fluid profile of a bubble.
    bubble = Bubble(model, v_wall=0.6, alpha_n=0.1, label_with_model=False)

    # Compute the gravitational wave spectrum for the bubble.
    spectrum = Spectrum(
        bubble, nuc_type=NucType.EXPONENTIAL, r_star=0.1, low_k=False,
        y=np.logspace(-0.5, 3, 1000),
        label_latex=f"$T_*={as_latex(DEFAULT_T_STAR)}$"
    )
    fig, axs = spectrum.plot_multi_flat(legend=True)
    return fig


if __name__ == "__main__":
    _fig = main()
    save_and_show_fig(_fig, "ssm_process")
