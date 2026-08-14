"""
Basic usage
===========

Plot a single bubble
"""

import os.path

import matplotlib.pyplot as plt

from examples.utils import FIG_DIR, save_fig
from pttools.bubble import Bubble
from pttools.models import BagModel
from pttools.ssm import NucType
from pttools.omgw0 import Spectrum


def main() -> tuple[plt.Figure, plt.Figure]:
    """Plot a single bubble"""
    # Create the equation of state.
    # If you don't specify a_s and a_b or g_s and g_b,
    # you have to specify a minimum alpha_n for which the model will be valid.
    model = BagModel(alpha_n_min=0.01)

    # Create and simulate the fluid profile of a bubble.
    bubble = Bubble(model, v_wall=0.5, alpha_n=0.2)
    save_fig(bubble_fig, "bag_bubble")
    bubble.export(os.path.join(FIG_DIR, "bag_bubble.json"))

    # Compute the gravitational wave spectrum for the bubble.
    spectrum = Spectrum(bubble, nuc_type=NucType.EXPONENTIAL, r_star=0.1)
    spectrum_fig, axs = spectrum.plot_multi()
    save_fig(spectrum_fig, "bag_spectrum")
    bubble.export(os.path.join(FIG_DIR, "bag_spectrum.json"))

    return bubble_fig, spectrum_fig


if __name__ == "__main__":
    main()
    plt.show()
