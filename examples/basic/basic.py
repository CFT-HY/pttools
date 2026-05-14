"""
Basic usage
===========

Plot a single bubble
"""

import os.path

import matplotlib.pyplot as plt

from examples.utils import FIG_DIR, save
from pttools.bubble import Bubble
from pttools.models import BagModel
from pttools.ssm import NucType
from pttools.omgw0 import Spectrum


def main():
    """Plot a single bubble"""
    # Create the equation of state.
    # If you don't specify a_s and a_b or g_s and g_b,
    # you have to specify a minimum alpha_n for which the model will be valid.
    model = BagModel(alpha_n_min=0.01)

    # Create and simulate the fluid profile of a bubble.
    bubble = Bubble(model, v_wall=0.5, alpha_n=0.2)
    bubble_fig = bubble.plot()
    save(bubble_fig, "bag_bubble.png")
    bubble.export(os.path.join(FIG_DIR, "bubble.json"))

    # Compute the gravitational wave spectrum for the bubble.
    spectrum = Spectrum(bubble, nuc_type=NucType.EXPONENTIAL, r_star=0.1)
    spectrum_fig, axs = spectrum.plot_multi()
    save(spectrum_fig, "bag_spectrum.png")
    bubble.export(os.path.join(FIG_DIR, "spectrum.json"))


if __name__ == "__main__":
    main()
    plt.show()
