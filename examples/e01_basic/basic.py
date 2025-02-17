"""
Basic usage
===========

Plot a single bubble
"""

import matplotlib.pyplot as plt

from examples.utils import save
from pttools.bubble import Bubble
from pttools.models import BagModel
from pttools.ssmtools import NucType
from pttools.omgw0 import Spectrum


def main():
    model = BagModel(alpha_n_min=0.01)

    bubble = Bubble(model, v_wall=0.5, alpha_n=0.2)
    bubble_fig = bubble.plot()
    save(bubble_fig, "bag_bubble.png")

    spectrum = Spectrum(bubble, nuc_type=NucType.EXPONENTIAL)
    spectrum_fig = spectrum.plot_multi()
    save(spectrum_fig, "bag_spectrum.png")


if __name__ == "__main__":
    main()
    plt.show()
