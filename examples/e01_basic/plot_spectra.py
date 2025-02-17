"""
Plot multiple spectra
=================

Plot the velocity and GW spectra of multiple bubbles
"""

from pttools.analysis import plot_spectra

from examples import utils
from pttools.bubble import Bubble
from pttools.models import BagModel, ConstCSModel
from pttools.omgw0 import Spectrum


def main():
    model1 = BagModel(alpha_n_min=0.1)
    model2 = ConstCSModel(css2=1/4, csb2=1/3, a_s=1.5, a_b=1, V_s=1)
    bubbles = [
        Bubble(model1, v_wall=0.5, alpha_n=0.2),
        Bubble(model2, v_wall=0.5, alpha_n=0.2)
    ]
    spectra = [Spectrum(bubble) for bubble in bubbles]

    return plot_spectra(spectra)


if __name__ == "__main__":
    fig = main()
    utils.save_and_show(fig, "spectra.png")
