"""
Spectrum plotting
=================

Plot velocity and GW spectra
"""

from pttools.analysis import plot_spectra

from examples import utils
from pttools.bubble import Bubble
from pttools.models import BagModel, ConstCSModel
from pttools.ssmtools import SSMSpectrum


def main():
    a_s = 100
    a_b = 50
    V_s = 1e-3
    model1 = BagModel(a_s=a_s, a_b=a_b, V_s=V_s)
    model2 = ConstCSModel(css2=1/4, csb2=1/3, a_s=a_s, a_b=a_b, V_s=V_s)
    bubbles = [
        Bubble(model1, v_wall=0.5, alpha_n=0.2),
        Bubble(model2, v_wall=0.5, alpha_n=0.2)
    ]
    spectra = [SSMSpectrum(bubble) for bubble in bubbles]

    return plot_spectra(spectra)


if __name__ == "__main__":
    fig = main()
    utils.save_and_show(fig, "spectra.png")
