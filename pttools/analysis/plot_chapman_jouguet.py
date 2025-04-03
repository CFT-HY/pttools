"""Plot Chapman-Jouguet speed"""

import typing as tp

from matplotlib.legend import Legend
import matplotlib.pyplot as plt
import numpy as np

from pttools.analysis.utils import legend
from pttools.bubble.chapman_jouguet import v_chapman_jouguet
from pttools.models.model import Model


class ChapmanJouguetPlot:
    def __init__(self, alpha_n: np.ndarray):
        self.alpha_n = alpha_n

        self.fig: plt.Figure = plt.figure()
        self.ax = self.fig.add_subplot()

        self.ax.set_xlabel(r"$\alpha_n$")
        self.ax.set_ylabel("$v_{CJ}$")
        self.fig.tight_layout()

    def add(self, model: Model, analytical: bool = True, label: str = None, ls: str = "-"):
        v_cj = np.empty_like(self.alpha_n)
        for i in range(self.alpha_n.size):
            v_cj[i] = v_chapman_jouguet(alpha_n=self.alpha_n[i], model=model, analytical=analytical)

        if label is None:
            label = model.label_latex
        self.ax.plot(self.alpha_n, v_cj, label=label, ls=ls)

    def process(self) -> tp.Optional[Legend]:
        return legend(self.ax, fontsize="x-small")
