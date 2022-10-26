import matplotlib.pyplot as plt
import numpy as np

from pttools import bubble, models


class ModelComparison:
    def __init__(self, temp: np.ndarray, phase: bubble.Phase.BROKEN):
        self.temp = temp
        self.phase = phase

        fig: plt.Figure = plt.figure(figsize=(11.69, 8.27))
        self.axs: np.ndarray = fig.subplots(nrows=3, ncols=3)
        self.ax_p = self.axs[0, 0]
        self.ax_s = self.axs[0, 1]
        self.ax_w = self.axs[0, 2]
        self.ax_e = self.axs[1, 0]
        self.ax_temp = self.axs[1, 1]
        self.ax_cs2 = self.axs[1, 2]
        self.ax_alpha_n = self.axs[2, 1]

        for ax in [self.ax_p, self.ax_s, self.ax_w, self.ax_e]:
            ax.set_yscale("log")

        for ax in self.axs.flat:
            ax.set_xlabel("T")
        self.ax_p.set_ylabel("p")
        self.ax_s.set_ylabel("s")
        self.ax_w.set_ylabel("w")
        self.ax_e.set_ylabel("e")

        self.ax_temp.set_xlabel("w")
        self.ax_temp.set_ylabel("T")
        self.ax_cs2.set_ylabel("$c_s^2$")
        self.ax_alpha_n.set_ylabel(r"$\alpha_n$")

        fig.tight_layout()

    def add(self, model: models.Model, ls="-"):
        w = model.w(self.temp, self.phase)

        self.ax_p.plot(self.temp, model.p_temp(self.temp, self.phase), label=model.label, ls=ls)
        self.ax_s.plot(self.temp, model.s_temp(self.temp, self.phase), label=model.label, ls=ls)
        self.ax_w.plot(self.temp, w, label=model.label, ls=ls)
        self.ax_e.plot(self.temp, model.e_temp(self.temp, self.phase), label=model.label, ls=ls)

        self.ax_temp.plot(w, model.temp(w, self.phase), label=model.label, ls=ls)
        self.ax_cs2.plot(self.temp, model.cs2(w, self.phase), label=model.label, ls=ls)
        self.ax_alpha_n.plot(self.temp, model.alpha_n(w), label=model.label, ls=ls)

    def process(self):
        for ax in self.axs.flat:
            ax.legend()


def main():
    model_bag = models.BagModel(a_s=1.1, a_b=1, V_s=1)
    # model_const_cs_like_bag = models.ConstCSModel(a_s=1.1, a_b=1, css2=1/3, csb2=1/3, V_s=1)
    model_thermo_bag = models.FullModel(thermo=models.ConstCSThermoModel(a_s=1.1, a_b=1, V_s=1, css2=1/3, csb2=1/3))

    comp = ModelComparison(
        temp=np.linspace(1, 100, 100),
        phase=models.Phase.SYMMETRIC
    )
    comp.add(model_bag)
    comp.add(model_thermo_bag, ls="--")
    # comp.process()


if __name__ == "__main__":
    main()
    plt.show()
