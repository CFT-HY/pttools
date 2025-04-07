r"""A plot of the $\xi-v$ plane"""

import matplotlib.pyplot as plt
import numpy as np

from pttools.bubble.boundary import Phase
from pttools.bubble import const
from pttools.bubble.integrate import fluid_integrate_param
from pttools.bubble import props
from pttools.bubble import shock
from pttools.models.const_cs import ConstCSModel
from pttools.models.model import Model


class XIVPlanePlot:
    r"""A plot of the $\xi-v$ plane"""
    def __init__(self, model: Model, fig: plt.Figure = None, ax: plt.Axes = None):
        self.fig: plt.Figure
        if fig is None:
            if ax is not None:
                raise ValueError("Cannot provide ax without fig")
            self.fig = plt.figure()
        else:
            self.fig = fig
        self.ax: plt.Axes = self.fig.add_subplot() if ax is None else ax
        self.model = model

        self.ax.plot([0, 1], [0, 1], c="k", ls=":", label=r"$v=\xi$")
        self.ax.grid()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel(r"$\xi$")
        self.ax.set_ylabel("$v$")

    def curves(
            self,
            wn: float,
            w0: float = None,
            w_mu: float = None,
            method: str = "odeint",
            n_curves_s: int = 6,
            n_curves_b: int = 6,
            n_curves_right: int = 3,
            n_xi: int = const.N_XI_DEFAULT,
            t_forwards_end: float = const.T_END_DEFAULT,
            t_backwards_end: float = -const.T_END_DEFAULT):
        """Add background curves to the plot"""
        if w0 is None:
            w0 = wn
        if w_mu is None:
            w_mu = wn
        csb2_mu = self.model.cs2(w_mu, Phase.BROKEN)
        csb_mu = np.sqrt(csb2_mu)

        xi0_step_s = 1 / (n_curves_s + 1)
        # xi0_step_b = (1 - np.sqrt(cs2_b)) / (n_xi0_b + 1)
        xi0_s = np.linspace(xi0_step_s, 1, n_curves_s, endpoint=False)
        xi0_b = np.linspace(csb2_mu, 1, n_curves_b, endpoint=False)

        xi_mu = np.linspace(csb2_mu, 1, 100)
        v_mu = props.v_max_behind(xi_mu, csb_mu)
        self.ax.plot(xi_mu, v_mu, label=r"$\mu$")

        xi_sh, v_sh = shock.v_shock_curve(model=self.model, wn=wn)
        print(xi_sh, v_sh)
        self.ax.plot(xi_sh, v_sh, label="$v_{sh}$")

        grey = (0.8, 0.8, 0.8)

        data_s = np.zeros((n_curves_s, 6, n_xi))
        for i, xi0 in enumerate(xi0_s):
            # Make lines starting from v = xi, forward and back
            # Curves below the v=xi line
            v_b, w_b, xi_b, _ = fluid_integrate_param(
                v0=xi0, w0=w0, xi0=xi0,
                t_end=t_backwards_end, n_xi=n_xi, df_dtau_ptr=self.model.df_dtau_ptr(),
                method=method, phase=Phase.SYMMETRIC
            )
            # Curves above the v=xi
            v_f, w_f, xi_f, _ = fluid_integrate_param(
                v0=xi0, w0=w0, xi0=xi0,
                t_end=t_forwards_end, n_xi=n_xi, df_dtau_ptr=self.model.df_dtau_ptr(),
                method=method, phase=Phase.SYMMETRIC
            )

            v_b[v_b < np.interp(xi_b, xp=xi_sh, fp=v_sh)] = np.nan
            data_s[i, :, :] = [
                v_b, w_b, xi_b,
                v_f, w_f, xi_f
            ]
            self.ax.plot(xi_b, v_b, c="k")
            self.ax.plot(xi_f, v_f, c=grey)

        data_b = np.zeros((n_curves_b, 3, n_xi))
        for i, xi0 in enumerate(xi0_b):
            v0 = props.v_max_behind(xi0, csb_mu)
            v, w, xi, _ = fluid_integrate_param(
                v0=v0, w0=w0, xi0=xi0,
                t_end=t_backwards_end, n_xi=n_xi, df_dtau_ptr=self.model.df_dtau_ptr(),
                method=method, phase=Phase.BROKEN
            )
            data_b[i, :, :] = [v, w, xi]
            self.ax.plot(xi, v, c="k")

        v0_step_right = 1 / (n_curves_right + 1)
        v0_right = np.linspace(v0_step_right, 1, n_curves_right, endpoint=False)
        data_right = np.zeros((n_curves_right, 3, n_xi))
        for i, v0 in enumerate(v0_right):
            v, w, xi, _ = fluid_integrate_param(
                v0=v0, w0=w0, xi0=1,
                t_end=t_backwards_end, n_xi=n_xi, df_dtau_ptr=self.model.df_dtau_ptr(),
                method=method, phase=Phase.BROKEN
            )
            data_right[i, :, :] = [v, w, xi]
            self.ax.plot(xi, v, c="k")

        self.ax.legend()


if __name__ == "__main__":
    from pttools.logging import setup_logging
    setup_logging()
    model = ConstCSModel(css2=1/3 - 0.05, csb2=1/3, a_s=1.5, a_b=1, V_s=1)
    plot = XIVPlanePlot(model)
    plot.curves(wn=model.wn(0.3))
    plot.fig.savefig("plane_test.png")
    plt.show()
