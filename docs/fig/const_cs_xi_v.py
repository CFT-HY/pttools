r"""$(\xi, v)$ plane figure for the constant sound speed model"""

import matplotlib.pyplot as plt
import numpy as np

from pttools.bubble.fluid import add_df_dtau, fluid_shell, fluid_shell_generic
from pttools.bubble.boundary import SolutionType
from pttools.bubble.relativity import lorentz
from pttools.models.const_cs import ConstCSModel
from pttools.models.const_cs_thermo import ConstCSThermoModel
from pttools.models.full import FullModel
from tests.paper.plane import xiv_plane
from tests.paper.plot_plane import plot_plane


def main():
    model = ConstCSModel(a_s=1.1, a_b=1, css2=0.4**2, csb2=1/3, V_s=1, V_b=0)
    df_dtau_ptr = add_df_dtau("const_cs", model.cs2)
    data_b, data_s = xiv_plane(df_dtau_ptr=df_dtau_ptr, cs2_s=model.css2, cs2_b=model.csb2)
    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot()
    plot_plane(ax=ax, data_b=data_b, data_s=data_s, cs2_s=model.css2, cs2_b=model.csb2, selected_solutions=False)

    xi = np.linspace(0.4, 1, 20)
    ax.plot(xi, lorentz(xi, 0.4), c="r", ls="-.", label=r"$\mu(\xi,c_{sb}=0.4)$")

    model2 = ConstCSModel(a_s=1.1, a_b=1, css2=1/3, csb2=1/3, V_s=1, V_b=0)
    v, w, xi = fluid_shell_generic(model2, v_wall=0.85, alpha_n=0.05, sol_type=SolutionType.DETON)
    ax.plot(xi, v, c="b", label=r"$c_{sb}=\frac{1}{\sqrt{3}}$")
    v, w, xi = fluid_shell(v_wall=0.85, alpha_n=0.05)
    ax.plot(xi, v, c="g", label="bag", ls=":")
    v, w, xi = fluid_shell_generic(model2, v_wall=0.5, alpha_n=0.578, sol_type=SolutionType.SUB_DEF, reverse=False)
    ax.plot(xi, v, c="b")
    v, w, xi = fluid_shell(v_wall=0.5, alpha_n=0.578)
    ax.plot(xi, v, c="g", ls=":")

    # model3 = ConstCSModel(a_s=1.1, a_b=1, css2=1/3, csb2=0.4**2, V_s=1, V_b=0)
    model3 = ConstCSModel(a_s=1.1, a_b=1, css2=0.4**2, csb2=1/3, V_s=1, V_b=0)
    v, w, xi = fluid_shell_generic(model3, v_wall=0.95, alpha_n=0.15, sol_type=SolutionType.DETON)
    ax.plot(xi, v, c="r", label=r"$c_{sb}=0.4$")
    v, w, xi = fluid_shell_generic(
        model3, v_wall=0.39, alpha_n=0.6, sol_type=SolutionType.SUB_DEF, reverse=False, allow_failure=True)
    ax.plot(xi, v, c="r")

    ax.legend()


if __name__ == "__main__":
    main()
    plt.show()
