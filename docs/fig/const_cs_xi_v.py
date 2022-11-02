r"""$(\xi, v)$ plane figure for the constant sound speed model"""

import matplotlib.pyplot as plt
import numpy as np

from pttools.bubble.fluid import add_df_dtau, fluid_shell_generic
from pttools.bubble.boundary import Phase, SolutionType
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

    model2 = FullModel(thermo=ConstCSThermoModel(a_s=1.1, a_b=1, css2=0.4**2, csb2=1/3, V_s=1, V_b=0))
    v, w, xi = fluid_shell_generic(model2, v_wall=0.8, alpha_n=0.15, sol_type=SolutionType.DETON)
    print(np.array([xi, v]).T)

    ax.plot(xi, v, c="b")


if __name__ == "__main__":
    main()
    plt.show()
