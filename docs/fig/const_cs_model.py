r"""$(\xi, v)$ plane figure for the constant sound speed model"""

import matplotlib.pyplot as plt

from pttools.bubble.fluid import add_df_dtau
from pttools.models.const_cs import ConstCSModel
from tests.paper.plane import xiv_plane
from tests.paper.plot_plane import plot_plane


def main():
    model = ConstCSModel(a_s=1, a_b=1, css2=0.4**2, csb2=1/3, eps=1)
    df_dtau_ptr = add_df_dtau("const_cs", model.cs2)
    data_b, data_s = xiv_plane(df_dtau_ptr=df_dtau_ptr, cs2_s=model.css2, cs2_b=model.csb2)
    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot()
    plot_plane(ax=ax, data_b=data_b, data_s=data_s, cs2_s=model.css2, cs2_b=model.csb2, selected_solutions=False)


if __name__ == "__main__":
    main()
    plt.show()
