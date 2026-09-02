"""
Integration
=========

How to integrate a fluid shell profile manually
"""

import matplotlib.pyplot as plt

from pttools.bubble import \
    DEFAULT_FLUID_INTEGRATE_METHOD, DEFAULT_T_END, Phase, add_df_dtau, fluid_integrate_param


def cs2(w: float, phase: float) -> float:
    return 1/3 - 0.001 * phase


def main():
    df_dtau_ptr = add_df_dtau(name="test", cs2_fun=cs2)
    v, w, xi, t = fluid_integrate_param(
        v0=0.2, w0=1., xi0=0.3, phase=Phase.SYMMETRIC,
        # This chooses the direction of the integration
        t_end=-DEFAULT_T_END,
        # Add this argument if you have a non-bag EoS
        df_dtau_ptr=df_dtau_ptr,
        method=DEFAULT_FLUID_INTEGRATE_METHOD,
    )

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(xi, v)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


if __name__ == "__main__":
    main()
    plt.show()
