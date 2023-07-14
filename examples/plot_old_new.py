"""
Old-new comparison
============

Comparison of old and new solvers
"""

import matplotlib.pyplot as plt
import numpy as np

from examples import utils
from pttools.bubble import boundary
from pttools.bubble.boundary import Phase, SolutionType
from pttools.bubble.bubble import Bubble
from pttools.bubble import fluid_bag
from pttools.bubble import relativity
from pttools.models.model import Model
from pttools.models.bag import BagModel
from tests.paper.plane import xiv_plane
from tests.paper.plot_plane import plot_plane


def validate(model: Model, v: np.ndarray, w: np.ndarray, xi: np.ndarray, sol_type: SolutionType):
    if sol_type == SolutionType.SUB_DEF:
        validate_def(model, v, w, xi, sol_type)
    elif sol_type == SolutionType.HYBRID:
        validate_def(model, v, w, xi, sol_type)
        validate_shock(model, v, w, xi, sol_type)
    elif sol_type == SolutionType.DETON:
        validate_shock(model, v, w, xi, sol_type)


def validate_def(model: Model, v: np.ndarray, w: np.ndarray, xi: np.ndarray, sol_type: SolutionType):
    i_wall = np.argmax(v)
    v_wall = xi[i_wall]
    v1p = v[i_wall-1]
    v2p = v[i_wall]
    v1w = -relativity.lorentz(v1p, v_wall)
    v2w = -relativity.lorentz(v2p, v_wall)
    w1 = w[i_wall-1]
    w2 = w[i_wall]
    validate2(model, v1p, v2p, v1w, v2w, w1, w2, Phase.BROKEN, Phase.SYMMETRIC, sol_type)


def validate_shock(model: Model, v: np.ndarray, w: np.ndarray, xi: np.ndarray, sol_type: SolutionType):
    v_wall = xi[-2]
    v1p = v[-3]
    v2p = 0
    v1w = -relativity.lorentz(v1p, v_wall)
    v2w = -relativity.lorentz(v2p, v_wall)
    w1 = w[-3]
    w2 = w[-2]
    if sol_type == SolutionType.DETON:
        phase1 = Phase.BROKEN
        phase2 = Phase.SYMMETRIC
    else:
        phase1 = Phase.SYMMETRIC
        phase2 = Phase.SYMMETRIC
    validate2(model, v1p, v2p, v1w, v2w, w1, w2, phase1, phase2, sol_type)


def validate2(
        model: Model,
        v1p: float, v2p: float,
        v1w: float, v2w: float,
        w1: float, w2: float,
        phase1: Phase, phase2: Phase,
        sol_type: SolutionType):
    dev = boundary.junction_conditions_solvable(np.array([v2w, w2]), model, v1w, w1, phase1, phase2)
    print(f"sol_type={sol_type}, v1p={v1p}, v2p={v2p}, v1w={v1w}, v2w={v2w}, w1={w1}, w2={w2}, dev={dev}")


def main():
    bag = BagModel(a_s=1.1, a_b=1, V_s=2)

    v_walls = [0.5, 0.7, 0.77]
    alpha_ns = [0.578, 0.151, 0.091]
    sol_types = [SolutionType.SUB_DEF, SolutionType.HYBRID, SolutionType.DETON]

    bag_def = Bubble(bag, v_wall=v_walls[0], alpha_n=alpha_ns[0], sol_type=sol_types[0])
    bag_hybrid = Bubble(bag, v_wall=v_walls[1], alpha_n=alpha_ns[1], sol_type=sol_types[1])
    bag_det = Bubble(bag, v_wall=v_walls[2], alpha_n=alpha_ns[2], sol_type=sol_types[2])

    data = xiv_plane(separate_phases=False)
    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot()
    plot_plane(ax=ax, data_s=data, selected_solutions=False)

    print("Solving old")
    for v_wall, alpha_n, sol_type in zip(v_walls, alpha_ns, sol_types):
        v, w, xi = fluid_bag.fluid_shell_bag(v_wall=v_wall, alpha_n=alpha_n)
        ax.plot(xi, v, color="blue", label=rf"$v_w={v_wall}, \alpha_n={alpha_n}")
        validate(bag, v, w, xi, sol_type)

    print("Solving new")
    for bubble in [bag_def, bag_hybrid, bag_det]:
        bubble.solve()
        ax.plot(bubble.xi, bubble.v, ls=":", color="red")
        validate(bag, bubble.v, bubble.w, bubble.xi, bubble.sol_type)

    # ax.legend()
    return fig


if __name__ == "__main__":
    fig = main()
    utils.save_and_show(fig, "old_new.png")
