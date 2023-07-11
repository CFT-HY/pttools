"""
Entropy profiles
================

Plot the entropy profiles of a few bubbles.
"""

from examples import utils
from pttools.analysis.plot_entropy import plot_entropy
from pttools.bubble.bubble import Bubble
from pttools.models.bag import BagModel


def main():
    model = BagModel(a_s=1.1, a_b=1, V_s=1)
    bubbles = [
        Bubble(model, v_wall=0.9, alpha_n=0.1),
        Bubble(model, v_wall=0.7, alpha_n=0.1),
        Bubble(model, v_wall=0.3, alpha_n=0.1)
    ]
    for bubble in bubbles:
        bubble.solve()

    return plot_entropy(bubbles, colors=["r", "g", "b"])


if __name__ == "__main__":
    fig = main()
    utils.save_and_show(fig, "entropy_profile.png")
