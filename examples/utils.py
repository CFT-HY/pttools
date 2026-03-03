"""Utilities for PTtools examples"""

import os.path

from matplotlib.figure import Figure
import matplotlib.pyplot as plt

FIG_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
os.makedirs(FIG_DIR, exist_ok=True)


def save(fig: Figure, path: str, **kwargs):
    """Save a figure in the examples figure directory"""
    has_extension = "." in path
    if not os.path.isabs(path):
        path = os.path.join(FIG_DIR, path)
    if has_extension:
        fig.savefig(path, **kwargs)
    else:
        for ext in ["eps", "pdf", "png", "svg"]:
            fig.savefig(f"{path}.{ext}", **kwargs)


def save_and_show(fig: Figure, path: str):
    """Save a figure in the examples figure directory and show it"""
    save(fig, path)
    plt.show()
