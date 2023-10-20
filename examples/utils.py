import os.path

import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
os.makedirs(FIG_DIR, exist_ok=True)


def save(fig: plt.Figure, path: str):
    if os.path.isabs(path):
        fig.savefig(path)
    else:
        fig.savefig(os.path.join(FIG_DIR, path))


def save_and_show(fig: plt.Figure, path: str):
    save(fig, path)
    plt.show()
