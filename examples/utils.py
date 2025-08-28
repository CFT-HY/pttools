import os.path

import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
os.makedirs(FIG_DIR, exist_ok=True)


def save(fig: plt.Figure, path: str, **kwargs):
    has_extension = "." in path
    if not os.path.isabs(path):
        path = os.path.join(FIG_DIR, path)
    if has_extension:
        fig.savefig(path, **kwargs)
    else:
        for ext in ["eps", "pdf", "png", "svg"]:
            fig.savefig(f"{path}.{ext}", **kwargs)


def save_and_show(fig: plt.Figure, path: str):
    save(fig, path)
    plt.show()
