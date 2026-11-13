r"""Plotly figures of a bubble and a spectrum for the dataset viewer.

These correspond to the Matplotlib figures of :py:func:`pttools.plot.bubbles.plot_bubbles`
and :py:func:`pttools.plot.spectra.plot_spectra`, and use the same labels and axis limits.
The figures can be created for both the computed and the imported bubbles and spectra.
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pttools.math.formatting import as_unicode
from pttools.plot.bubbles import V_LABEL, W_LABEL, XI_LABEL, xi_limits
from pttools.plot.spectra import F_LABEL, POW_GW0_H2_LABEL, noise_curve, noise_label
import pttools.type_hints as th

if th.TYPE_CHECKING_OR_DOCS:
    from pttools.bubble.bubble import BaseBubble
    from pttools.omgw0 import BaseOmgw0

__all__ = [
    "UIREVISION",
    "bubble_fig",
    "empty_fig",
    "spectrum_fig",
]

#: Margins of the figures in pixels
MARGIN: dict[str, int] = {"l": 70, "r": 20, "t": 50, "b": 55}

#: Constant ``uirevision`` of the figures, which makes Plotly keep the zoom of the user when the data is updated
UIREVISION: str = "pttools"


def bubble_fig(bubble: BaseBubble, full_range: bool = False) -> go.Figure:
    r"""Plot the velocity profile $v(\xi)$ and the enthalpy profile $w(\xi)$ of a bubble.

    :param bubble: the bubble, which has to have a fluid profile
    :param full_range: whether to show the full range of $\xi$ instead of the fluid shell
    :return: the figure
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=bubble.xi, y=bubble.v, mode="lines", name=V_LABEL), row=1, col=1)
    fig.add_trace(go.Scatter(x=bubble.xi, y=bubble.w, mode="lines", name=W_LABEL), row=2, col=1)
    fig.update_xaxes(range=xi_limits([bubble], full_range=full_range))
    fig.update_xaxes(title_text=XI_LABEL, row=2, col=1)
    fig.update_yaxes(title_text=V_LABEL, row=1, col=1)
    fig.update_yaxes(title_text=W_LABEL, row=2, col=1)
    fig.update_layout(title_text=bubble.label_unicode, showlegend=False, margin=MARGIN, uirevision=UIREVISION)
    return fig


def empty_fig(message: str) -> go.Figure:
    """Create an empty figure with a message, e.g. for the parameter values that have no data.

    :param message: the message
    :return: the figure
    """
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font={"size": 16})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=MARGIN)
    return fig


def spectrum_fig(spectrum: BaseOmgw0, eb: bool = True, gb: bool = True) -> go.Figure:
    r"""Plot the GW power spectrum today $\mathcal{P}_{\text{gw},0}(f) h^2$ together with the LISA noise.

    :param spectrum: the spectrum
    :param eb: whether to include the extragalactic compact binary noise in the noise and the SNR
    :param gb: whether to include the galactic compact binary noise in the noise and the SNR
    :return: the figure
    """
    # Plotly renders only the labels that are a single LaTeX expression, and therefore the Unicode label is used.
    snr: float = spectrum.snr(noise_eb=eb, noise_gb=gb)[0]
    f_noise, noise = noise_curve([spectrum], eb=eb, gb=gb)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spectrum.f(), y=spectrum.omgw0_h2(), mode="lines", name=POW_GW0_H2_LABEL))
    fig.add_trace(go.Scatter(x=f_noise, y=noise, mode="lines", name=noise_label(eb=eb, gb=gb)))
    fig.update_xaxes(title_text=F_LABEL, type="log", exponentformat="power")
    fig.update_yaxes(title_text=POW_GW0_H2_LABEL, type="log", exponentformat="power")
    fig.update_layout(
        title_text=f"{spectrum.label_unicode}, SNR={as_unicode(snr)}",
        legend={"x": 0.01, "y": 0.01, "xanchor": "left", "yanchor": "bottom"},
        margin=MARGIN,
        uirevision=UIREVISION,
    )
    return fig
