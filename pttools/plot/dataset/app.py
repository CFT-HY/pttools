r"""Interactive viewer for exploring a dataset of bubbles and spectra in a web browser.

The viewer shows a slider for each parameter of the dataset, with a step for each of its stored values,
and the stored fluid profile of the bubble, the spectrum and the scalar values of the selected parameter values.
The data is read from the file with :py:class:`pttools.export.importer.Importer` one bubble and spectrum at a time,
and the rows are found with :py:class:`pttools.export.index.DatasetIndex`.
Therefore, the viewer can be used for files that are much larger than the memory.

.. code-block:: python

    from pttools.plot.dataset.app import run

    run("dataset.h5")

The viewer needs `Dash <https://dash.plotly.com/>`_, which is installed with ``pip install "pttools-gw[dashboard]"``.
By default, the server listens only to the connections from the same computer.
To use a viewer that is running on a remote computer, forward the port with SSH,
e.g. ``ssh -L 8050:localhost:8050 user@host``, and open ``http://localhost:8050`` in a local browser.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import logging
import math
import os
import threading
import typing as tp

import numpy as np

try:
    from dash import Dash, Input, Output, dcc, html
except ImportError as err:
    raise ImportError(
        "The dataset viewer needs Dash, which is an optional dependency of PTtools. "
        "Install it with: pip install \"pttools-gw[dashboard]\""
    ) from err

from pttools.export.importer import TABLE_TITLES, Importer
from pttools.export.index import Axis, DatasetIndex, Selection, Value
from pttools.export.records import Table
from pttools.math.formatting import compact
from pttools.omgw0 import BaseOmgw0
from pttools.plot.dataset.figures import bubble_fig, empty_fig, spectrum_fig
import pttools.type_hints as th
from pttools.utils.fields import FieldShape
from pttools.utils.files import format_size

if th.TYPE_CHECKING_OR_DOCS:
    import plotly.graph_objects as go

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DatasetViewer",
    "ScalarRow",
    "ViewState",
    "create_app",
    "run",
]

logger: logging.Logger = logging.getLogger(__name__)

#: Default address of the server, which accepts connections only from the same computer
DEFAULT_HOST: str = "127.0.0.1"

#: Default port of the server
DEFAULT_PORT: int = 8050

#: Number of imported bubbles after which the cache of the importer is cleared
MAX_CACHED_BUBBLES: int = 100

#: Maximum number of labelled values on a slider
MAX_MARKS: int = 11

#: CSS class of the tables of the scalar values, which are styled in ``assets/viewer.css``
SCALARS_CLASS: str = "pttools-scalars"

#: Titles of the tables of the scalar values
SCALAR_TITLES: dict[Table, str] = {
    Table.MODELS: "Model",
    Table.BUBBLES: "Bubble",
    Table.SPECTRA_Y: "Spectrum",
    Table.SPECTRA_F: "Spectrum",
}

_STYLE_PAGE: dict[str, tp.Any] = {"fontFamily": "system-ui, sans-serif", "fontSize": "14px", "padding": "12px 16px"}
_STYLE_BODY: dict[str, tp.Any] = {"display": "flex", "flexWrap": "wrap", "gap": "24px", "alignItems": "flex-start"}
_STYLE_SIDEBAR: dict[str, tp.Any] = {"flex": "0 0 340px", "maxWidth": "100%"}
_STYLE_MAIN: dict[str, tp.Any] = {"flex": "1 1 600px", "minWidth": "0"}
_STYLE_ROW: dict[str, tp.Any] = {"display": "flex", "flexWrap": "wrap", "gap": "16px"}
_STYLE_GRAPH: dict[str, tp.Any] = {"flex": "1 1 480px", "minWidth": "0", "height": "520px"}
_STYLE_TABLE: dict[str, tp.Any] = {"minWidth": "0", "overflowX": "auto"}
# The tables are below the controls and the figures in three columns of equal width.
_STYLE_TABLES: dict[str, tp.Any] = {
    "display": "grid", "gridTemplateColumns": "repeat(3, minmax(0, 1fr))", "gap": "16px", "marginTop": "8px"}
_STYLE_CONTROL: dict[str, tp.Any] = {"marginBottom": "18px"}


@dataclasses.dataclass(frozen=True, slots=True)
class ScalarRow:
    """A stored scalar value of a model, a bubble or a spectrum.

    :param name: name of the field
    :param value: the value
    :param description: description of the field
    """

    name: str
    value: tp.Any
    description: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class ViewState:
    """The contents of the viewer for a combination of parameter values.

    :param selection: the rows of the model, the bubble and the spectrum
    :param bubble_fig: the figure of the fluid profile of the bubble
    :param spectrum_fig: the figure of the spectrum
    :param messages: messages about the selection, e.g. the errors of the solution of the bubble
    :param scalars: the stored scalar values of the model, the bubble and the spectrum by their tables
    """

    selection: Selection
    bubble_fig: go.Figure
    spectrum_fig: go.Figure
    messages: tuple[str, ...] = ()
    scalars: Mapping[Table, tuple[ScalarRow, ...]] = dataclasses.field(default_factory=dict)


class DatasetViewer:
    """Viewer for exploring a dataset by the values of its parameters.

    The contents for a combination of parameter values are given by :py:meth:`update`,
    which does not depend on Dash and can therefore be used also for other user interfaces.

    :param importer: the importer of the file, which has to stay open while the viewer is used
    :param index: the index of the parameters. If None, it's created with the default parameters.
    """

    def __init__(self, importer: Importer, index: DatasetIndex | None = None) -> None:
        """Create the index if needed, and read the descriptions of the fields."""
        self.importer: Importer = importer
        self.index: DatasetIndex = DatasetIndex(importer) if index is None else index
        #: The descriptions of the scalar fields of each table by their names
        self.descriptions: dict[Table, dict[str, str]] = {Table.MODELS: self.index.model_descriptions()}
        for table in (Table.BUBBLES, self.index.table):
            infos = {name: importer.field_info(table, name) for name in importer.fields(table)}
            self.descriptions[table] = {
                name: info.get("description", "") for name, info in infos.items()
                if info.get("kind") == FieldShape.SCALAR
            }
        #: The parameters that have several values, and therefore a control
        self.controls: tuple[Axis, ...] = tuple(axis for axis in self.index.axes if len(axis.values) > 1)
        #: The values of the parameters that have a single value
        self.fixed: dict[str, Value] = {
            axis.name: axis.values[0] for axis in self.index.axes if len(axis.values) == 1}
        # The Flask server handles the requests in threads, but the importer and its caches are not thread-safe.
        self._lock: threading.Lock = threading.Lock()
        self._cached_bubbles: set[int] = set()

    def control_values(self, states: tp.Sequence[tp.Any]) -> dict[str, Value]:
        """Convert the states of the controls to the values of the parameters.

        :param states: the states of the :py:attr:`controls`: the indices of the values for the sliders
            of the numeric parameters, and the values themselves for the other parameters
        :return: the values of all the parameters by their names
        """
        values = dict(self.fixed)
        for axis, state in zip(self.controls, states, strict=True):
            values[axis.name] = axis.values[int(state)] if axis.numeric else state
        return values

    def layout(self) -> html.Div:
        """Create the layout of the application with the controls set to the default values of the index."""
        info = self.importer.info()
        contents = ", ".join(
            f"{TABLE_TITLES.get(name, name).lower()}: {n_rows}"
            for name, n_rows in self.importer.contents().items() if n_rows)
        defaults = self.index.default_values()
        return html.Div(style=_STYLE_PAGE, children=[
            html.H2("PTtools dataset viewer", style={"margin": "0 0 4px"}),
            html.Div(
                f"{info['path']} ({format_size(info['size'])}; {contents})",
                style={"color": "#555", "marginBottom": "16px", "overflowWrap": "anywhere"}),
            html.Div(style=_STYLE_BODY, children=[
                html.Div(style=_STYLE_SIDEBAR, children=[
                    *(self._control(axis, defaults[axis.name]) for axis in self.controls),
                    *(dcc.Markdown(_axis_label(axis, self.fixed[axis.name]), mathjax=True)
                      for axis in self.index.axes if axis.name in self.fixed),
                ]),
                html.Div(style=_STYLE_MAIN, children=[
                    dcc.Markdown(id="messages", mathjax=True),
                    html.Div(style=_STYLE_ROW, children=[
                        dcc.Graph(id="bubble-graph", mathjax=True, style=_STYLE_GRAPH),
                        dcc.Graph(id="spectrum-graph", mathjax=True, style=_STYLE_GRAPH),
                    ]),
                ]),
            ]),
            html.Div(style=_STYLE_TABLES, children=[
                html.Div(style=_STYLE_TABLE, children=[
                    html.H3(SCALAR_TITLES[table]),
                    dcc.Markdown(id=f"scalars-{table}", mathjax=True, className=SCALARS_CLASS),
                ])
                for table in self.descriptions
            ]),
        ])

    def register(self, app: Dash) -> None:
        """Set the layout of the application and register the callback that updates it.

        :param app: the Dash application
        """
        app.layout = self.layout()
        tables = tuple(self.descriptions)

        @app.callback(
            Output("messages", "children"),
            Output("bubble-graph", "figure"),
            Output("spectrum-graph", "figure"),
            *(Output(f"scalars-{table}", "children") for table in tables),
            *(Output(f"label-{axis.name}", "children") for axis in self.controls),
            *(Input(f"control-{axis.name}", "value") for axis in self.controls),
        )
        def _update(*states: tp.Any) -> tuple[tp.Any, ...]:
            values = self.control_values(states)
            state = self.update(values)
            return (
                "\n\n".join(state.messages),
                state.bubble_fig,
                state.spectrum_fig,
                *(_scalars_markdown(state.scalars.get(table, ())) for table in tables),
                *(_axis_label(axis, values[axis.name]) for axis in self.controls),
            )

    def update(self, values: Mapping[str, Value]) -> ViewState:
        """Read the data of a combination of parameter values and create its figures.

        :param values: the values of all the parameters of the index by their names
        :return: the contents of the viewer
        """
        with self._lock:
            return self._update(values)

    def _control(self, axis: Axis, value: Value) -> html.Div:
        """Create the control of a parameter: a slider for the numeric parameters, and radio buttons for the others.

        The slider selects the index of the value, so that its steps are the stored values
        with even spacing regardless of whether the values are on a linear or a logarithmic axis.
        """
        control_id = f"control-{axis.name}"
        if axis.numeric:
            n_values = len(axis.values)
            step = math.ceil((n_values - 1) / (MAX_MARKS - 1))
            marked = sorted({*range(0, n_values, step), n_values - 1})
            control: tp.Any = dcc.Slider(
                id=control_id,
                min=0,
                max=n_values - 1,
                step=1,
                value=axis.values.index(value),
                marks={i: f"{axis.values[i]:.3g}" for i in marked},
                included=False,
                updatemode="drag",
                allow_direct_input=False,
            )
        else:
            control = dcc.RadioItems(
                id=control_id, options=[str(option) for option in axis.values], value=value, inline=True,
                labelStyle={"marginRight": "16px"})
        return html.Div(style=_STYLE_CONTROL, children=[
            dcc.Markdown(_axis_label(axis, value), id=f"label-{axis.name}", mathjax=True),
            control,
        ])

    def _scalars(self, table: Table, values: Mapping[str, tp.Any]) -> tuple[ScalarRow, ...]:
        """Combine the scalar values of a row with the descriptions of their fields."""
        descriptions = self.descriptions[table]
        return tuple(ScalarRow(name, value, descriptions.get(name, "")) for name, value in values.items())

    def _update(self, values: Mapping[str, Value]) -> ViewState:
        """Read the data of a combination of parameter values without locking, see :py:meth:`update`."""
        importer = self.importer
        table = self.index.table
        selection = self.index.lookup(values)
        if selection.model is None:
            message = "No model is stored for these parameter values."
            return ViewState(selection, empty_fig(message), empty_fig(message), (f"**{message}**",))
        scalars: dict[Table, tuple[ScalarRow, ...]] = {
            Table.MODELS: self._scalars(Table.MODELS, {
                "class": importer.read(Table.MODELS, "class", selection.model),
                **importer.model_params(selection.model),
            }),
        }
        if selection.bubble is None:
            message = "No bubble is stored for these parameter values."
            return ViewState(selection, empty_fig(message), empty_fig(message), (f"**{message}**",), scalars)

        # Each imported bubble holds its fluid profiles, so the cache would grow without a limit.
        if len(self._cached_bubbles) >= MAX_CACHED_BUBBLES:
            importer.clear_cache()
            self._cached_bubbles.clear()
        self._cached_bubbles.add(selection.bubble)

        bubble_scalars = importer.read_scalars(Table.BUBBLES, self.descriptions[Table.BUBBLES], selection.bubble)
        scalars[Table.BUBBLES] = self._scalars(Table.BUBBLES, bubble_scalars)
        messages = [f"Model {selection.model}, bubble {selection.bubble}"]
        if bubble_scalars.get("failed", False):
            errors = [name for name, value in bubble_scalars.items()
                      if isinstance(value, (bool, np.bool_)) and value and name != "failed"]
            messages.append(f"**The solution of the bubble has errors:** {', '.join(errors)}")

        if selection.spectrum is None:
            bubble = importer.import_bubble(selection.bubble)
            message = "No spectrum is stored for these parameter values."
            messages.append(f"**{message}**")
            fig_spectrum = empty_fig(message)
        else:
            spectrum = importer.import_spectrum(selection.spectrum, table=table)
            bubble = spectrum.bubble
            messages[0] += f", spectrum {selection.spectrum} of the table `{table}`"
            scalars[table] = self._scalars(
                table, importer.read_scalars(table, self.descriptions[table], selection.spectrum))
            fig_spectrum = spectrum_fig(spectrum) if isinstance(spectrum, BaseOmgw0) \
                else empty_fig("The spectrum has no GW power spectrum today.")
        fig_bubble = bubble_fig(bubble) if bubble.solved else empty_fig("The bubble has no fluid profile.")
        messages += [f"Note: {note}" for note in bubble.notes]
        return ViewState(selection, fig_bubble, fig_spectrum, tuple(messages), scalars)


def _axis_label(axis: Axis, value: Value) -> str:
    """The Markdown label of a parameter with its value."""
    return f"{axis.description or axis.name}: **{_format(value)}**"


def create_app(importer: Importer, **kwargs: tp.Any) -> Dash:
    """Create the Dash application of the viewer.

    :param importer: the importer of the file, which has to stay open while the application is used
    :param kwargs: arguments for :py:class:`pttools.export.index.DatasetIndex`, e.g. ``model_params``
    :return: the application, which can be started with its ``run()`` method
    """
    viewer = DatasetViewer(importer, DatasetIndex(importer, **kwargs))
    # The empty update title keeps the title of the browser tab from flickering when the sliders are moved.
    app = Dash(__name__, title=f"{importer.path.name} - PTtools dataset viewer", update_title="")
    viewer.register(app)
    return app


def _escape(text: str) -> str:
    """Escape the column separators of Markdown tables."""
    return text.replace("|", r"\|").replace("\n", " ")


def _format(value: tp.Any) -> str:
    """Format a value for the user interface."""
    if value is None:
        return ""
    if isinstance(value, (str, np.str_)):
        return str(value)
    return compact(value)


def run(
        path: str | os.PathLike[str],
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        debug: bool = False,
        verify: bool = False,
        **kwargs: tp.Any) -> None:
    """Open a dataset and serve the viewer until interrupted.

    :param path: path of the HDF5 file
    :param host: address to listen to. The default accepts connections only from the same computer.
        The viewer has no access control, so listen to other addresses only in trusted networks.
    :param port: port to listen to
    :param debug: whether to enable the debugging tools of Dash
    :param verify: whether to verify the file against its SHA-256 checksum file
    :param kwargs: arguments for :py:class:`pttools.export.index.DatasetIndex`, e.g. ``model_params``
    """
    with Importer(path, verify=verify) as importer:
        app = create_app(importer, **kwargs)
        logger.info("Starting the dataset viewer of %s at http://%s:%d", importer.path, host, port)
        # The reloader would start a second process, which would create the index again.
        app.run(host=host, port=port, debug=debug, use_reloader=False)


def _scalars_markdown(rows: tp.Sequence[ScalarRow]) -> str:
    """Format scalar values as a Markdown table, or as an empty string if there are no values."""
    if not rows:
        return ""
    return "\n".join([
        "| Name | Value | Description |",
        "|---|---|---|",
        *(f"| `{row.name}` | {_escape(_format(row.value))} | {_escape(row.description)} |" for row in rows),
    ])
