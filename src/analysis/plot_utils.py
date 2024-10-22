from typing import Any

import matplotlib.colors as mcolors
import numpy as np
import plotly.graph_objs as go
import seaborn as sns

PHASE_MAP = {
    0: "clear",
    1: "liquid",
    2: "ice",
    3: "mixed",
    4: "drizzle",
    5: "liq_driz",
    6: "rain",
    7: "snow",
    # 8: "unknown",
}
PHASE_TO_NUM_MAP = {
    "clear": 0,
    "liquid": 1,
    "ice": 2,
    "mixed": 3,
    "drizzle": 4,
    "liq_driz": 5,
    "rain": 6,
    "snow": 7,
    # "unknown": 8,
}
colormap = {
    "liquid": "green",
    "ice": "blue",
    "mixed": "red",
    "drizzle": "lightseagreen",
    "liq_driz": "pink",
    "rain": "gold",
    "snow": "black",
    "unknown": "dimgray",
}

_cblind = sns.color_palette("colorblind")
cblind_cmap = {
    "clear": "white",
    "liquid": _cblind[2],  # green
    "ice": _cblind[0],  # darker blue
    "mixed": _cblind[3],  # burnt orange
    "drizzle": _cblind[9],  # lighter blue
    "liq_driz": _cblind[6],  # light pink
    "rain": _cblind[8],  # yellow
    "snow": _cblind[5],  # brown
    "unknown": "black",
}


def fCMAP(data) -> sns.color_palette:
    """Data should be array-like with category values (liquid, ice, etc.)"""
    data = np.array(data)
    _, unique_indices = np.unique(data, return_index=True)
    unique_indices.sort()
    uniq = data[unique_indices]
    return sns.color_palette([cblind_cmap[v] for v in uniq])


def rgb_to_hex(rgb):
    """Function to convert the sns colormap/values to hex for plotly"""
    return mcolors.to_hex(rgb)


PX_CMAP = {p: rgb_to_hex(c) for p, c in cblind_cmap.items()}

CMAP = fCMAP(list(cblind_cmap))


def create_figure(
    size: tuple[int, int] = (1200, 800),
    legend_loc: tuple[float, float] = (0.075, 1.0),
    xaxis: dict[str, Any] | None = None,
    xaxis2: dict[str, Any] | None = None,
    yaxis: dict[str, Any] | None = None,
    yaxis2: dict[str, Any] | None = None,
    legend: dict[str, Any] | None = None,
    **kwargs,
):
    """Create a plotly.graph_objects.Figure with some preferred settings.

    Args:
        size (tuple[int, int], optional): Width, height tuple. Defaults to (1200, 800).
        legend_loc (tuple[float, float], optional): Legend location. Defaults to (0.075, 1.0).
        xaxis (dict[str, Any] | None, optional): Optional extra keyword arguments for go.Layout(xaxis=...). Defaults to None.
        xaxis2 (dict[str, Any] | None, optional): Optional extra keyword arguments for go.Layout(xaxis2=...). Defaults to None.
        yaxis (dict[str, Any] | None, optional): Optional extra keyword arguments for go.Layout(yaxis=...). Defaults to None.
        yaxis2 (dict[str, Any] | None, optional): Optional extra keyword arguments for go.Layout(yaxis2=...). Defaults to None.
        legend (dict[str, Any] | None, optional): Optional extra keyword arguments for go.Layout(legend=...). Defaults to None.

    Returns:
        plotly.graph_objects.Figure: The plotly figure with specified + preferred layout settings.
    """
    xaxis = xaxis or {}
    xaxis2 = xaxis2 or {}
    yaxis = yaxis or {}
    yaxis2 = yaxis2 or {}
    legend = legend or {}
    layout_settings = dict(
        width=size[0],
        height=size[1],
        barmode="stack",
        bargap=0,
        bargroupgap=0,
        font=dict(family="serif", size=36, color="black"),
        xaxis={
            **dict(
                showline=True,
                linewidth=2,
                linecolor="black",
                ticks="outside",
                tickwidth=2,
                tickfont_size=22,
                mirror=True,
            ),
            **xaxis,
        },
        xaxis2={
            **dict(
                showline=True,
                linewidth=2,
                linecolor="black",
                ticks="outside",
                tickwidth=2,
                tickfont_size=22,
            ),
            **xaxis2,
        },
        yaxis={
            **dict(
                side="left",
                showgrid=False,
                showline=True,
                linewidth=2,
                linecolor="black",
                ticks="outside",
                tickwidth=2,
                tickfont_size=22,
                mirror=True,
            ),
            **yaxis,
        },
        yaxis2={
            **dict(
                overlaying="y",
                side="right",
                showgrid=False,
                showline=True,
                linewidth=2,
                linecolor="black",
                ticks="outside",
                tickfont_size=22,
                tickwidth=2,
            ),
            **yaxis2,
        },
        plot_bgcolor="white",
        legend={
            **dict(
                x=legend_loc[0],
                y=legend_loc[1],
                bgcolor="rgba(0,0,0,0.1)",
                orientation="h",
                traceorder="normal",
                font_size=28,
            ),
            **legend,
        },
        margin=dict(t=0, b=0, l=0, r=10),
    )
    layout_settings.update(**kwargs)
    fig = go.Figure(layout=go.Layout(**layout_settings))
    return fig
