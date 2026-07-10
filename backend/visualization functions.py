"""
visualization_functions.py
--------------------------
General-purpose Plotly visualization building blocks for the Olist e-commerce dataset.

Each function accepts a DataFrame (or dict/Series) returned by a function in
computational_functions.py plus explicit column-name parameters.
Every function returns a plotly.graph_objects.Figure — nothing is rendered here;
call fig.show() or pass it to Dash / Streamlit yourself.

Typical usage pattern
---------------------
result_df = compute_revenue_share(df, group_col="seller_id", revenue_col="price")
fig       = plot_bar(result_df, x="seller_id", y="revenue", title="Revenue by Seller", top_n=20)
fig.show()
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ──────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS (internal)
# ──────────────────────────────────────────────────────────────────────────────

_PALETTE = px.colors.qualitative.Plotly   # default colour sequence

def _apply_top_n(df, value_col, top_n, ascending=False):
    """Return the top/bottom N rows sorted by value_col."""
    if top_n is None:
        return df
    return df.nlargest(top_n, value_col) if not ascending else df.nsmallest(top_n, value_col)


# ──────────────────────────────────────────────────────────────────────────────
# 1. KPI CARDS
# ──────────────────────────────────────────────────────────────────────────────

def plot_kpi_cards(kpis, value_format=None):
    """
    Render a row of KPI indicator cards from a dict of scalar values.

    Feed with the output of compute_kpis() or any flat dict of numbers.

    Parameters
    ----------
    kpis         : dict — {label: value}.  Labels become card titles.
    value_format : dict | None — {label: format_string} e.g. {"total_revenue": "R$ {:,.2f}"}.
                   Labels not in this dict are auto-formatted (int or 2 dp float).

    Returns
    -------
    go.Figure — one Indicator trace per KPI, laid out in a single row.
    """
    labels = list(kpis.keys())
    values = list(kpis.values())
    n = len(labels)

    fig = make_subplots(rows=1, cols=n, specs=[[{"type": "indicator"}] * n])

    for i, (label, value) in enumerate(zip(labels, values)):
        if value_format and label in value_format:
            fmt_value = value_format[label].format(value)
        elif isinstance(value, float):
            fmt_value = f"{value:,.2f}"
        else:
            fmt_value = f"{value:,}"

        fig.add_trace(
            go.Indicator(
                mode="number",
                value=value,
                number={"valueformat": ",.2f" if isinstance(value, float) else ",d"},
                title={"text": label.replace("_", " ").title()},
            ),
            row=1, col=i + 1,
        )

    fig.update_layout(
        height=200,
        margin=dict(t=40, b=10, l=10, r=10),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 2. LINE CHART  (time series / trend)
# ──────────────────────────────────────────────────────────────────────────────

def plot_line(df, x, y_cols, title, y_labels=None, x_label=None, y_label=None,
              markers=False, height=420):
    """
    Plot one or more lines on a shared axis.

    Designed for: monthly revenue trend, growth rate, rolling averages,
    monthly delivery trend, monthly rating trend.

    Parameters
    ----------
    df       : DataFrame — must contain x and all y_cols.
    x        : str — x-axis column (period, date, month label …).
    y_cols   : str | list[str] — column(s) to plot as lines.
    title    : str
    y_labels : list[str] | None — legend names matching y_cols order.
    x_label  : str | None — x-axis title.
    y_label  : str | None — y-axis title.
    markers  : bool — show markers on line points (default False).
    height   : int

    Returns
    -------
    go.Figure
    """
    if isinstance(y_cols, str):
        y_cols = [y_cols]

    y_labels = y_labels or y_cols
    x_vals   = df[x].astype(str)   # periods → str for Plotly

    fig = go.Figure()
    for col, label, colour in zip(y_cols, y_labels, _PALETTE):
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=df[col],
            mode="lines+markers" if markers else "lines",
            name=label,
            line=dict(color=colour),
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label or x,
        yaxis_title=y_label or "",
        height=height,
        hovermode="x unified",
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 3. BAR CHART
# ──────────────────────────────────────────────────────────────────────────────

def plot_bar(df, x, y, title, orientation="v", color=None, top_n=None,
             x_label=None, y_label=None, height=450, text_col=None):
    """
    Generic bar chart — vertical or horizontal.

    Designed for: revenue by seller/category/state, top products, payment types,
    delivery time by state, rating by category, etc.

    Parameters
    ----------
    df          : DataFrame
    x           : str — column for x-axis (or y-axis labels when orientation='h').
    y           : str — column for bar height (or bar length when orientation='h').
    title       : str
    orientation : 'v' | 'h'
    color       : str | None — column used to colour bars (categorical).
    top_n       : int | None — keep only top N rows by y value before plotting.
    x_label     : str | None
    y_label     : str | None
    height      : int
    text_col    : str | None — column whose values to display on top of bars
                  (e.g. "revenue_share_pct" for labelling share %).

    Returns
    -------
    go.Figure
    """
    plot_df = _apply_top_n(df, y, top_n) if top_n else df.copy()

    if orientation == "h":
        plot_df = plot_df.sort_values(y, ascending=True)   # largest at top
        fig = px.bar(plot_df, x=y, y=x, orientation="h",
                     color=color, text=text_col, title=title,
                     labels={x: x_label or x, y: y_label or y},
                     height=height, color_discrete_sequence=_PALETTE)
    else:
        fig = px.bar(plot_df, x=x, y=y, orientation="v",
                     color=color, text=text_col, title=title,
                     labels={x: x_label or x, y: y_label or y},
                     height=height, color_discrete_sequence=_PALETTE)

    if text_col:
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")

    fig.update_layout(showlegend=color is not None)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 4. PARETO CHART  (bar + cumulative line)
# ──────────────────────────────────────────────────────────────────────────────

def plot_pareto(df, x, bar_col, cumulative_col, title,
                x_label=None, bar_label="Revenue", line_label="Cumulative %",
                height=450):
    """
    Dual-axis Pareto chart: bars for individual values, line for cumulative %.

    Feed with the output of compute_pareto().

    Parameters
    ----------
    df             : DataFrame — already sorted descending by bar_col.
    x              : str — entity column (category, product, seller …).
    bar_col        : str — revenue or count column for bars.
    cumulative_col : str — cumulative share % column for the line.
    title          : str
    x_label        : str | None
    bar_label      : str — legend name for bars.
    line_label     : str — legend name for cumulative line.
    height         : int

    Returns
    -------
    go.Figure
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=df[x].astype(str), y=df[bar_col], name=bar_label,
               marker_color=_PALETTE[0]),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df[x].astype(str), y=df[cumulative_col],
                   name=line_label, mode="lines+markers",
                   line=dict(color=_PALETTE[1])),
        secondary_y=True,
    )

    # 80 % reference line
    fig.add_hline(y=80, line_dash="dash", line_color="red",
                  annotation_text="80%", secondary_y=True)

    fig.update_layout(title=title, height=height,
                      xaxis_title=x_label or x, hovermode="x unified")
    fig.update_yaxes(title_text=bar_label, secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", range=[0, 105], secondary_y=True)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 5. SCATTER PLOT
# ──────────────────────────────────────────────────────────────────────────────

def plot_scatter(df, x, y, title, color=None, size=None, label_col=None,
                 trendline=False, x_label=None, y_label=None, height=450):
    """
    Scatter / bubble chart — correlations and outlier detection.

    Designed for: delivery time vs rating, basket size vs revenue,
    freight ratio vs revenue, AOV distribution, instalment vs spend.

    Parameters
    ----------
    df         : DataFrame
    x          : str — x-axis numeric column.
    y          : str — y-axis numeric column.
    title      : str
    color      : str | None — categorical column for colour encoding.
    size       : str | None — numeric column for bubble size.
    label_col  : str | None — column whose values appear as hover labels.
    trendline  : bool — add an OLS trendline (requires statsmodels).
    x_label    : str | None
    y_label    : str | None
    height     : int

    Returns
    -------
    go.Figure
    """
    trend = "ols" if trendline else None

    fig = px.scatter(
        df, x=x, y=y, color=color, size=size,
        hover_name=label_col, trendline=trend,
        title=title,
        labels={x: x_label or x, y: y_label or y},
        height=height,
        color_discrete_sequence=_PALETTE,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 6. DISTRIBUTION  (histogram + optional box)
# ──────────────────────────────────────────────────────────────────────────────

def plot_distribution(series, title, bins=40, show_box=True,
                      x_label=None, height=420):
    """
    Histogram of a numeric Series with an optional marginal box plot.

    Feed with the entity_series returned by compute_distribution_stats()
    or compute_basket_size().

    Parameters
    ----------
    series   : pd.Series — numeric values to plot.
    title    : str
    bins     : int — number of histogram bins.
    show_box : bool — show marginal box plot above the histogram.
    x_label  : str | None
    height   : int

    Returns
    -------
    go.Figure
    """
    marginal = "box" if show_box else None
    fig = px.histogram(
        series, nbins=bins, marginal=marginal,
        title=title,
        labels={"value": x_label or series.name or "Value"},
        height=height,
        color_discrete_sequence=[_PALETTE[0]],
    )
    fig.update_layout(showlegend=False)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 7. COMBO CHART  (bar + line on dual axis)
# ──────────────────────────────────────────────────────────────────────────────

def plot_combo(df, x, bar_col, line_col, title, bar_label="", line_label="",
               x_label=None, bar_y_label=None, line_y_label=None, height=420):
    """
    Dual-axis chart: grouped bars on primary axis, line on secondary axis.

    Designed for: revenue (bar) + growth rate (line), orders (bar) + AOV (line),
    review count (bar) + mean rating (line).

    Parameters
    ----------
    df            : DataFrame
    x             : str — shared x-axis column.
    bar_col       : str — column for bar heights (primary y-axis).
    line_col      : str — column for line values (secondary y-axis).
    title         : str
    bar_label     : str — legend name for bars.
    line_label    : str — legend name for line.
    x_label       : str | None
    bar_y_label   : str | None — primary y-axis title.
    line_y_label  : str | None — secondary y-axis title.
    height        : int

    Returns
    -------
    go.Figure
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=df[x].astype(str), y=df[bar_col], name=bar_label or bar_col,
               marker_color=_PALETTE[0]),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df[x].astype(str), y=df[line_col],
                   name=line_label or line_col, mode="lines+markers",
                   line=dict(color=_PALETTE[1])),
        secondary_y=True,
    )

    fig.update_layout(title=title, xaxis_title=x_label or x,
                      height=height, hovermode="x unified")
    fig.update_yaxes(title_text=bar_y_label or bar_col, secondary_y=False)
    fig.update_yaxes(title_text=line_y_label or line_col, secondary_y=True)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 8. TREEMAP
# ──────────────────────────────────────────────────────────────────────────────

def plot_treemap(df, path_cols, value_col, title, color_col=None, height=500):
    """
    Hierarchical treemap — good for nested category / product / region breakdowns.

    Designed for: category revenue share, product revenue, geographic revenue.

    Parameters
    ----------
    df        : DataFrame
    path_cols : list[str] — hierarchy levels from outermost to innermost
                e.g. ["state", "city"] or ["category", "product"].
    value_col : str — numeric column that determines rectangle size.
    title     : str
    color_col : str | None — numeric column to colour rectangles (e.g. avg_rating).
    height    : int

    Returns
    -------
    go.Figure
    """
    fig = px.treemap(
        df,
        path=[px.Constant("All")] + path_cols,
        values=value_col,
        color=color_col,
        title=title,
        height=height,
        color_continuous_scale="RdYlGn" if color_col else None,
    )
    fig.update_traces(textinfo="label+percent entry")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 9. HEATMAP
# ──────────────────────────────────────────────────────────────────────────────

def plot_heatmap(df, x, y, z, title, x_label=None, y_label=None,
                 colorscale="RdYlGn", height=450):
    """
    Pivot-based heatmap for cross-dimensional analysis.

    Designed for: rating by category × month, delivery time by state × month,
    freight ratio by region × category.

    Parameters
    ----------
    df         : DataFrame — long format (one row per x/y combination).
    x          : str — column for x-axis categories.
    y          : str — column for y-axis categories.
    z          : str — numeric column for cell colour/intensity.
    title      : str
    x_label    : str | None
    y_label    : str | None
    colorscale : str — Plotly colorscale name (default "RdYlGn").
    height     : int

    Returns
    -------
    go.Figure
    """
    pivot = df.pivot_table(index=y, columns=x, values=z, aggfunc="mean")

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.astype(str).tolist(),
        y=pivot.index.astype(str).tolist(),
        colorscale=colorscale,
        hoverongaps=False,
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label or x,
        yaxis_title=y_label or y,
        height=height,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 10. BUBBLE / STRATEGIC QUADRANT
# ──────────────────────────────────────────────────────────────────────────────

def plot_quadrant(df, x, y, size, label_col, title,
                  x_label=None, y_label=None, height=520):
    """
    Bubble chart with quadrant reference lines — strategic positioning view.

    Designed for: composite score questions (Q34, Q35, Q36) where you want to
    see entities across two key dimensions with bubble size as a third signal.

    Reference lines are drawn at the median of x and y, creating four quadrants.

    Parameters
    ----------
    df        : DataFrame
    x         : str — first strategic metric column (e.g. revenue).
    y         : str — second strategic metric column (e.g. mean_rating).
    size      : str — third metric for bubble size (e.g. order_count).
    label_col : str — entity label column shown on hover.
    title     : str
    x_label   : str | None
    y_label   : str | None
    height    : int

    Returns
    -------
    go.Figure
    """
    fig = px.scatter(
        df, x=x, y=y, size=size, hover_name=label_col,
        title=title,
        labels={x: x_label or x, y: y_label or y},
        height=height,
        color_discrete_sequence=[_PALETTE[0]],
    )

    # Median reference lines
    fig.add_vline(x=df[x].median(), line_dash="dash", line_color="grey",
                  annotation_text=f"Median {x_label or x}", annotation_position="top left")
    fig.add_hline(y=df[y].median(), line_dash="dash", line_color="grey",
                  annotation_text=f"Median {y_label or y}", annotation_position="bottom right")

    return fig
