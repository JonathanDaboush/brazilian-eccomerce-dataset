"""
controller_functions.py
-----------------------
Single dispatch controller for the analytics workflow.

This module exposes one entry point, run_task(), that accepts a task name
string and routes the request to the correct computation and visualization
functions.

The controller does not compute metrics itself and does not create figures
itself. It only orchestrates prepared data between the computation layer and
visualization layer.

Design rules
------------
- Accept a task name string, not question numbers.
- Use the task name to select the appropriate compute function(s).
- Use the computed output as input to the correct plot function.
- Do not calculate metrics inside the controller.
- Do not group, aggregate, or transform data here unless the computation
  function already returns the prepared output.
- Return Plotly figures or a small dict of figures when a task needs multiple
  visuals.
"""

from typing import Any, Dict
import pandas as pd
from data_analysis_functions.computational_functions import *
from data_analysis_functions.visualization_functions import *


TASK_DISPATCH: Dict[str, str] = {
    "kpi_cards": "kpi_cards",
    "monthly_revenue_trend": "monthly_revenue_trend",
    "revenue_growth_trend": "revenue_growth_trend",
    "revenue_rolling_average": "revenue_rolling_average",
    "monthly_revenue_rank": "monthly_revenue_rank",
    "concentration_summary": "concentration_summary",
    "pareto_analysis": "pareto_analysis",
    "revenue_share": "revenue_share",
    "distribution_summary": "distribution_summary",
    "geographic_summary": "geographic_summary",
    "product_performance": "product_performance",
    "seller_performance": "seller_performance",
    "freight_ratio_analysis": "freight_ratio_analysis",
    "delivery_metrics": "delivery_metrics",
    "correlation_analysis": "correlation_analysis",
    "rating_summary": "rating_summary",
    "payment_summary": "payment_summary",
    "basket_size_analysis": "basket_size_analysis",
    "composite_score": "composite_score",
    "market_opportunity": "market_opportunity",
}


def run_task(task_name: str, df: pd.DataFrame, **kwargs) -> Any:
    """
    Run a named analytics task.

    Parameters
    ----------
    task_name:
        A descriptive task name such as:
        - "kpi_cards"
        - "monthly_revenue_trend"
        - "pareto_analysis"
        - "delivery_metrics"
        - "composite_score"
        - "market_opportunity"
    df:
        Input dataframe.
    **kwargs:
        Extra parameters forwarded to the computation and visualization layers.

    Returns
    -------
    Plotly figure or a dictionary of Plotly figures depending on the task.
    """
    task_key = task_name.lower().strip()

    if task_key not in TASK_DISPATCH:
        raise ValueError(
            f"Unknown task_name: {task_name}. Available tasks: {list(TASK_DISPATCH.keys())}"
        )

    if task_key == "kpi_cards":
        kpis = compute_kpis(df, **kwargs)
        return plot_kpi_cards(kpis, value_format=kwargs.get("value_format"))

    if task_key == "monthly_revenue_trend":
        monthly_df = compute_monthly_series(df, **kwargs)
        return plot_line(
            monthly_df,
            x=kwargs.get("x", "period"),
            y_cols=kwargs.get("y_cols", ["revenue"]),
            title=kwargs.get("title", "Monthly Revenue Trend"),
            y_labels=kwargs.get("y_labels"),
            x_label=kwargs.get("x_label"),
            y_label=kwargs.get("y_label"),
            markers=kwargs.get("markers", False),
            height=kwargs.get("height", 420),
        )

    if task_key == "revenue_growth_trend":
        growth_df = compute_growth_rate(df, **kwargs)
        rolling_df = compute_rolling_avg(df, **kwargs)
        return plot_combo(
            growth_df,
            x=kwargs.get("x", "period"),
            bar_col=kwargs.get("bar_col", "growth_rate"),
            line_col=kwargs.get("line_col", "rolling_avg"),
            title=kwargs.get("title", "Revenue Growth and Rolling Average"),
            bar_label=kwargs.get("bar_label", "Growth Rate"),
            line_label=kwargs.get("line_label", "Rolling Average"),
            x_label=kwargs.get("x_label"),
            bar_y_label=kwargs.get("bar_y_label"),
            line_y_label=kwargs.get("line_y_label"),
            height=kwargs.get("height", 420),
        )

    if task_key == "revenue_rolling_average":
        rolling_df = compute_rolling_avg(df, **kwargs)
        return plot_line(
            rolling_df,
            x=kwargs.get("x", "period"),
            y_cols=kwargs.get("y_cols", ["rolling_avg"]),
            title=kwargs.get("title", "Rolling Average"),
            y_labels=kwargs.get("y_labels"),
            x_label=kwargs.get("x_label"),
            y_label=kwargs.get("y_label"),
            markers=kwargs.get("markers", False),
            height=kwargs.get("height", 420),
        )

    if task_key == "monthly_revenue_rank":
        rank_df = compute_monthly_revenue_rank(df, **kwargs)
        return plot_bar(
            rank_df,
            x=kwargs.get("x", "category"),
            y=kwargs.get("y", "revenue"),
            title=kwargs.get("title", "Monthly Revenue Rank"),
            orientation=kwargs.get("orientation", "v"),
            color=kwargs.get("color"),
            top_n=kwargs.get("top_n"),
            x_label=kwargs.get("x_label"),
            y_label=kwargs.get("y_label"),
            height=kwargs.get("height", 450),
            text_col=kwargs.get("text_col"),
        )

    if task_key == "concentration_summary":
        stats = compute_concentration_stats(df, **kwargs)
        pareto_df = compute_pareto(df, **kwargs)
        return {
            "summary": plot_kpi_cards(stats, value_format=kwargs.get("value_format")),
            "pareto": plot_pareto(
                pareto_df,
                x=kwargs.get("x", "entity"),
                bar_col=kwargs.get("bar_col", "value"),
                cumulative_col=kwargs.get("cumulative_col", "cumulative_pct"),
                title=kwargs.get("title", "Pareto Analysis"),
                x_label=kwargs.get("x_label"),
                bar_label=kwargs.get("bar_label", "Value"),
                line_label=kwargs.get("line_label", "Cumulative %"),
                height=kwargs.get("height", 450),
            ),
        }

    if task_key == "pareto_analysis":
        pareto_df = compute_pareto(df, **kwargs)
        return plot_pareto(
            pareto_df,
            x=kwargs.get("x", "entity"),
            bar_col=kwargs.get("bar_col", "value"),
            cumulative_col=kwargs.get("cumulative_col", "cumulative_pct"),
            title=kwargs.get("title", "Pareto Analysis"),
            x_label=kwargs.get("x_label"),
            bar_label=kwargs.get("bar_label", "Value"),
            line_label=kwargs.get("line_label", "Cumulative %"),
            height=kwargs.get("height", 450),
        )

    if task_key == "revenue_share":
        share_df = compute_revenue_share(df, **kwargs)
        return plot_bar(
            share_df,
            x=kwargs.get("x", "category"),
            y=kwargs.get("y", "share"),
            title=kwargs.get("title", "Revenue Share"),
            orientation=kwargs.get("orientation", "v"),
            color=kwargs.get("color"),
            top_n=kwargs.get("top_n"),
            x_label=kwargs.get("x_label"),
            y_label=kwargs.get("y_label"),
            height=kwargs.get("height", 450),
            text_col=kwargs.get("text_col"),
        )

    if task_key == "distribution_summary":
        dist_df = compute_distribution_stats(df, **kwargs)
        return plot_distribution(
            dist_df,
            title=kwargs.get("title", "Distribution"),
            bins=kwargs.get("bins", 40),
            show_box=kwargs.get("show_box", True),
            x_label=kwargs.get("x_label"),
            height=kwargs.get("height", 420),
        )

    if task_key == "geographic_summary":
        geo_df = compute_geographic_summary(df, **kwargs)
        return plot_treemap(
            geo_df,
            path_cols=kwargs.get("path_cols", ["region", "state"]),
            value_col=kwargs.get("value_col", "value"),
            title=kwargs.get("title", "Geographic Summary"),
            color_col=kwargs.get("color_col"),
            height=kwargs.get("height", 500),
        )

    if task_key == "product_performance":
        prod_df = compute_product_performance(df, **kwargs)
        return plot_bar(
            prod_df,
            x=kwargs.get("x", "product"),
            y=kwargs.get("y", "value"),
            title=kwargs.get("title", "Product Performance"),
            orientation=kwargs.get("orientation", "v"),
            color=kwargs.get("color"),
            top_n=kwargs.get("top_n"),
            x_label=kwargs.get("x_label"),
            y_label=kwargs.get("y_label"),
            height=kwargs.get("height", 450),
            text_col=kwargs.get("text_col"),
        )

    if task_key == "seller_performance":
        seller_df = aggregate_by(df, **kwargs)
        return plot_scatter(
            seller_df,
            x=kwargs.get("x", "x_metric"),
            y=kwargs.get("y", "y_metric"),
            title=kwargs.get("title", "Seller Performance"),
            color=kwargs.get("color"),
            size=kwargs.get("size"),
            label_col=kwargs.get("label_col"),
            trendline=kwargs.get("trendline", False),
            x_label=kwargs.get("x_label"),
            y_label=kwargs.get("y_label"),
            height=kwargs.get("height", 450),
        )

    if task_key == "freight_ratio_analysis":
        freight_df = compute_freight_ratio(df, **kwargs)
        return plot_bar(
            freight_df,
            x=kwargs.get("x", "group"),
            y=kwargs.get("y", "freight_ratio"),
            title=kwargs.get("title", "Freight Ratio Analysis"),
            orientation=kwargs.get("orientation", "v"),
            color=kwargs.get("color"),
            top_n=kwargs.get("top_n"),
            x_label=kwargs.get("x_label"),
            y_label=kwargs.get("y_label"),
            height=kwargs.get("height", 450),
            text_col=kwargs.get("text_col"),
        )

    if task_key == "delivery_metrics":
        delivery_df = compute_delivery_metrics(df, **kwargs)
        return {
            "line": plot_line(
                delivery_df,
                x=kwargs.get("x", "period"),
                y_cols=kwargs.get("y_cols", ["delivery_days"]),
                title=kwargs.get("line_title", "Delivery Trend"),
                y_labels=kwargs.get("y_labels"),
                x_label=kwargs.get("x_label"),
                y_label=kwargs.get("y_label"),
                markers=kwargs.get("markers", False),
                height=kwargs.get("height", 420),
            ),
            "bar": plot_bar(
                delivery_df,
                x=kwargs.get("bar_x", "group"),
                y=kwargs.get("bar_y", "delivery_days"),
                title=kwargs.get("bar_title", "Delivery Performance"),
                orientation=kwargs.get("orientation", "v"),
                color=kwargs.get("color"),
                top_n=kwargs.get("top_n"),
                x_label=kwargs.get("x_label"),
                y_label=kwargs.get("y_label"),
                height=kwargs.get("bar_height", 450),
                text_col=kwargs.get("text_col"),
            ),
            "kpi": plot_kpi_cards(
                delivery_df.to_dict(orient="records")[0] if len(delivery_df) == 1 else delivery_df.iloc[0].to_dict(),
                value_format=kwargs.get("value_format"),
            ),
        }

    if task_key == "correlation_analysis":
        corr_df = compute_correlation(df, **kwargs)
        return plot_scatter(
            corr_df,
            x=kwargs.get("x", "x_value"),
            y=kwargs.get("y", "y_value"),
            title=kwargs.get("title", "Correlation Analysis"),
            color=kwargs.get("color"),
            size=kwargs.get("size"),
            label_col=kwargs.get("label_col"),
            trendline=kwargs.get("trendline", False),
            x_label=kwargs.get("x_label"),
            y_label=kwargs.get("y_label"),
            height=kwargs.get("height", 450),
        )

    if task_key == "rating_summary":
        rating_df = compute_rating_summary(df, **kwargs)
        return plot_line(
            rating_df,
            x=kwargs.get("x", "period"),
            y_cols=kwargs.get("y_cols", ["rating"]),
            title=kwargs.get("title", "Rating Summary"),
            y_labels=kwargs.get("y_labels"),
            x_label=kwargs.get("x_label"),
            y_label=kwargs.get("y_label"),
            markers=kwargs.get("markers", False),
            height=kwargs.get("height", 420),
        )

    if task_key == "payment_summary":
        payment_df = compute_payment_summary(df, **kwargs)
        return plot_bar(
            payment_df,
            x=kwargs.get("x", "payment_type"),
            y=kwargs.get("y", "value"),
            title=kwargs.get("title", "Payment Summary"),
            orientation=kwargs.get("orientation", "v"),
            color=kwargs.get("color"),
            top_n=kwargs.get("top_n"),
            x_label=kwargs.get("x_label"),
            y_label=kwargs.get("y_label"),
            height=kwargs.get("height", 450),
            text_col=kwargs.get("text_col"),
        )

    if task_key == "basket_size_analysis":
        basket_df = compute_basket_size(df, **kwargs)
        return {
            "distribution": plot_distribution(
                basket_df[kwargs.get("series_col", "basket_size")],
                title=kwargs.get("distribution_title", "Basket Size Distribution"),
                bins=kwargs.get("bins", 40),
                show_box=kwargs.get("show_box", True),
                x_label=kwargs.get("x_label"),
                height=kwargs.get("height", 420),
            ),
            "scatter": plot_scatter(
                basket_df,
                x=kwargs.get("x", "basket_size"),
                y=kwargs.get("y", "revenue"),
                title=kwargs.get("scatter_title", "Basket Size vs Revenue"),
                color=kwargs.get("color"),
                size=kwargs.get("size"),
                label_col=kwargs.get("label_col"),
                trendline=kwargs.get("trendline", False),
                x_label=kwargs.get("x_label"),
                y_label=kwargs.get("y_label"),
                height=kwargs.get("scatter_height", 450),
            ),
        }

    if task_key == "composite_score":
        score_df = compute_composite_score(df, **kwargs)
        return plot_quadrant(
            score_df,
            x=kwargs.get("x", "score_x"),
            y=kwargs.get("y", "score_y"),
            size=kwargs.get("size", "score_size"),
            label_col=kwargs.get("label_col", "entity"),
            title=kwargs.get("title", "Composite Score"),
            x_label=kwargs.get("x_label"),
            y_label=kwargs.get("y_label"),
            height=kwargs.get("height", 520),
        )

    if task_key == "market_opportunity":
        market_df = compute_market_opportunity(df, **kwargs)
        return plot_quadrant(
            market_df,
            x=kwargs.get("x", "opportunity_x"),
            y=kwargs.get("y", "opportunity_y"),
            size=kwargs.get("size", "opportunity_size"),
            label_col=kwargs.get("label_col", "entity"),
            title=kwargs.get("title", "Market Opportunity"),
            x_label=kwargs.get("x_label"),
            y_label=kwargs.get("y_label"),
            height=kwargs.get("height", 520),
        )

    raise ValueError(f"Unhandled task_name: {task_name}")