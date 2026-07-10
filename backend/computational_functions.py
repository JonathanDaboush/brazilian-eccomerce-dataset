"""
computational_functions.py
--------------------------
General-purpose analytical building blocks for the Olist e-commerce dataset.

Every function accepts a DataFrame plus explicit column-name parameters so the
same function works regardless of how columns are named upstream.  Functions
return plain DataFrames or dicts — nothing is printed or plotted here.

Typical usage pattern
---------------------
1. Join / filter your raw tables into a flat analysis DataFrame.
2. Call the building-block function(s) you need, passing the column names.
3. Pass the returned DataFrame / dict to the matching visualization function.
"""

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 1. SCALAR KPIs
# ──────────────────────────────────────────────────────────────────────────────

def compute_kpis(df, revenue_col, order_col, customer_col=None, seller_col=None, freight_col=None):
    """
    Return a dict of top-level scalar KPIs suitable for KPI cards.

    Parameters
    ----------
    df          : DataFrame — one row per order-item (or order) with all relevant cols.
    revenue_col : str — column containing item/order revenue (price).
    order_col   : str — column containing order identifiers.
    customer_col: str | None — column containing customer identifiers.
    seller_col  : str | None — column containing seller identifiers.
    freight_col : str | None — column containing freight value.

    Returns
    -------
    dict with keys:
        total_revenue, total_orders, aov,
        total_customers (if customer_col given),
        total_sellers   (if seller_col given),
        total_freight   (if freight_col given),
        freight_ratio   (if both freight_col and revenue_col given)
    """
    kpis = {}
    kpis["total_revenue"]  = df[revenue_col].sum()
    kpis["total_orders"]   = df[order_col].nunique()
    kpis["aov"]            = kpis["total_revenue"] / kpis["total_orders"] if kpis["total_orders"] else 0

    if customer_col:
        kpis["total_customers"] = df[customer_col].nunique()

    if seller_col:
        kpis["total_sellers"] = df[seller_col].nunique()

    if freight_col:
        kpis["total_freight"] = df[freight_col].sum()
        kpis["freight_ratio"] = kpis["total_freight"] / kpis["total_revenue"] if kpis["total_revenue"] else 0

    return kpis


# ──────────────────────────────────────────────────────────────────────────────
# 2. GENERIC GROUPBY AGGREGATION
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_by(df, group_col, agg_map, rank_col=None, top_n=None, ascending=False):
    """
    Group df by one or more columns and apply arbitrary aggregations.

    Parameters
    ----------
    df        : DataFrame
    group_col : str | list[str] — column(s) to group by.
    agg_map   : dict — {new_col_name: (source_col, agg_func)} passed to
                pd.NamedAgg, e.g. {"revenue": ("price", "sum"), "orders": ("order_id", "nunique")}.
    rank_col  : str | None — if set, adds a 'rank' column sorted by this col.
    top_n     : int | None — keep only the top N rows after ranking.
    ascending : bool — sort direction for ranking (default False = highest first).

    Returns
    -------
    DataFrame with group_col + all aggregated columns + optional rank.
    """
    named_aggs = {k: pd.NamedAgg(column=v[0], aggfunc=v[1]) for k, v in agg_map.items()}
    result = df.groupby(group_col, as_index=False).agg(**named_aggs)

    if rank_col:
        result = result.sort_values(rank_col, ascending=ascending).reset_index(drop=True)
        result["rank"] = result[rank_col].rank(ascending=ascending, method="dense").astype(int)

    if top_n is not None and rank_col:
        result = result.head(top_n)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 3. TIME-SERIES AGGREGATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_monthly_series(df, date_col, value_col, agg="sum", additional_cols=None):
    """
    Resample any numeric column to a monthly time series.

    Parameters
    ----------
    df              : DataFrame
    date_col        : str — datetime column (will be coerced if not already datetime).
    value_col       : str — column to aggregate.
    agg             : str — aggregation: 'sum', 'mean', 'count', 'nunique'.
    additional_cols : list[str] | None — extra numeric cols to aggregate the same way.

    Returns
    -------
    DataFrame with columns: period (Period[M]), {value_col}, [additional_cols].
    Sorted ascending by period.
    """
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work["period"] = work[date_col].dt.to_period("M")

    cols = [value_col] + (additional_cols or [])
    grouped = work.groupby("period")[cols].agg(agg).reset_index()
    return grouped.sort_values("period").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# 4. GROWTH RATE
# ──────────────────────────────────────────────────────────────────────────────

def compute_growth_rate(series, periods=1):
    """
    Compute period-over-period percentage change for a numeric Series.

    Parameters
    ----------
    series  : pd.Series — ordered time-series values.
    periods : int — number of steps to look back (default 1 = MoM).

    Returns
    -------
    pd.Series of growth rates (same index as input, NaN for initial rows).
    """
    return series.pct_change(periods=periods) * 100


# ──────────────────────────────────────────────────────────────────────────────
# 5. ROLLING AVERAGE
# ──────────────────────────────────────────────────────────────────────────────

def compute_rolling_avg(series, window=3, min_periods=1):
    """
    Compute a rolling mean over a numeric Series.

    Parameters
    ----------
    series      : pd.Series
    window      : int — rolling window size (default 3).
    min_periods : int — minimum observations required (default 1).

    Returns
    -------
    pd.Series of rolling means.
    """
    return series.rolling(window=window, min_periods=min_periods).mean()


# ──────────────────────────────────────────────────────────────────────────────
# 6. REVENUE / SHARE CONCENTRATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_revenue_share(df, group_col, revenue_col, top_n=None):
    """
    Compute revenue share (%) and rank for any grouping dimension.

    Covers: customer concentration, seller concentration, product/category share.

    Parameters
    ----------
    df          : DataFrame
    group_col   : str | list[str] — entity to group by (customer, seller, product …).
    revenue_col : str — revenue column.
    top_n       : int | None — return only the top N entities.

    Returns
    -------
    DataFrame with: group_col, revenue, revenue_share_pct, rank.
    Sorted by revenue descending.
    """
    agg = df.groupby(group_col, as_index=False)[revenue_col].sum()
    agg = agg.rename(columns={revenue_col: "revenue"})
    agg = agg.sort_values("revenue", ascending=False).reset_index(drop=True)
    agg["revenue_share_pct"] = agg["revenue"] / agg["revenue"].sum() * 100
    agg["rank"] = range(1, len(agg) + 1)

    if top_n:
        agg = agg.head(top_n)

    return agg


# ──────────────────────────────────────────────────────────────────────────────
# 7. PARETO / CUMULATIVE SHARE
# ──────────────────────────────────────────────────────────────────────────────

def compute_pareto(df, group_col, revenue_col):
    """
    Build a Pareto table: sorted entities with individual and cumulative revenue share.

    Useful for: customer 80/20, seller 80/20, product 80/20 analyses.

    Parameters
    ----------
    df          : DataFrame
    group_col   : str — entity column.
    revenue_col : str — revenue column.

    Returns
    -------
    DataFrame with: group_col, revenue, revenue_share_pct, cumulative_share_pct, rank.
    """
    result = compute_revenue_share(df, group_col, revenue_col)
    result["cumulative_share_pct"] = result["revenue_share_pct"].cumsum()
    return result


def compute_concentration_stats(df, group_col, revenue_col, percentiles=(0.01, 0.05, 0.10)):
    """
    Compute how much revenue the top X% of entities generate.

    Parameters
    ----------
    df           : DataFrame
    group_col    : str — entity column (e.g. customer_id).
    revenue_col  : str — revenue column.
    percentiles  : tuple — top-N fractions to evaluate (default 1%, 5%, 10%).

    Returns
    -------
    dict mapping each percentile label → revenue share % for that top slice.
    Example: {"top_1pct": 12.3, "top_5pct": 34.5, "top_10pct": 55.1, "pareto_ratio": 0.2}
    """
    entity_rev = df.groupby(group_col)[revenue_col].sum().sort_values(ascending=False)
    total = entity_rev.sum()
    n = len(entity_rev)
    stats = {}

    for p in percentiles:
        k = max(1, int(np.ceil(n * p)))
        label = f"top_{int(p * 100)}pct"
        stats[label] = entity_rev.iloc[:k].sum() / total * 100

    # Pareto ratio: fraction of entities that account for 80 % of revenue
    cumshare = entity_rev.cumsum() / total
    pareto_count = (cumshare < 0.80).sum() + 1
    stats["pareto_ratio"] = pareto_count / n

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# 8. DISTRIBUTION STATISTICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_distribution_stats(df, group_col, value_col, agg="sum"):
    """
    Compute per-entity totals then return descriptive statistics of that distribution.

    Covers: CLV distribution, AOV distribution, order-value distribution.

    Parameters
    ----------
    df        : DataFrame
    group_col : str — entity to aggregate to (e.g. customer_id, order_id).
    value_col : str — numeric column to aggregate.
    agg       : str — how to aggregate per entity ('sum' for CLV, 'mean' for AOV).

    Returns
    -------
    tuple:
        entity_series — pd.Series of per-entity aggregated values (for histogram input).
        stats_dict    — dict with mean, median, std, min, max, p25, p75.
    """
    entity_series = df.groupby(group_col)[value_col].agg(agg)

    stats_dict = {
        "mean":   entity_series.mean(),
        "median": entity_series.median(),
        "std":    entity_series.std(),
        "min":    entity_series.min(),
        "max":    entity_series.max(),
        "p25":    entity_series.quantile(0.25),
        "p75":    entity_series.quantile(0.75),
    }

    return entity_series, stats_dict


# ──────────────────────────────────────────────────────────────────────────────
# 9. DELIVERY METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_delivery_metrics(df, purchase_col, delivery_col, estimated_col=None, group_col=None):
    """
    Compute delivery time statistics and (optionally) on-time performance.

    Parameters
    ----------
    df             : DataFrame — one row per order.
    purchase_col   : str — datetime column for order purchase timestamp.
    delivery_col   : str — datetime column for actual delivery to customer.
    estimated_col  : str | None — datetime column for promised delivery date.
                     If provided, adds late_delivery_rate and avg_delay_days.
    group_col      : str | None — if set, computes metrics per group
                     (e.g. seller_id, customer_state). If None, returns overall stats.

    Returns
    -------
    DataFrame with: [group_col,] avg_delivery_days, median_delivery_days,
                    and (if estimated_col) late_rate_pct, avg_delay_days.
    """
    work = df.copy()
    work[purchase_col]  = pd.to_datetime(work[purchase_col])
    work[delivery_col]  = pd.to_datetime(work[delivery_col])
    work["delivery_days"] = (work[delivery_col] - work[purchase_col]).dt.days

    if estimated_col:
        work[estimated_col] = pd.to_datetime(work[estimated_col])
        work["delay_days"]  = (work[delivery_col] - work[estimated_col]).dt.days
        work["is_late"]     = work["delay_days"] > 0

    # Drop rows where delivery hasn't happened yet
    work = work.dropna(subset=["delivery_days"])
    work = work[work["delivery_days"] >= 0]

    if group_col is None:
        result = {
            "avg_delivery_days":    work["delivery_days"].mean(),
            "median_delivery_days": work["delivery_days"].median(),
        }
        if estimated_col:
            result["late_rate_pct"]  = work["is_late"].mean() * 100
            result["early_rate_pct"] = (~work["is_late"]).mean() * 100
            result["avg_delay_days"] = work["delay_days"].mean()
        return result
    else:
        agg_map = {
            "avg_delivery_days":    ("delivery_days", "mean"),
            "median_delivery_days": ("delivery_days", "median"),
            "order_count":          ("delivery_days", "count"),
        }
        if estimated_col:
            agg_map["late_rate_pct"] = ("is_late", "mean")
            agg_map["avg_delay_days"] = ("delay_days", "mean")

        named_aggs = {k: pd.NamedAgg(column=v[0], aggfunc=v[1]) for k, v in agg_map.items()}
        result = work.groupby(group_col, as_index=False).agg(**named_aggs)

        if "late_rate_pct" in result.columns:
            result["late_rate_pct"] = result["late_rate_pct"] * 100

        result = result.sort_values("avg_delivery_days", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# 10. FREIGHT RATIO
# ──────────────────────────────────────────────────────────────────────────────

def compute_freight_ratio(df, freight_col, revenue_col, group_col=None):
    """
    Compute freight cost as a percentage of revenue.

    Parameters
    ----------
    df          : DataFrame
    freight_col : str — freight value column.
    revenue_col : str — revenue column.
    group_col   : str | None — if set, computes ratio per group (region, seller, order).

    Returns
    -------
    If group_col is None: dict with total_freight, total_revenue, freight_ratio_pct.
    If group_col given:   DataFrame with group_col, total_freight, total_revenue,
                          freight_ratio_pct, sorted by freight_ratio_pct descending.
    """
    if group_col is None:
        total_freight  = df[freight_col].sum()
        total_revenue  = df[revenue_col].sum()
        return {
            "total_freight":     total_freight,
            "total_revenue":     total_revenue,
            "freight_ratio_pct": (total_freight / total_revenue * 100) if total_revenue else 0,
        }

    result = df.groupby(group_col, as_index=False).agg(
        total_freight=(freight_col, "sum"),
        total_revenue=(revenue_col, "sum"),
    )
    result["freight_ratio_pct"] = (
        result["total_freight"] / result["total_revenue"].replace(0, np.nan) * 100
    )
    return result.sort_values("freight_ratio_pct", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# 11. CORRELATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_correlation(df, col_x, col_y, group_col=None, agg="mean"):
    """
    Compute Pearson correlation between two columns.

    If group_col is provided, aggregates each column per group first
    (e.g. avg delivery time vs avg review score per seller).

    Parameters
    ----------
    df        : DataFrame
    col_x     : str — first numeric column.
    col_y     : str — second numeric column.
    group_col : str | None — entity to aggregate to before correlating.
    agg       : str — aggregation to apply when group_col is set.

    Returns
    -------
    dict with: correlation (float), scatter_df (DataFrame with col_x, col_y[, group_col]).
    """
    if group_col:
        scatter_df = df.groupby(group_col, as_index=False)[[col_x, col_y]].agg(agg)
    else:
        scatter_df = df[[col_x, col_y]].dropna()

    corr = scatter_df[col_x].corr(scatter_df[col_y])
    return {"correlation": corr, "scatter_df": scatter_df}


# ──────────────────────────────────────────────────────────────────────────────
# 12. BASKET SIZE
# ──────────────────────────────────────────────────────────────────────────────

def compute_basket_size(df, order_col, item_col=None, revenue_col=None):
    """
    Compute items-per-order statistics and (optionally) revenue per basket.

    Parameters
    ----------
    df          : DataFrame — one row per order-item.
    order_col   : str — order identifier column.
    item_col    : str | None — item identifier or any col to count rows per order.
                  If None, counts rows.
    revenue_col : str | None — if given, also returns avg revenue per order.

    Returns
    -------
    tuple:
        basket_series — pd.Series of item counts per order (for histogram input).
        stats_dict    — dict with mean_basket, median_basket, std_basket,
                        and (if revenue_col) avg_order_revenue, correlation.
    """
    if item_col:
        basket = df.groupby(order_col)[item_col].count()
    else:
        basket = df.groupby(order_col).size()

    basket.name = "basket_size"

    stats = {
        "mean_basket":   basket.mean(),
        "median_basket": basket.median(),
        "std_basket":    basket.std(),
        "max_basket":    basket.max(),
    }

    if revenue_col:
        order_rev = df.groupby(order_col)[revenue_col].sum()
        combined  = pd.concat([basket, order_rev], axis=1).dropna()
        stats["avg_order_revenue"] = combined[revenue_col].mean()
        stats["basket_revenue_corr"] = combined["basket_size"].corr(combined[revenue_col])

    return basket, stats


# ──────────────────────────────────────────────────────────────────────────────
# 13. RATING / SATISFACTION
# ──────────────────────────────────────────────────────────────────────────────

def compute_rating_summary(df, rating_col, group_col=None, date_col=None, min_reviews=5):
    """
    Summarise review scores — overall, by entity, or over time.

    Parameters
    ----------
    df          : DataFrame
    rating_col  : str — numeric review score column.
    group_col   : str | None — entity to group by (seller, category, state).
    date_col    : str | None — if given, returns a monthly time series instead.
    min_reviews : int — filter out entities with fewer reviews than this threshold.

    Returns
    -------
    If neither group_col nor date_col: dict with mean, median, distribution counts.
    If group_col: DataFrame with group_col, mean_rating, review_count, sorted descending.
    If date_col:  DataFrame with period (Month), mean_rating, review_count.
    """
    work = df.dropna(subset=[rating_col])

    if date_col:
        work[date_col] = pd.to_datetime(work[date_col])
        work["period"] = work[date_col].dt.to_period("M")
        result = work.groupby("period", as_index=False).agg(
            mean_rating=(rating_col, "mean"),
            review_count=(rating_col, "count"),
        )
        return result.sort_values("period").reset_index(drop=True)

    if group_col:
        result = work.groupby(group_col, as_index=False).agg(
            mean_rating=(rating_col, "mean"),
            review_count=(rating_col, "count"),
        )
        result = result[result["review_count"] >= min_reviews]
        return result.sort_values("mean_rating", ascending=False).reset_index(drop=True)

    return {
        "mean":   work[rating_col].mean(),
        "median": work[rating_col].median(),
        "distribution": work[rating_col].value_counts().sort_index().to_dict(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 14. PAYMENT ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def compute_payment_summary(df, payment_type_col, value_col, installment_col=None):
    """
    Summarise revenue and order value by payment method and instalment behaviour.

    Parameters
    ----------
    df               : DataFrame — one row per payment record.
    payment_type_col : str — payment method column (credit_card, boleto, etc.).
    value_col        : str — payment value column.
    installment_col  : str | None — number of instalments column.
                       If given, also returns installment-spend analysis.

    Returns
    -------
    dict with:
        by_type        — DataFrame: payment_type, total_revenue, revenue_share_pct,
                         avg_payment, order_count.
        installments   — DataFrame (only if installment_col given):
                         installments, avg_payment, order_count, total_revenue.
    """
    by_type = df.groupby(payment_type_col, as_index=False).agg(
        total_revenue=(value_col, "sum"),
        avg_payment=(value_col, "mean"),
        order_count=(value_col, "count"),
    )
    by_type["revenue_share_pct"] = by_type["total_revenue"] / by_type["total_revenue"].sum() * 100
    by_type = by_type.sort_values("total_revenue", ascending=False).reset_index(drop=True)

    result = {"by_type": by_type}

    if installment_col:
        inst = df.groupby(installment_col, as_index=False).agg(
            avg_payment=(value_col, "mean"),
            order_count=(value_col, "count"),
            total_revenue=(value_col, "sum"),
        )
        result["installments"] = inst.sort_values(installment_col)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 15. COMPOSITE SCORE  (strategic ranking)
# ──────────────────────────────────────────────────────────────────────────────

def compute_composite_score(df, index_col, metric_cols, weights):
    """
    Build a weighted composite score across multiple normalised metrics.

    Used for strategic questions like "best category to invest in" or
    "strongest seller partnership" where several signals must be combined.

    Parameters
    ----------
    df          : DataFrame — one row per entity (already aggregated).
    index_col   : str — entity identifier column.
    metric_cols : list[str] — columns to include in the score.
                  Higher values = better for every column.  Flip sign upstream
                  for metrics where lower = better (e.g. delivery_days * -1).
    weights     : list[float] — weight for each metric_col (must sum to 1).

    Returns
    -------
    DataFrame with: index_col, each metric_col (normalised 0-1), composite_score, rank.
    Sorted by composite_score descending.

    Example
    -------
    compute_composite_score(
        df          = cat_summary,
        index_col   = "product_category",
        metric_cols = ["revenue", "units_sold", "mean_rating", "revenue_growth"],
        weights     = [0.40,       0.20,          0.25,          0.15],
    )
    """
    assert len(metric_cols) == len(weights), "metric_cols and weights must have the same length."
    assert abs(sum(weights) - 1.0) < 1e-6, "weights must sum to 1."

    work = df[[index_col] + metric_cols].copy().dropna()

    # Min-max normalise each metric to [0, 1]
    for col in metric_cols:
        col_min = work[col].min()
        col_max = work[col].max()
        denom   = col_max - col_min
        work[f"{col}_norm"] = (work[col] - col_min) / denom if denom else 0.0

    norm_cols = [f"{col}_norm" for col in metric_cols]
    work["composite_score"] = sum(work[nc] * w for nc, w in zip(norm_cols, weights))
    work = work.sort_values("composite_score", ascending=False).reset_index(drop=True)
    work["rank"] = range(1, len(work) + 1)

    return work
