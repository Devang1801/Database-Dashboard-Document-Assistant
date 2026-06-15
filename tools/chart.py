"""
tools/chart.py
──────────────
LangGraph tool: smart chart generation from SQL result sets.

Changes vs previous version
────────────────────────────
• donut → pie  — chart_type now "pie" (hole=0.38) to match frontend Plotly type
• _build_all_charts → _build_primary_chart  — backend sends ONE best-fit chart;
  the frontend table-chart dropdown now handles on-demand bar/line/pie/scatter
• chart_type strings aligned with frontend: "bar" | "hbar" | "line" | "pie" | "scatter"
• _should_chart  — simplified (keyword hint was dead code; logic unchanged)
• _CHART_KEYWORDS removed (unused)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_core.tools import tool

log = logging.getLogger("agent.tool.chart")

# Columns that are almost never meaningful as a y-axis (IDS-specific)
_ID_LIKE_COLS = {
    "proposal_refrence_num", "proposal_reference_num", "name_of_proposal",
    "thread_id", "message_id", "user_id", "username", "created_by",
}

_COMMON_LAYOUT = dict(
    margin={"t": 56, "l": 28, "r": 28, "b": 48},
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

_COLOR_SEQ = [
    "#1c2b4a", "#b5892a", "#8b1a1a", "#2d6a4f",
    "#5a4a8a", "#c07020", "#1a5276", "#6c3483",
    "#145a32", "#7e5109",
]


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Try to convert string columns that look numeric, handling commas and currency."""
    for col in df.columns:
        if df[col].dtype == object:
            cleaned = df[col].astype(str).str.replace(r'[^0-9.-]', '', regex=True)
            if cleaned.str.len().sum() > 0:
                converted = pd.to_numeric(cleaned, errors="coerce")
                if converted.notna().sum() > len(df) * 0.3:
                    df[col] = converted
    return df


def _coerce_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns whose name hints at a date/year into datetime."""
    date_hints = ("date", "yr", "year", "month", "period", "fin_yr")
    for col in df.columns:
        if df[col].dtype == object and any(h in col.lower() for h in date_hints):
            converted = pd.to_datetime(df[col], errors="coerce")
            if converted.notna().sum() > len(df) * 0.5:
                df[col] = converted
    return df


def _smart_prepare(
    columns: List[str], rows: List[List[Any]]
) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Return (df, x_col, y_cols) ready for Plotly, with aggregation applied
    whenever the raw data has repeated x values.

    Priority for x_col:  datetime  >  low-cardinality categorical
    y_cols:  all valid numeric columns, or ["Count"] if none.
    """
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        raise ValueError("Empty dataframe")

    df = _coerce_numerics(df)
    df = _coerce_datetimes(df)

    num_cols = list(df.select_dtypes(include="number").columns)
    dt_cols  = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols and c not in dt_cols]

    # ── Pick x_col ────────────────────────────────────────────────────────────
    x_col: Optional[str] = None
    if dt_cols:
        x_col = dt_cols[0]
    if x_col is None:
        candidates = [
            c for c in cat_cols
            if c.lower() not in _ID_LIKE_COLS and df[c].nunique() <= 100
        ]
        if candidates:
            x_col = min(candidates, key=lambda c: df[c].nunique())
    if x_col is None:
        x_col = df.columns[0]

    # ── Pick y_cols (all valid numeric columns) ───────────────────────────────
    y_cols: List[str] = []
    for col in num_cols:
        if col == x_col or col.lower() in _ID_LIKE_COLS:
            continue
        try:
            temp_y = pd.to_numeric(df[col], errors="coerce")
            if temp_y.notna().sum() > 0:
                df[col] = temp_y
                y_cols.append(col)
        except Exception:
            pass

    # No numeric y → create a count column
    if not y_cols:
        counts = df[x_col].value_counts().reset_index()
        counts.columns = [x_col, "Count"]
        counts = counts.sort_values("Count", ascending=False)
        return counts, x_col, ["Count"]

    # ── Aggregate repeated x values ───────────────────────────────────────────
    if df[x_col].duplicated().any():
        try:
            df = df.groupby(x_col, dropna=False, as_index=False)[y_cols].sum()
        except Exception as exc:
            log.warning(f"Aggregation failed: {exc}")
    else:
        for col in y_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ── Final cleanup ─────────────────────────────────────────────────────────
    df[x_col] = df[x_col].fillna("Unknown").astype(str)

    if not pd.api.types.is_datetime64_any_dtype(df[x_col]) and y_cols:
        try:
            df = df.sort_values(y_cols[0], ascending=False)
        except Exception:
            pass

    log.info(f"_smart_prepare: x='{x_col}', y={y_cols}, shape={df.shape}")
    return df, x_col, y_cols


def _calculate_kpis(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Calculate meaningful KPIs from the dataframe.
    Requires at least one numeric column.
    """
    kpis = []
    num_cols = list(df.select_dtypes(include="number").columns)
    cat_cols = [c for c in df.columns if c not in num_cols]
    
    if not num_cols:
        # If no numeric columns, maybe just count
        kpis.append({
            "label": "Total Records",
            "value": f"{len(df)}",
            "sub": "Count"
        })
        return kpis

    def fmt(n):
        if abs(n) >= 1e7: return f"{n / 1e7:.2f} Cr"
        if abs(n) >= 1e5: return f"{n / 1e5:.2f} L"
        if abs(n) >= 1e3: return f"{n / 1e3:.2f} K"
        return f"{n:.2f}"

    # General KPIs for numeric columns
    for col in num_cols[:2]:  # Limit to first 2 numeric columns
        total = df[col].sum()
        avg = df[col].mean()
        kpis.append({
            "label": f"Total {col}",
            "value": fmt(total),
            "sub": "Sum"
        })
        kpis.append({
            "label": f"Avg {col}",
            "value": fmt(avg),
            "sub": "Average"
        })

    # Cross-reference with categorical columns for "Top X"
    if cat_cols and num_cols:
        for c_col in cat_cols[:2]:
            for n_col in num_cols[:1]:
                try:
                    top = df.groupby(c_col)[n_col].sum().sort_values(ascending=False)
                    if not top.empty:
                        top_name = top.index[0]
                        top_val = top.iloc[0]
                        kpis.append({
                            "label": f"Top {c_col}",
                            "value": str(top_name)[:20],
                            "sub": f"{n_col}: {fmt(top_val)}"
                        })
                except Exception:
                    continue

    return kpis[:6] # Max 6 KPIs


# ─────────────────────────────────────────────────────────────────────────────
#  CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _bar(df: pd.DataFrame, x: str, y: str, title: str) -> Optional[str]:
    n = df[x].nunique()
    try:
        fig = px.bar(
            df, x=x, y=y, title=title,
            color=x if n <= 12 else None,
            text_auto='.2s',
            color_discrete_sequence=_COLOR_SEQ,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(**_COMMON_LAYOUT)
        return fig.to_json()
    except Exception as exc:
        log.warning(f"_bar error: {exc}")
        return None


def _hbar(df: pd.DataFrame, x: str, y: str, title: str) -> Optional[str]:
    n = df[x].nunique()
    try:
        df_s = df.sort_values(y, ascending=True)
        fig = px.bar(
            df_s, y=x, x=y, title=title,
            orientation="h",
            color=x if n <= 12 else None,
            text_auto='.2s',
            color_discrete_sequence=_COLOR_SEQ,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(**_COMMON_LAYOUT)
        return fig.to_json()
    except Exception as exc:
        log.warning(f"_hbar error: {exc}")
        return None


def _line(df: pd.DataFrame, x: str, y: str, title: str) -> Optional[str]:
    try:
        df_s = df.sort_values(x)
        fig = px.line(
            df_s, x=x, y=y, title=title,
            markers=True,
            color_discrete_sequence=_COLOR_SEQ,
        )
        fig.update_layout(**_COMMON_LAYOUT)
        return fig.to_json()
    except Exception as exc:
        log.warning(f"_line error: {exc}")
        return None


def _pie(df: pd.DataFrame, x: str, y: str, title: str) -> Optional[str]:
    """Donut-style pie — hole=0.38 matches the frontend renderDynamicCharts pie."""
    try:
        fig = px.pie(
            df, names=x, values=y, title=title,
            hole=0.38,
            color_discrete_sequence=_COLOR_SEQ,
        )
        fig.update_traces(textinfo="label+percent", textposition="inside")
        fig.update_layout(**_COMMON_LAYOUT)
        return fig.to_json()
    except Exception as exc:
        log.warning(f"_pie error: {exc}")
        return None


def _scatter(
    df: pd.DataFrame, x: str, y: str, label: str, title: str
) -> Optional[str]:
    try:
        fig = px.scatter(
            df, x=x, y=y,
            text=label if df[label].nunique() <= 20 else None,
            title=title,
            color_discrete_sequence=_COLOR_SEQ,
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(**_COMMON_LAYOUT)
        return fig.to_json()
    except Exception as exc:
        log.warning(f"_scatter error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  PRIMARY CHART SELECTION
#  The frontend now handles on-demand bar/line/pie/scatter via the table-chart
#  dropdown, so the backend picks ONE best-fit chart to accompany the response.
# ─────────────────────────────────────────────────────────────────────────────

def _pick_primary_chart_type(is_dt: bool, n_unique: int, n_y_cols: int) -> str:
    """
    Rule-based selection of the single best chart type for a result set.

    is_dt      — True when the x-axis is a datetime column
    n_unique   — cardinality of the x-axis
    n_y_cols   — number of numeric y columns available
    """
    if is_dt or n_unique >= 4:
        return "line"      # time-series or many categories → trend line
    if 2 <= n_unique <= 8:
        return "pie"       # few categories with a numeric metric → pie
    return "bar"           # fallback


def _build_primary_chart(
    columns: List[str], rows: List[List[Any]], query: str = ""
) -> Optional[Dict[str, str]]:
    """
    Build the single most meaningful chart for this result set.
    Returns None if no chart is possible.
    """
    try:
        df, x_col, y_cols = _smart_prepare(columns, rows)
        df = df[df[x_col].notna()]
    except Exception as exc:
        log.error(f"_smart_prepare failed: {exc}")
        return None

    if df.empty or len(df) < 2:
        return None

    kpis = _calculate_kpis(df)

    is_dt    = pd.api.types.is_datetime64_any_dtype(df[x_col])
    n_unique = df[x_col].nunique()
    y_col    = y_cols[0]
    title    = f"{y_col} by {x_col}"

    chart_type = _pick_primary_chart_type(is_dt, n_unique, len(y_cols))
    log.info(f"Primary chart: {chart_type} (n_unique={n_unique}, is_dt={is_dt})")

    builders = {
        "bar":     lambda: _bar(df, x_col, y_col, f"Bar — {title}"),
        "hbar":    lambda: _hbar(df, x_col, y_col, f"Horizontal Bar — {title}"),
        "line":    lambda: _line(df, x_col, y_col, f"Trend — {title}"),
        "pie":     lambda: _pie(df, x_col, y_col, f"Distribution — {title}"),
        "scatter": lambda: (
            _scatter(df, y_cols[0], y_cols[1], x_col,
                     f"Comparison — {y_cols[0]} vs {y_cols[1]}")
            if len(y_cols) >= 2 else None
        ),
    }

    json_str = builders[chart_type]()
    if not json_str:
        # Fallback to bar
        json_str = _bar(df, x_col, y_col, f"Bar — {title}")
        chart_type = "bar"

    return {
        "chart_json": json_str, 
        "chart_type": chart_type,
        "kpis": kpis
    } if json_str else None


# ─────────────────────────────────────────────────────────────────────────────
#  BACKWARD-COMPAT: _build_all_charts still available for callers that need it
# ─────────────────────────────────────────────────────────────────────────────

def _build_all_charts(
    columns: List[str], rows: List[List[Any]], query: str = ""
) -> List[Dict[str, str]]:
    """
    Build ALL applicable chart types. Kept for backward compatibility.
    Prefer _build_primary_chart in new code.

    chart_type strings: "bar" | "hbar" | "line" | "pie" | "scatter"
    (was: "donut" — now renamed to "pie" to align with Plotly and frontend)
    """
    results: List[Dict[str, str]] = []

    try:
        df, x_col, y_cols = _smart_prepare(columns, rows)
        df = df[df[x_col].notna()]
    except Exception as exc:
        log.error(f"_smart_prepare failed: {exc}")
        return []

    if df.empty or len(df) < 2:
        return []

    is_dt    = pd.api.types.is_datetime64_any_dtype(df[x_col])
    n_unique = df[x_col].nunique()

    for y_col in y_cols[:3]:
        title = f"{y_col} by {x_col}"

        # Bar / HBar
        if not is_dt and n_unique <= 30:
            j = _bar(df, x_col, y_col, f"Bar Chart — {title}")
            if j: results.append({"chart_json": j, "chart_type": "bar"})

            if n_unique <= 15:
                j = _hbar(df, x_col, y_col, f"Horizontal Bar — {title}")
                if j: results.append({"chart_json": j, "chart_type": "hbar"})

        # Line
        if is_dt or n_unique >= 3:
            j = _line(df, x_col, y_col, f"Trend — {title}")
            if j: results.append({"chart_json": j, "chart_type": "line"})

        # Pie  (was "donut" — renamed to "pie")
        if not is_dt and 2 <= n_unique <= 10:
            j = _pie(df, x_col, y_col, f"Distribution — {title}")
            if j: results.append({"chart_json": j, "chart_type": "pie"})

    # Scatter (cross-metric comparison)
    if len(y_cols) >= 2:
        j = _scatter(df, y_cols[0], y_cols[1], x_col,
                     f"Comparison — {y_cols[0]} vs {y_cols[1]}")
        if j: results.append({"chart_json": j, "chart_type": "scatter"})

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  BACKWARD-COMPAT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _should_chart(columns: List[str], rows: List[List[Any]], query: str = "") -> bool:
    """Return True when the result set is worth charting."""
    if not columns or not rows:
        return False
    if len(columns) < 2 or len(rows) < 2:
        return False
    if len(rows) > 3000:
        return False
    return True


def _build_chart(
    columns: List[str], rows: List[List[Any]], query: str = ""
) -> Dict[str, Any]:
    result = _build_primary_chart(columns, rows, query)
    if result:
        return {
            "chart_json": result["chart_json"], 
            "chart_type": result["chart_type"],
            "kpis": result.get("kpis", [])
        }
    return {"chart_json": "", "chart_type": "", "kpis": [], "error": "No chart could be built"}


# ─────────────────────────────────────────────────────────────────────────────
#  LANGCHAIN @tools
# ─────────────────────────────────────────────────────────────────────────────

@tool
def chart_decision_tool(
    columns: List[str], rows: List[List[Any]], query: str
) -> Dict[str, bool]:
    """
    Decide whether the SQL result set warrants a chart.

    Args:
        columns: Column names from the SQL result.
        rows:    Row data from the SQL result.
        query:   Original user query.

    Returns:
        dict with key: should_chart (bool).
    """
    decision = _should_chart(columns, rows, query)
    log.info(f"chart_decision_tool → should_chart={decision}")
    return {"should_chart": decision}


@tool
def chart_build_tool(
    columns: List[str], rows: List[List[Any]], query: str = ""
) -> Dict[str, Any]:
    """
    Build the best-fit Plotly chart from a SQL result set.

    The frontend table-chart dropdown handles on-demand bar/line/pie/scatter
    rendering, so the backend sends ONE primary chart (the most information-
    dense view for the data shape).

    chart_jsons is still a list for backward compat — it contains one entry.

    chart_type strings: "bar" | "hbar" | "line" | "pie" | "scatter"

    Args:
        columns: Column names from the SQL result.
        rows:    Row data from the SQL result.
        query:   Optional original query (used for chart titles).

    Returns:
        dict with keys:
          chart_json   (str)       — Plotly JSON of the primary chart
          chart_jsons  (list[str]) — [chart_json] single-item list (compat)
          chart_type   (str)       — type of the primary chart
          error        (str)       — non-empty on failure
    """
    try:
        result = _build_primary_chart(columns, rows, query)
        if not result:
            return {
                "chart_json": "", "chart_jsons": [],
                "chart_type": "", "kpis": [], "error": "No chart could be built",
            }
        return {
            "chart_json":  result["chart_json"],
            "chart_jsons": [result["chart_json"]],
            "chart_type":  result["chart_type"],
            "kpis":        result.get("kpis", []),
            "error": "",
        }
    except Exception as exc:
        log.error(f"chart_build_tool error: {exc}")
        return {
            "chart_json": "", "chart_jsons": [],
            "chart_type": "", "kpis": [], "error": str(exc),
        }