"""
tools/__init__.py
─────────────────
Central export for every @tool and startup warmup.
main.py only needs: from tools import ALL_TOOLS, warmup_all
"""

from tools.chart import chart_build_tool, chart_decision_tool
from tools.rag import rag_tool, warmup as rag_warmup
from tools.sql import (
    db_connect_tool,
    sql_execute_tool,
    sql_generate_tool,
    warmup as sql_warmup,
)

# Full ordered list — handed to LangGraph's ToolNode
ALL_TOOLS = [
    db_connect_tool,  # 1 — ping the DB
    sql_generate_tool,  # 2 — NL → SQL
    sql_execute_tool,  # 3 — validate + run SQL
    chart_decision_tool,  # 4 — should we chart?
    chart_build_tool,  # 5 — build Plotly chart
    rag_tool,  # 6 — policy Q&A
]


def warmup_all() -> None:
    """Pre-load all heavy models at process startup."""
    sql_warmup()
    rag_warmup()
