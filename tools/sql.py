"""
tools/sql.py
────────────
...
Three focused @tools for the SQL pipeline:

  1. db_connect_tool    — validate config & verify PostgreSQL is reachable
  2. sql_generate_tool  — NL → SQL via local Qwen LLM
  3. sql_execute_tool   — validate (no mutations) + execute + return rows
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psycopg2
    from psycopg2.extensions import connection as _PgConnection

    DB_CLIENT = "psycopg2"
except Exception:
    import pg8000

    _PgConnection = Any
    DB_CLIENT = "pg8000"

from gateway.llm_manager import get_shared_pipeline, get_shared_model_path
from langchain_core.tools import tool

log = logging.getLogger("agent.tool.sql")

from datetime import datetime, date
import decimal

# ═════════════════════════════════════════════════════════════════════════════
# DATABASE CONFIG
# ═════════════════════════════════════════════════════════════════════════════

DB_CONFIG: Dict[str, Any] = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
    "dbname": os.getenv("DB_NAME", "stock"),
    "port": int(os.getenv("DB_PORT", "5432")),
}

DB_CONFIG_PUBLIC: Dict[str, Any] = {
    k: v for k, v in DB_CONFIG.items() if k != "password"
}


def _open_connection() -> _PgConnection:
    config = dict(DB_CONFIG)

    if DB_CLIENT == "pg8000":
        config["database"] = config.pop("dbname")
        return pg8000.connect(**config)

    return psycopg2.connect(**config)


@tool
def db_connect_tool() -> Dict[str, Any]:
    """Validate config and verify PostgreSQL is reachable."""
    try:
        conn = _open_connection()
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()[0].split("\n")[0]
        conn.close()

        return {
            "connected": True,
            "config": DB_CONFIG_PUBLIC,
            "server_version": version,
            "error": "",
        }

    except Exception as exc:
        return {
            "connected": False,
            "config": DB_CONFIG_PUBLIC,
            "server_version": "",
            "error": str(exc),
        }


# ═════════════════════════════════════════════════════════════════════════════
# SQL GENERATION
# ═════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SQL_MODEL_PATH = os.getenv("SQL_MODEL_PATH", get_shared_model_path())

SQL_LLM_CONFIG = {
    "max_new_tokens": 256,
    "max_length": None,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "repetition_penalty": 1.1,
    "do_sample": False,
}

DATABASE_SCHEMA = """
Table: public.market

COLUMN NAME             TYPE                ALIASES / NOTES
stock_id                double precision    Stock ID
ticker                  text                Stock Symbol
company_name            text                Company Name
country                 text                Country
sector                  text                Sector
exchange                text                Stock Exchange
currency                text                Currency
trade_date              text                Trading Date
open_price              double precision    Open Price
high_price              double precision    High Price
low_price               double precision    Low Price
close_price             double precision    Close Price
adjusted_close          double precision    Adjusted Close Price
volume                  double precision    Trading Volume
market_cap_billion      double precision    Market Cap in Billions
pe_ratio                double precision    PE Ratio
dividend_yield          double precision    Dividend Yield
"""

_GENERATION_SYSTEM_PROMPT = f"""
You are a senior PostgreSQL engineer specialized in stock market analytics.

SCHEMA:
{DATABASE_SCHEMA}

RULES:

1. Output ONLY raw SQL SELECT query.
2. Never use INSERT/UPDATE/DELETE/DROP/TRUNCATE.
3. Use exact column names only.
4. Use ILIKE for text filtering.
5. Never assume filters not asked by user.
6. Add LIMIT 2000 for row queries.
7. Omit LIMIT for aggregations.
8. Never use markdown or comments.
9. Use SELECT * when user asks all details.
10. Use GROUP BY for category queries.
11. Default columns:
   ticker,
   company_name,
   sector,
   country,
   close_price,
   volume

EXAMPLES:

Q: show all stocks
SELECT ticker, company_name, sector, country, close_price, volume
FROM public.market
LIMIT 2000

Q: show all details
SELECT * FROM public.market LIMIT 2000

Q: top companies by market cap
SELECT company_name, market_cap_billion
FROM public.market
ORDER BY market_cap_billion DESC
LIMIT 10

Q: highest volume stocks
SELECT ticker, company_name, volume
FROM public.market
ORDER BY volume DESC
LIMIT 10

Q: average PE ratio by country
SELECT country, AVG(pe_ratio) AS avg_pe_ratio
FROM public.market
GROUP BY country
ORDER BY avg_pe_ratio DESC

Return ONLY SQL query.
"""

_FENCE_RE = re.compile(r"```sql|```", re.IGNORECASE)
_STOP_RE = re.compile(r"<\|im_end\|>.*", re.DOTALL)

_sql_pipeline = None


def get_sql_pipeline():
    global _sql_pipeline

    if _sql_pipeline is not None:
        return _sql_pipeline

    _sql_pipeline = get_shared_pipeline(SQL_MODEL_PATH)

    return _sql_pipeline


def _build_generation_prompt(
    query: str,
    memory: Optional[List[Dict[str, str]]] = None,
) -> str:

    parts = [f"<|im_start|>system\n{_GENERATION_SYSTEM_PROMPT}<|im_end|>"]

    for turn in (memory or [])[-3:]:
        role = "user" if turn["role"] == "user" else "assistant"

        parts.append(f"<|im_start|>{role}\n{turn['content'][:300]}<|im_end|>")

    parts.append(f"<|im_start|>user\nQuestion: {query}<|im_end|>")

    parts.append("<|im_start|>assistant\nSELECT")

    return "\n".join(parts)


def _clean_sql(raw: str) -> str:
    sql = _STOP_RE.sub("", raw).strip()
    sql = _FENCE_RE.sub("", sql).strip()

    if not sql.upper().startswith("SELECT"):
        sql = "SELECT " + sql

    sql = re.sub(r"--[^\n]*", "", sql)

    return sql.strip().rstrip(";").strip()


_ALL_DETAILS_KEYWORDS = {
    "all details",
    "all columns",
    "full details",
    "all information",
    "everything",
}


def _force_select_star_if_needed(query: str, sql: str) -> str:
    q_lower = query.lower()

    if not any(k in q_lower for k in _ALL_DETAILS_KEYWORDS):
        return sql

    if re.match(r"SELECT\s+\*", sql, re.IGNORECASE):
        return sql

    fixed = re.sub(
        r"^SELECT\s+.+?\s+FROM\b",
        "SELECT * FROM",
        sql,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return fixed


@tool
def sql_generate_tool(query: str, memory: str = "[]") -> Dict[str, Any]:
    """Generate a SQL SELECT query from a natural language question."""
    t0 = time.time()

    try:
        mem = json.loads(memory) if isinstance(memory, str) else memory

        prompt = _build_generation_prompt(query, mem)

        output = get_sql_pipeline()(prompt, **SQL_LLM_CONFIG)

        sql = _clean_sql(output[0]["generated_text"])

        sql = _force_select_star_if_needed(query, sql)

        return {
            "sql_query": sql,
            "elapsed_sec": round(time.time() - t0, 2),
            "error": "",
        }

    except Exception as exc:
        return {
            "sql_query": "",
            "elapsed_sec": round(time.time() - t0, 2),
            "error": str(exc),
        }


# ═════════════════════════════════════════════════════════════════════════════
# SQL EXECUTION
# ═════════════════════════════════════════════════════════════════════════════

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|truncate|alter|create|grant|revoke)\b",
    re.IGNORECASE,
)


def _validate_sql(sql: str) -> None:
    if _FORBIDDEN.search(sql):
        raise ValueError(f"Unsafe SQL blocked: {sql[:200]}")


def _jsonify_row(row: list) -> list:
    out = []

    for v in row:
        if isinstance(v, (datetime, date)):
            out.append(v.isoformat())

        elif isinstance(v, decimal.Decimal):
            out.append(float(v))

        elif isinstance(v, int):
            out.append(int(v))

        elif isinstance(v, float):
            out.append(float(v))

        elif v is None or isinstance(v, (str, bool)):
            out.append(v)

        else:
            out.append(str(v))

    return out


@tool
def sql_execute_tool(sql_query: str) -> Dict[str, Any]:
    """Validate and execute a SQL SELECT query, returning rows."""
    t0 = time.time()

    try:
        _validate_sql(sql_query)

        conn = _open_connection()

        try:
            cur = conn.cursor()

            cur.execute(sql_query)

            columns = [desc[0] for desc in cur.description]

            rows = [_jsonify_row(list(row)) for row in cur.fetchall()]

        finally:
            conn.close()

        return {
            "sql_query": sql_query,
            "columns": columns,
            "rows": rows,
            "count": len(rows),
            "elapsed_sec": round(time.time() - t0, 2),
            "error": "",
        }

    except ValueError as exc:
        return {
            "sql_query": sql_query,
            "columns": [],
            "rows": [],
            "count": 0,
            "elapsed_sec": round(time.time() - t0, 2),
            "error": f"BLOCKED — {exc}",
        }

    except Exception as exc:
        return {
            "sql_query": sql_query,
            "columns": [],
            "rows": [],
            "count": 0,
            "elapsed_sec": round(time.time() - t0, 2),
            "error": str(exc),
        }


def warmup() -> None:
    get_sql_pipeline()
