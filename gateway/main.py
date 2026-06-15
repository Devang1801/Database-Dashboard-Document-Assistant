"""
main.py
───────
RakshaYantra — Single-agent gateway (v4.2).

Changes vs v4.1
───────────────
• Intent routing is now LLM-based, not keyword-based.
• Model is warmed up once in the FastAPI startup event (warmup_all()).
• node_build_response now uses the LLM to generate a real natural-language
  answer from SQL results (counts, lists, aggregations) instead of just
  printing "Found N records."  The table is still rendered in the frontend.

SQL pipeline (serial):
    db_connect_tool → sql_generate_tool → sql_execute_tool → chart_build_tool → LLM answer

RAG pipeline:
    rag_tool
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg2
from fastapi import Depends, FastAPI, HTTPException, Response, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import plotly.io as pio
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from typing_extensions import TypedDict

from gateway.auth import get_current_user
from tools import ALL_TOOLS, warmup_all
from tools.chart import _should_chart, chart_build_tool
from tools.rag import rag_tool
from tools.sql import db_connect_tool, sql_execute_tool, sql_generate_tool

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | main | %(message)s",
)
log = logging.getLogger("main")

# ── PostgreSQL config ─────────────────────────────────────────────────────────
DB_CONFIG: Dict[str, Any] = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
    "dbname": os.getenv("DB_NAME", "ids"),
    "port": int(os.getenv("DB_PORT", "5432")),
}

THREADS_TABLE = "public.chat_threads"
MESSAGES_TABLE = "public.chat_messages"
MEMORY_WINDOW = 6
TITLE_MAX_LEN = 120

# ─────────────────────────────────────────────────────────────────────────────
# LLM-BASED INTENT CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

_INTENT_SYSTEM = (
    "You are a query classifier for the Stock Market Analytics system.\n"
    "Your only job is to decide whether a user query should be answered by:\n"
    "  SQL  — fetching records/numbers/statistics from the stock market database\n"
    "  RAG  — looking up financial concepts, stock market regulations, historical events, trading rules, exchanges, or investment concepts in the complete guide\n\n"
    "Rules:\n"
    "- If the query asks for data, records, counts, stock prices, volume, ticker names,\n"
    "  dates, PE ratios, or any numerical or historical database analysis → respond SQL\n"
    "- If the query asks what a rule/concept/definition/regulation says,\n"
    "  or asks to explain a general topic in stock market or a chapter in the guide → respond RAG\n"
    "- Respond with EXACTLY one word: SQL   or   RAG\n"
    "- Never explain your choice."
)

_INTENT_CLASSIFY_CONFIG = {
    "max_new_tokens": 4,
    "max_length": None,
    "temperature": 0.0,
    "top_k": 1,
    "do_sample": False,
}


def classify_intent(query: str) -> str:
    """
    Ask the shared Qwen model to classify the query as SQL or RAG.
    Falls back to simple heuristics if the LLM is not ready yet or fails.
    """
    from gateway.llm_manager import get_shared_pipeline, is_shared_pipeline_ready, get_shared_model_path

    model_path = get_shared_model_path()

    if not is_shared_pipeline_ready(model_path):
        # Model still loading — safe fallback (heuristic)
        log.warning("LLM not ready for intent classification, using heuristic fallback")
        return _heuristic_intent(query)

    prompt = (
        f"<|im_start|>system\n{_INTENT_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    try:
        pipe = get_shared_pipeline(model_path)
        out = pipe(prompt, **_INTENT_CLASSIFY_CONFIG)
        raw = out[0]["generated_text"].strip().upper()
        intent = "RAG" if raw.startswith("RAG") else "SQL"
        log.info(f"🤖 LLM classified intent: {intent!r}  (raw={raw!r})")
        return intent
    except Exception as exc:
        log.warning(f"Intent LLM call failed ({exc}), using heuristic fallback")
        return _heuristic_intent(query)


_RAG_HINTS = {
    "policy", "rule", "guideline", "procedure", "regulation",
    "act", "law", "directive", "explain", "what is", "define",
    "how does", "what does", "meaning of", "concept", "chapter",
    "guide", "history", "psychology", "strategy", "strategies",
    "event", "case study", "case studies",
}


def _heuristic_intent(query: str) -> str:
    q = query.lower()
    return "RAG" if any(kw in q for kw in _RAG_HINTS) else "SQL"


# ─────────────────────────────────────────────────────────────────────────────
# AGENT STATE
# ─────────────────────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    query: str
    user_id: str
    thread_id: str

    messages: List[BaseMessage]
    memory: List[Dict[str, str]]

    intent: str          # "SQL" | "RAG"

    db_verified: bool
    sql_generated: bool
    sql_executed: bool

    sql_query: str
    columns: List[str]
    rows: List[List[Any]]

    do_chart: bool
    chart_json: Optional[str]
    chart_jsons: List[str]
    
    rag_answer: str
    rag_sources: List[Dict]

    response: str
    error: str


def make_state(query: str, user_id: str, thread_id: str) -> AgentState:
    return AgentState(
        query=query, user_id=user_id, thread_id=thread_id,
        messages=[], memory=[], intent="",
        db_verified=False, sql_generated=False, sql_executed=False,
        sql_query="", columns=[], rows=[],
        do_chart=False, chart_json=None, chart_jsons=[],
        rag_answer="", rag_sources=[],
        response="", error="",
    )


# ─────────────────────────────────────────────────────────────────────────────
# DB HELPERS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────


def _get_conn():
    return psycopg2.connect(**DB_CONFIG)


def ensure_chat_tables() -> None:
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {THREADS_TABLE} (
        thread_id  VARCHAR      PRIMARY KEY,
        user_id    VARCHAR      NOT NULL,
        title      VARCHAR(300) DEFAULT '',
        created_at TIMESTAMP    DEFAULT NOW(),
        updated_at TIMESTAMP    DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_threads_user
        ON {THREADS_TABLE}(user_id, updated_at DESC);

    CREATE TABLE IF NOT EXISTS {MESSAGES_TABLE} (
        message_id   SERIAL      PRIMARY KEY,
        thread_id    VARCHAR     NOT NULL REFERENCES {THREADS_TABLE}(thread_id),
        user_id      VARCHAR     NOT NULL,
        role         VARCHAR(20) NOT NULL,
        content      TEXT,
        sql_query    TEXT,
        intent       VARCHAR(20),
        chart_json   TEXT,
        columns_json TEXT,
        rows_json    TEXT,
        created_at   TIMESTAMP   DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_messages_user_created
        ON {MESSAGES_TABLE}(user_id, created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_messages_thread
        ON {MESSAGES_TABLE}(thread_id, created_at ASC);
    """
    # Migration: add columns_json / rows_json if they don't exist yet
    migrate_ddl = f"""
    DO $$ BEGIN
        ALTER TABLE {MESSAGES_TABLE} ADD COLUMN IF NOT EXISTS columns_json TEXT;
        ALTER TABLE {MESSAGES_TABLE} ADD COLUMN IF NOT EXISTS rows_json TEXT;
    EXCEPTION WHEN others THEN NULL;
    END $$;
    """
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(ddl)
        cur.execute(migrate_ddl)
        conn.commit()
        conn.close()
        log.info("✅ Chat tables ready")
    except Exception as exc:
        log.warning(f"ensure_chat_tables (non-fatal): {exc}")


def create_thread(user_id: str) -> Dict[str, str]:
    thread_id = str(uuid.uuid4())
    try:
        conn = _get_conn()
        conn.cursor().execute(
            f"INSERT INTO {THREADS_TABLE}(thread_id, user_id) VALUES (%s, %s)",
            (thread_id, user_id),
        )
        conn.commit()
        conn.close()
        log.info(f"🆕 Thread created: {thread_id} for user={user_id}")
    except Exception as exc:
        log.error(f"create_thread failed: {exc}")
        raise
    return {"thread_id": thread_id}


def _set_thread_title(thread_id: str, first_message: str) -> None:
    title = first_message[:TITLE_MAX_LEN].strip()
    try:
        conn = _get_conn()
        conn.cursor().execute(
            f"UPDATE {THREADS_TABLE} SET title = %s WHERE thread_id = %s AND (title IS NULL OR title = '')",
            (title, thread_id),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        log.warning(f"_set_thread_title (non-fatal): {exc}")


def _touch_thread(thread_id: str) -> None:
    try:
        conn = _get_conn()
        conn.cursor().execute(
            f"UPDATE {THREADS_TABLE} SET updated_at = NOW() WHERE thread_id = %s",
            (thread_id,),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        log.warning(f"_touch_thread (non-fatal): {exc}")


def get_user_threads(user_id: str) -> List[Dict[str, Any]]:
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            f"SELECT thread_id, title, created_at, updated_at FROM {THREADS_TABLE} "
            f"WHERE user_id = %s ORDER BY updated_at DESC",
            (user_id,),
        )
        rows = cur.fetchall()
        conn.close()
        return [
            {
                "thread_id": r[0],
                "title": r[1] or "New conversation",
                "created_at": r[2].isoformat() if r[2] else None,
                "updated_at": r[3].isoformat() if r[3] else None,
            }
            for r in rows
        ]
    except Exception as exc:
        log.warning(f"get_user_threads (non-fatal): {exc}")
        return []


def _validate_thread_owner(thread_id: str, user_id: str) -> None:
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            f"SELECT user_id FROM {THREADS_TABLE} WHERE thread_id = %s",
            (thread_id,),
        )
        row = cur.fetchone()
        conn.close()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"DB error: {exc}")

    if row is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    if row[0] != user_id:
        raise HTTPException(status_code=403, detail="Thread belongs to another user")


def load_memory(user_id: str) -> List[Dict[str, str]]:
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            f"SELECT role, content FROM {MESSAGES_TABLE} "
            f"WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
            (user_id, MEMORY_WINDOW),
        )
        rows = cur.fetchall()
        conn.close()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
    except Exception as exc:
        log.warning(f"load_memory (non-fatal): {exc}")
        return []


def load_thread_history(thread_id: str) -> List[Dict[str, Any]]:
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            f"SELECT message_id, role, content, intent, sql_query, chart_json, created_at, columns_json, rows_json "
            f"FROM {MESSAGES_TABLE} WHERE thread_id = %s ORDER BY created_at ASC",
            (thread_id,),
        )
        rows = cur.fetchall()
        conn.close()
        return [
            {
                "message_id": r[0], "role": r[1], "content": r[2], "intent": r[3],
                "sql_query": r[4], "chart_json": r[5],
                "created_at": r[6].isoformat() if r[6] else None,
                "columns": json.loads(r[7]) if r[7] else None,
                "rows":    json.loads(r[8]) if r[8] else None,
            }
            for r in rows
        ]
    except Exception as exc:
        log.warning(f"load_thread_history (non-fatal): {exc}")
        return []


def persist_turn(state: AgentState) -> None:
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            f"INSERT INTO {MESSAGES_TABLE} (thread_id, user_id, role, content, intent, sql_query) "
            f"VALUES (%s, %s, 'user', %s, %s, %s)",
            (state["thread_id"], state["user_id"], state["query"],
             state.get("intent", ""), state.get("sql_query", "")),
        )
        chart_str   = json.dumps(state["chart_jsons"]) if state.get("chart_jsons") else None
        columns_str = json.dumps(state["columns"]) if state.get("columns") else None
        rows_str    = json.dumps(state["rows"])    if state.get("rows")    else None
        cur.execute(
            f"INSERT INTO {MESSAGES_TABLE} "
            f"(thread_id, user_id, role, content, intent, sql_query, chart_json, columns_json, rows_json) "
            f"VALUES (%s, %s, 'assistant', %s, %s, %s, %s, %s, %s)",
            (state["thread_id"], state["user_id"], state.get("response", ""),
             state.get("intent", ""), state.get("sql_query", ""),
             chart_str, columns_str, rows_str),
        )
        conn.commit()
        conn.close()
        _set_thread_title(state["thread_id"], state["query"])
        _touch_thread(state["thread_id"])
        log.info(f"💾 Turn persisted — thread={state['thread_id']} user={state['user_id']}")
    except Exception as exc:
        log.warning(f"persist_turn (non-fatal): {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# AGENT NODE  — serial, LLM-routed
# ─────────────────────────────────────────────────────────────────────────────


def _make_tool_call(name: str, args: Dict) -> Dict:
    return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}


def node_agent(state: AgentState) -> AgentState:
    """
    Serial tool scheduler with LLM-based intent routing.

    On each call the node looks at what has already been done and schedules
    exactly the NEXT tool in the pipeline.  The LangGraph edge will loop
    back here after each tool finishes.

    Pipeline order
    ──────────────
    SQL intent:  db_connect → sql_generate → sql_execute → [chart_build] → done
    RAG intent:  rag_tool → done
    """
    query = state["query"]
    memory_json = json.dumps(state.get("memory", []))

    # ── Step 1: classify intent (once per conversation turn) ─────────────────
    if not state.get("intent"):
        intent = classify_intent(query)
        state["intent"] = intent
        log.info(f"🎯 Intent: {intent}")

    # ════════════════════════════════════════
    # RAG PIPELINE
    # ════════════════════════════════════════
    if state["intent"] == "RAG":
        if not state.get("rag_answer") and not state.get("error"):
            log.info("📚 Step RAG-1: rag_tool")
            state["messages"] = state.get("messages", []) + [
                AIMessage(content="", tool_calls=[
                    _make_tool_call(rag_tool.name, {"query": query, "memory": memory_json})
                ])
            ]
            return state

        # RAG done → emit __done__ so the graph exits to build_response
        state["messages"] = state.get("messages", []) + [
            AIMessage(content="__done__", tool_calls=[])
        ]
        return state

    # ════════════════════════════════════════
    # SQL PIPELINE  (strictly serial)
    # ════════════════════════════════════════

    # Step SQL-1: verify DB connectivity
    if not state.get("db_verified"):
        log.info("🔌 Step SQL-1: db_connect_tool")
        state["messages"] = state.get("messages", []) + [
            AIMessage(content="", tool_calls=[_make_tool_call(db_connect_tool.name, {})])
        ]
        return state

    # Step SQL-2: generate SQL (only after DB is confirmed live)
    if not state.get("sql_generated"):
        log.info("📝 Step SQL-2: sql_generate_tool")
        state["messages"] = state.get("messages", []) + [
            AIMessage(content="", tool_calls=[
                _make_tool_call(sql_generate_tool.name, {"query": query, "memory": memory_json})
            ])
        ]
        return state

    # Step SQL-3: execute SQL (only after query is generated)
    if not state.get("sql_executed"):
        log.info("▶️  Step SQL-3: sql_execute_tool")
        state["messages"] = state.get("messages", []) + [
            AIMessage(content="", tool_calls=[
                _make_tool_call(sql_execute_tool.name, {"sql_query": state["sql_query"]})
            ])
        ]
        return state

    # Step SQL-4: chart (optional, only after execution, only once)
    if state.get("rows") and not state.get("do_chart") and not state.get("chart_json"):
        should = _should_chart(state["columns"], state["rows"], query)
        state["do_chart"] = True   # mark as "chart decision made" regardless
        log.info(f"📊 Chart decision: {should} (rows={len(state['rows'])}, cols={len(state['columns'])})")
        if should:
            log.info("📊 Step SQL-4: chart_build_tool")
            state["messages"] = state.get("messages", []) + [
                AIMessage(content="", tool_calls=[
                    _make_tool_call(chart_build_tool.name, {
                        "columns": state["columns"],
                        "rows": state["rows"],
                        "query": query,
                    })
                ])
            ]
            return state

    # All steps done — signal completion
    state["messages"] = state.get("messages", []) + [
        AIMessage(content="__done__", tool_calls=[])
    ]
    return state


# ─────────────────────────────────────────────────────────────────────────────
# TOOL EXECUTOR NODE
# ─────────────────────────────────────────────────────────────────────────────

_tool_node = ToolNode(tools=ALL_TOOLS)


def node_tool_executor(state: AgentState) -> AgentState:
    updated = _tool_node.invoke(state)
    state["messages"] = updated.get("messages", state.get("messages", []))

    for msg in reversed(state["messages"]):
        if not isinstance(msg, ToolMessage):
            continue
        try:
            result = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            result = {}

        tool_name = msg.name

        if tool_name == db_connect_tool.name:
            if result.get("connected"):
                state["db_verified"] = True
            else:
                state["error"] = f"DB unreachable: {result.get('error', '')}"
                state["db_verified"] = state["sql_generated"] = state["sql_executed"] = True

        elif tool_name == sql_generate_tool.name:
            if result.get("error"):
                state["error"] = result["error"]
                state["sql_generated"] = state["sql_executed"] = True
            else:
                state["sql_query"] = result.get("sql_query", "")
                state["sql_generated"] = True

        elif tool_name == sql_execute_tool.name:
            state["sql_executed"] = True
            if result.get("error"):
                state["error"] = result["error"]
            else:
                state["columns"] = result.get("columns", [])
                state["rows"] = result.get("rows", [])

        elif tool_name == chart_build_tool.name:
            if not result.get("error"):
                state["chart_jsons"] = result.get("chart_jsons", [])
                state["chart_json"]  = result.get("chart_json") or (state["chart_jsons"][0] if state["chart_jsons"] else None)
                log.info(f"✅ Charts extracted: {len(state['chart_jsons'])} chart(s)")
            else:
                log.warning(f"chart_build_tool error: {result.get('error')}")

        elif tool_name == rag_tool.name:
            if result.get("error"):
                state["error"] = result["error"]
            else:
                state["rag_answer"] = result.get("answer", "")
                state["rag_sources"] = result.get("sources", [])

        break  # only process the most recent ToolMessage

    return state


# ─────────────────────────────────────────────────────────────────────────────
# OTHER NODES
# ─────────────────────────────────────────────────────────────────────────────


def node_load_memory(state: AgentState) -> AgentState:
    state["memory"] = load_memory(state["user_id"])
    log.info(f"🧠 Memory: {len(state['memory'])} messages for user={state['user_id']}")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# LLM-BASED SQL ANSWER GENERATION
# ─────────────────────────────────────────────────────────────────────────────

_ANSWER_SYSTEM = (
    "You are a Stock Market analyst assistant.\n"
    "You are given a user's question and SQL query results.\n"
    "Write concise analytical answers using exact values.\n"
    "Examples:\n"
    "Q: Top company by market cap?\n"
    "A: Apple has the highest market capitalization.\n"
    "Q: Average close price in Technology sector?\n"
    "A: The average close price in the Technology sector is 178.45.\n"
    "Q: Highest volume stocks?\n"
    "A: These are the stocks with the highest trading volume.\n"
)
_ANSWER_GEN_CONFIG = {
    "max_new_tokens": 120,
    "temperature": 0.1,
    "max_length": None,
    "top_k": 10,
    "do_sample": False,
    "repetition_penalty": 1.1,

}


def _generate_sql_answer(
    query: str,
    columns: List[str],
    rows: List[List[Any]],
    has_chart: bool,
) -> str:
    """
    Use the shared LLM to produce a natural-language answer from SQL results.
    Falls back to a formatted template if the LLM is not available.
    """
    from gateway.llm_manager import (
        get_shared_model_path,
        get_shared_pipeline,
        is_shared_pipeline_ready,
    )

    # ── Build a compact data preview (max 10 rows to stay within context) ─────
    preview = rows[:10]
    header = " | ".join(str(c) for c in columns)
    body = "\n".join(" | ".join(str(v) for v in row) for row in preview)
    suffix = f"\n… and {len(rows) - 10} more rows" if len(rows) > 10 else ""
    data_block = f"{header}\n{body}{suffix}"

    prompt = (
        f"<|im_start|>system\n{_ANSWER_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Question: {query}\n\n"
        f"SQL Result:\n{data_block}\n\n"
        f"Answer the question in 1–2 sentences using the exact numbers:<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    model_path = get_shared_model_path()

    if is_shared_pipeline_ready(model_path):
        try:
            pipe = get_shared_pipeline(model_path)
            out = pipe(prompt, **_ANSWER_GEN_CONFIG)
            answer = out[0]["generated_text"].strip()
            # Strip any stop-token leakage
            answer = answer.split("<|im_end|>")[0].split("<|im_start|>")[0].strip()
            if answer:
                if has_chart:
                    answer += "\n\n"
                log.info(f"✍️  LLM answer generated ({len(answer)} chars)")
                return answer
        except Exception as exc:
            log.warning(f"_generate_sql_answer LLM call failed: {exc}")

    # ── Fallback: hand-crafted summary ───────────────────────────────────────
    if len(rows) == 1 and len(columns) == 1:
        col, val = columns[0].lower(), rows[0][0]
        return f"**{columns[0]}**: **{val}**"

    tail = " A chart has been generated." if has_chart else ""
    return f"Retrieved **{len(rows)}** stock market record(s).{tail}"


def node_build_response(state: AgentState) -> AgentState:
    if state.get("error"):
        state["response"] = f"⚠️ {state['error']}"
        return state

    if state["intent"] == "RAG":
        state["response"] = (
            state.get("rag_answer") or "No answer found in policy documents."
        )
        return state

    rows = state.get("rows", [])
    if not rows:
        state["response"] = "No records found matching your query."
        return state

    # Use the LLM to produce a proper natural-language answer.
    # The raw table is still sent in `columns` / `rows` and rendered as a
    # table in the frontend — this text is what appears in the chat bubble.
    state["response"] = _generate_sql_answer(
        query=state["query"],
        columns=state["columns"],
        rows=rows,
        has_chart=bool(state.get("chart_json")),
    )
    return state


def node_save_memory(state: AgentState) -> AgentState:
    persist_turn(state)
    return state


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING & GRAPH
# ─────────────────────────────────────────────────────────────────────────────


def _should_continue(state: AgentState) -> str:
    last = state["messages"][-1] if state.get("messages") else None
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "build_response"


builder = StateGraph(AgentState)
builder.add_node("load_memory", node_load_memory)
builder.add_node("agent", node_agent)
builder.add_node("tools", node_tool_executor)
builder.add_node("build_response", node_build_response)
builder.add_node("save_memory", node_save_memory)

builder.set_entry_point("load_memory")
builder.add_edge("load_memory", "agent")
builder.add_conditional_edges(
    "agent", _should_continue,
    {"tools": "tools", "build_response": "build_response"},
)
builder.add_edge("tools", "agent")
builder.add_edge("build_response", "save_memory")
builder.add_edge("save_memory", END)

agent_graph = builder.compile()


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Stock Market Analytics Gateway", version="4.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

INDEX_FILE = Path(__file__).resolve().parent.parent / "index.html"


@app.get("/", response_class=FileResponse)
async def root() -> FileResponse:
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(str(INDEX_FILE))


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "version": "4.2.0"}


@app.get("/model_status")
async def model_status() -> Dict[str, Any]:
    from gateway.llm_manager import get_shared_model_path, is_shared_pipeline_ready
    model_path = get_shared_model_path()
    ready = is_shared_pipeline_ready(model_path)
    return {
        "ready": ready,
        "message": "AI model loaded and ready" if ready else "Loading AI model…",
    }


# ── Thread endpoints (unchanged) ──────────────────────────────────────────────

class NewThreadResponse(BaseModel):
    thread_id: str


@app.post("/threads/new", response_model=NewThreadResponse)
def new_thread(user_id: str = Depends(get_current_user)) -> NewThreadResponse:
    result = create_thread(user_id)
    return NewThreadResponse(thread_id=result["thread_id"])


class ThreadSummary(BaseModel):
    thread_id: str
    title: str
    created_at: Optional[str]
    updated_at: Optional[str]


@app.get("/threads", response_model=List[ThreadSummary])
def list_threads(user_id: str = Depends(get_current_user)) -> List[ThreadSummary]:
    return [ThreadSummary(**t) for t in get_user_threads(user_id)]


@app.get("/threads/{thread_id}/history")
async def get_history(thread_id: str, user: str = Depends(get_current_user)):
    return {"messages": load_thread_history(thread_id)}


@app.delete("/threads/{thread_id}/truncate/{message_id}")
async def truncate_thread(thread_id: str, message_id: int, user: str = Depends(get_current_user)):
    """Delete all messages in a thread that were created AT or AFTER a specific message_id."""
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            f"DELETE FROM {MESSAGES_TABLE} WHERE thread_id = %s AND message_id >= %s",
            (thread_id, message_id)
        )
        conn.commit()
        conn.close()
        return {"status": "truncated", "deleted_from": message_id}
    except Exception as exc:
        log.error(f"truncate_thread failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    thread_id: str


class ChatResponse(BaseModel):
    thread_id: str
    user_id: str
    intent: str
    response: str
    sql_query: Optional[str] = None
    columns: Optional[List[str]] = None
    rows: Optional[List[List]] = None
    chart_json: Optional[str] = None
    chart_jsons: Optional[List[str]] = None
    rag_sources: Optional[List[Dict]] = None
    error: Optional[str] = None


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, user_id: str = Depends(get_current_user)) -> ChatResponse:
    _validate_thread_owner(req.thread_id, user_id)
    state = make_state(req.query, user_id, req.thread_id)
    try:
        result = agent_graph.invoke(state)
    except Exception as exc:
        log.error(f"/chat graph error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(
        thread_id=result["thread_id"],
        user_id=result["user_id"],
        intent=result.get("intent", ""),
        response=result.get("response", ""),
        sql_query=result.get("sql_query") or None,
        columns=result.get("columns") or None,
        rows=result.get("rows") or None,
        chart_json=result.get("chart_json") or None,
        chart_jsons=result.get("chart_jsons") or None,
        rag_sources=result.get("rag_sources") or None,
        error=result.get("error") or None,
    )


@app.post("/ingest")
def ingest(source: str, text: str, user_id: str = Depends(get_current_user)) -> Dict:
    from tools.rag import ingest_document
    return ingest_document(text, source)


# ─────────────────────────────────────────────────────────────────────────────
# CHART EXPORT & TEMP FILE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

# Store temp directories per user for cleanup
_TEMP_DIRS: Dict[str, str] = {}


@app.post("/charts/export")
def export_charts(
    body: Dict[str, Any] = Body(...),
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Save charts from JSON to temp PNG files.
    Returns a dict mapping chart JSON to temp file path.
    """
    try:
        thread_id = body.get("thread_id")
        chart_json_list = body.get("chart_json_list", [])

        if not isinstance(chart_json_list, list):
            raise HTTPException(status_code=422, detail="chart_json_list must be an array")

        # Convert any non-string chart entries to JSON strings
        normalized_charts: List[str] = []
        for chart in chart_json_list:
            if isinstance(chart, str):
                normalized_charts.append(chart)
            else:
                normalized_charts.append(json.dumps(chart))

        chart_json_list = normalized_charts
        # Create temp directory for this user's charts
        if user_id not in _TEMP_DIRS:
            temp_dir = tempfile.mkdtemp(prefix=f"stock_{user_id}_charts_")
            _TEMP_DIRS[user_id] = temp_dir
        else:
            temp_dir = _TEMP_DIRS[user_id]
            # Clean up old files in this directory
            for f in Path(temp_dir).glob("*.png"):
                try:
                    f.unlink()
                except Exception:
                    pass
        
        file_paths = {}
        
        for i, chart_json in enumerate(chart_json_list):
            if not chart_json:
                continue
            
            try:
                # Parse the JSON
                chart_dict = json.loads(chart_json) if isinstance(chart_json, str) else chart_json
                
                # Generate filename
                chart_hash = str(hash(chart_json))[:16].replace("-", "")
                output_path = os.path.join(temp_dir, f"chart_{i}_{chart_hash}.png")
                
                # Use plotly to convert to static image (requires kaleido)
                try:
                    import plotly.graph_objects as go
                    fig = go.Figure(chart_dict)
                    fig.write_image(output_path, width=1000, height=600)
                    file_paths[str(i)] = output_path
                    log.info(f"Chart {i} saved to {output_path}")
                except ImportError:
                    # Fallback: if kaleido not available, just save the JSON reference
                    log.warning("kaleido not installed, using JSON fallback for chart export")
                    json_path = os.path.join(temp_dir, f"chart_{i}_{chart_hash}.json")
                    with open(json_path, "w") as f:
                        json.dump(chart_dict, f)
                    file_paths[str(i)] = json_path
                
            except Exception as e:
                log.error(f"Error saving chart {i}: {e}")
                continue
        
        return {
            "success": True,
            "file_paths": file_paths,
            "temp_dir": temp_dir
        }
    
    except Exception as e:
        log.error(f"Error in export_charts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/charts/cleanup")
def cleanup_temp_charts(user_id: str = Depends(get_current_user)) -> Dict[str, str]:
    """
    Remove temporary chart files for the current user.
    
    Returns:
        {"status": "success", "message": "Cleanup completed"}
    """
    try:
        if user_id in _TEMP_DIRS:
            temp_dir = _TEMP_DIRS[user_id]
            try:
                shutil.rmtree(temp_dir)
                log.info(f"Cleaned up temp charts dir: {temp_dir}")
            except Exception as e:
                log.warning(f"Error removing temp dir {temp_dir}: {e}")
            finally:
                del _TEMP_DIRS[user_id]
        
        return {"status": "success", "message": "Cleanup completed"}
    
    except Exception as e:
        log.error(f"Error in cleanup_temp_charts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/charts/file/{filename}")
def get_temp_chart_file(filename: str, user_id: str = Depends(get_current_user)):
    """
    Retrieve a temporarily stored chart file.
    """
    if user_id not in _TEMP_DIRS:
        raise HTTPException(status_code=404, detail="No temp files for this user")
    
    temp_dir = _TEMP_DIRS[user_id]
    file_path = os.path.join(temp_dir, filename)
    
    # Security: ensure the file is within the temp directory
    real_path = os.path.realpath(file_path)
    real_temp = os.path.realpath(temp_dir)
    if not real_path.startswith(real_temp):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, media_type="image/png")


@app.on_event("startup")
def startup() -> None:
    ensure_chat_tables()
    # ── Warm up: load tokenizer + model + pipeline ONCE at startup ────────────
    # This means the first user request will never trigger a cold load.
    # warmup_all() calls get_shared_pipeline() which internally calls
    # get_shared_model() → get_shared_tokenizer(), so everything is
    # loaded in a single sequential pass.
    log.info("🔥 Starting LLM warmup (this takes ~30 s on first run)…")
    try:
        warmup_all()
        log.info("🔥 LLM warmup complete — server is ready")
    except Exception as exc:
        log.error(f"LLM warmup failed: {exc}", exc_info=True)
        # Don't crash the server; the first request will trigger load lazily


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
