"""
tools/rag.py
────────────
LangGraph tool: FAISS vector search + local Qwen 4B LLM for
policy / document Q&A (RAG — Retrieval-Augmented Generation).
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from gateway.llm_manager import get_shared_pipeline, get_shared_model_path
from langchain_core.tools import tool

log = logging.getLogger("agent.tool.rag")

# ── Paths & config ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_MODEL = str(os.getenv("EMBEDDING_MODEL_PATH", PROJECT_ROOT / "all-mini_v12"))
MODEL_PATH = os.getenv("RAG_MODEL_PATH", get_shared_model_path())
DOCS_DIR = str(os.getenv("DOCS_DIR", PROJECT_ROOT / "docs"))
VECTORSTORE_DIR = PROJECT_ROOT / "faiss_index"
DOC_SUMMARY_PATH = VECTORSTORE_DIR / "doc_summary.json"
USE_4BIT = os.getenv("USE_4BIT", "true").lower() == "true"

TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "6"))
TOP_K_DOCS = int(os.getenv("TOP_K_DOCS", "3"))

RAG_LLM_CONFIG = {
    "max_new_tokens": 256,
    "max_length": None,   
    "temperature": 0.1,
    "top_p": 0.85,
    "top_k": 40,
    "repetition_penalty": 2.15,
    "do_sample": True,
}

SYSTEM_PROMPT = (
    "You are a Stock Market analyst assistant.\n"
    "Your role is to answer questions about stock market basics, metrics, regulations, exchanges, trading, sector analysis, historical events, risk management, and investment concepts based on the provided documents.\n\n"
    "Guidelines:\n"
    "- Answer ONLY using the provided context documents.\n"
    "- Be concise, factual, and well-structured.\n"
    "- Cite the source document name wherever possible.\n"
    "- If the answer is NOT in the context, say: "
    "'The provided documents do not contain information on this topic.'\n"
    "- Never fabricate facts, numbers, or concepts.\n"
    "- Use bullet points when listing multiple items."
)

# ── Lazy singletons ────────────────────────────────────────────────────────────
_embeddings = None
_llm_pipeline = None
_vectorstore = None


def _get_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception as exc:
        log.warning(f"torch unavailable for RAG pipeline, falling back to cpu: {exc}")
        return "cpu"


def get_embeddings():
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings

    log.info(f"Loading embeddings from: {EMBEDDING_MODEL}")
    _embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": _get_device()},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
    log.info("Embedding model loaded")
    return _embeddings


def get_rag_pipeline():
    global _llm_pipeline
    if _llm_pipeline is not None:
        return _llm_pipeline

    _llm_pipeline = get_shared_pipeline(MODEL_PATH)
    return _llm_pipeline


def get_vectorstore():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    index_path = str(VECTORSTORE_DIR)

    if (VECTORSTORE_DIR / "index.faiss").exists():
        log.info(f"Loading existing FAISS index: {index_path}")
        _vectorstore = FAISS.load_local(
            index_path, get_embeddings(), allow_dangerous_deserialization=True
        )
        return _vectorstore

    log.info(f"Building FAISS index from docs: {DOCS_DIR}")
    os.makedirs(DOCS_DIR, exist_ok=True)

    docs_path = Path(DOCS_DIR)
    all_files = list(docs_path.rglob("*.txt")) + list(docs_path.rglob("*.md"))
    if all_files:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import TextLoader

        loaded_docs = []
        for file_path in all_files:
            try:
                loader = TextLoader(str(file_path), encoding="utf-8")
                loaded_docs.extend(loader.load())
            except Exception as e:
                log.warning(f"Error loading {file_path}: {e}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=60,
            separators=["\n\n", "\n", ".", " "],
        )
        docs = splitter.split_documents(loaded_docs)
        _save_doc_summary(docs)
    else:
        log.warning("No .txt or .md docs found — using placeholder index")
        docs = [
            Document(
                page_content=(
                    "Stock Market Complete Guide knowledge base. "
                    "Place .md or .txt documents in the ./docs folder and restart."
                ),
                metadata={"source": "placeholder"},
            )
        ]

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    _vectorstore = FAISS.from_documents(docs, get_embeddings())
    _vectorstore.save_local(index_path)
    log.info(f"FAISS saved ({len(docs)} chunks)")
    return _vectorstore


def _save_doc_summary(docs: List[Any]) -> None:
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, int] = {}
    for d in docs:
        src = d.metadata.get("source", "unknown")
        summary[src] = summary.get(src, 0) + 1
    with open(DOC_SUMMARY_PATH, "w") as fh:
        json.dump(summary, fh, indent=2)


def _unique_source_docs(hits: List[Any], top_k: int) -> List[Any]:
    seen, unique = set(), []
    for d in hits:
        src = d.metadata.get("source", "")
        if src not in seen:
            seen.add(src)
            unique.append(d)
        if len(unique) >= top_k:
            break
    return unique


def _build_prompt(
    query: str, context: str, memory: Optional[List[Dict[str, str]]] = None
) -> str:
    parts = [f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>"]
    for turn in (memory or [])[-2:]:
        role = "user" if turn["role"] == "user" else "assistant"
        parts.append(f"<|im_start|>{role}\n{turn['content'][:400]}<|im_end|>")
    user_block = f"Context Documents:\n{context}\n\nQuestion: {query}"
    parts.append(f"<|im_start|>user\n{user_block}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# ── Public tool ────────────────────────────────────────────────────────────────


@tool
def rag_tool(query: str, memory: str = "[]") -> Dict[str, Any]:
    """
    Answer a policy / document question using FAISS retrieval + Qwen 4B LLM.

    Use this for questions about regulations, policies, rules, procedures,
    guidelines, acts, directives, or any 'what is / explain / define' queries.

    Args:
        query:  The user's natural-language question.
        memory: JSON-serialised list of prior conversation turns
                (list[{"role": str, "content": str}]).

    Returns:
        dict with keys: answer, sources (list[dict]), chunks_used, elapsed_sec, error.
    """
    t0 = time.time()
    try:
        mem = json.loads(memory) if isinstance(memory, str) else memory
        vs = get_vectorstore()
        hits = vs.similarity_search(query, k=TOP_K_CHUNKS)

        context = "\n\n".join(
            f"[Doc {i + 1} — {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for i, d in enumerate(hits)
        )
        prompt = _build_prompt(query, context, mem)
        output = get_rag_pipeline()(prompt, **RAG_LLM_CONFIG)
        answer = output[0]["generated_text"].strip()

        log.info(f"RAG answered in {round(time.time()-t0, 2)}s, {len(hits)} chunks")
        return {
            "answer": answer,
            "sources": [d.metadata for d in _unique_source_docs(hits, TOP_K_DOCS)],
            "chunks_used": len(hits),
            "elapsed_sec": round(time.time() - t0, 2),
            "error": "",
        }

    except Exception as exc:
        log.error(f"rag_tool error: {exc}")
        return {
            "answer": "",
            "sources": [],
            "chunks_used": 0,
            "elapsed_sec": round(time.time() - t0, 2),
            "error": str(exc),
        }


def ingest_document(text: str, source: str = "runtime") -> Dict[str, str]:
    """Add a document chunk to the live FAISS index (not a @tool — called directly)."""
    from langchain_core.documents import Document

    doc = Document(page_content=text, metadata={"source": source})
    vs = get_vectorstore()
    vs.add_documents([doc])
    vs.save_local(str(VECTORSTORE_DIR))
    log.info(f"Document ingested: {source}")
    return {"status": "ok", "source": source}


def warmup() -> None:
    """Pre-load embeddings, vectorstore, and LLM at startup."""
    get_embeddings()
    get_vectorstore()
    get_rag_pipeline()
