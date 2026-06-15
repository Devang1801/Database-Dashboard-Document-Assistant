"""
gateway/llm_manager.py
───────────────────────
Shared LLM manager — loads the local Qwen 4B model ONCE and reuses
the same tokenizer / model / pipeline for both SQL and RAG generation.

Fixes vs previous version
──────────────────────────
• BitsAndBytesConfig: removed llm_int8_enable_fp32_cpu_offload (8-bit param,
  not valid for 4-bit); use bnb_4bit_compute_dtype kwarg correctly.
• AutoModelForCausalLM: changed deprecated torch_dtype= → dtype= kwarg.
• 4-bit fallback: if BNB raises an error during model load (e.g. version
  mismatch) we automatically retry with float16 / no quantization so the
  server always starts.
• warmup() exported so main.py startup event can call it once.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("gateway.llm_manager")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "qwen_4b")
SHARED_MODEL_PATH = os.getenv("LLM_MODEL_PATH", DEFAULT_MODEL_PATH)
USE_4BIT = os.getenv("USE_4BIT", "true").lower() == "true"

_shared_tokenizers: Dict[str, Any] = {}
_shared_models: Dict[str, Any] = {}
_shared_pipelines: Dict[str, Any] = {}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _normalize(path: Optional[str] = None) -> str:
    return str(Path(path or SHARED_MODEL_PATH).resolve())


def _get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception as exc:
        log.warning(f"torch unavailable, falling back to cpu: {exc}")
        return "cpu"


def get_shared_model_path() -> str:
    return _normalize(SHARED_MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────────────────────────────────────────


def get_shared_tokenizer(model_path: Optional[str] = None):
    key = _normalize(model_path)
    if key in _shared_tokenizers:
        return _shared_tokenizers[key]

    from transformers import AutoTokenizer
    log.info(f"Loading shared tokenizer from: {key}")
    tok = AutoTokenizer.from_pretrained(
        key, trust_remote_code=True, local_files_only=True
    )
    tok.pad_token = tok.eos_token
    _shared_tokenizers[key] = tok
    return tok


# ─────────────────────────────────────────────────────────────────────────────
# MODEL  (with 4-bit → fp16 fallback)
# ─────────────────────────────────────────────────────────────────────────────


def _load_with_4bit(key: str, device: str):
    """Try to load with 4-bit BNB quantization. Returns model or raises."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        # NOTE: llm_int8_enable_fp32_cpu_offload is an 8-bit param — do NOT
        # include it here; it causes Params4bit.__new__() keyword errors.
    )

    return AutoModelForCausalLM.from_pretrained(
        key,
        quantization_config=bnb,
        device_map="auto",
        dtype=torch.float16,          # ← use dtype= (torch_dtype is deprecated)
        trust_remote_code=True,
        local_files_only=True,
    )


def _load_without_quantization(key: str, device: str):
    """Fallback: load in fp16 (cuda) or fp32 (cpu) without BNB."""
    import torch
    from transformers import AutoModelForCausalLM

    log.warning("Loading model WITHOUT 4-bit quantization (BNB fallback path)")
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        key,
        device_map="auto" if device == "cuda" else None,
        dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    )
    if device == "cpu":
        model = model.to("cpu")
    return model


def get_shared_model(model_path: Optional[str] = None):
    key = _normalize(model_path)
    if key in _shared_models:
        return _shared_models[key]

    device = _get_device()
    log.info(f"Loading shared LLM from: {key}  device={device}")

    if USE_4BIT and device == "cuda":
        try:
            model = _load_with_4bit(key, device)
            log.info("✅ Model loaded with 4-bit quantization")
        except Exception as exc:
            log.warning(
                f"4-bit load failed ({exc!r}), retrying without quantization …"
            )
            model = _load_without_quantization(key, device)
    else:
        model = _load_without_quantization(key, device)

    model.eval()
    _shared_models[key] = model
    return model


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────


def get_shared_pipeline(model_path: Optional[str] = None):
    key = _normalize(model_path)
    if key in _shared_pipelines:
        return _shared_pipelines[key]

    from transformers import pipeline as hf_pipeline

    tok = get_shared_tokenizer(key)
    mdl = get_shared_model(key)

    pipe = hf_pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        return_full_text=False,
        pad_token_id=tok.eos_token_id,
    )
    log.info("✅ Shared text-generation pipeline ready")
    _shared_pipelines[key] = pipe
    return pipe


def is_shared_pipeline_ready(model_path: Optional[str] = None) -> bool:
    return _normalize(model_path) in _shared_pipelines


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP WARMUP
# ─────────────────────────────────────────────────────────────────────────────


def warmup() -> None:
    """
    Pre-load the shared tokenizer, model, and pipeline at startup so that
    the very first user request is not slow.
    Call this once from the FastAPI startup event.
    """
    get_shared_pipeline()
    log.info("🔥 LLM warmup complete")