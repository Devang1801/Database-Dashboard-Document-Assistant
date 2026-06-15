"""
gateway/config.py
────────────────
Configuration for the RakshaYantra gateway.

Environment variables control behavior at runtime.
"""

import os
from typing import Literal

# ── Context Memory Configuration ─────────────────────────────────────────────

# Enable context-based memory augmentation for related questions
# Set to "true" or "false"
CONTEXT_MEMORY_ENABLED = os.getenv("CONTEXT_MEMORY_ENABLED", "true").lower() == "true"

# Use LLM-based relation detection (more accurate but slower)
# Falls back to heuristic-based if set to "false"
CONTEXT_MEMORY_USE_LLM = os.getenv("CONTEXT_MEMORY_USE_LLM", "false").lower() == "true"

# Maximum number of previous messages to consider for context
CONTEXT_MEMORY_WINDOW = int(os.getenv("CONTEXT_MEMORY_WINDOW", "3"))

# Minimum overlap ratio for heuristic-based relation detection (0.0 to 1.0)
# Higher = stricter matching
CONTEXT_MEMORY_OVERLAP_THRESHOLD = float(
    os.getenv("CONTEXT_MEMORY_OVERLAP_THRESHOLD", "0.3")
)

# Include context in SQL/RAG generation system prompts
CONTEXT_IN_SYSTEM_PROMPT = os.getenv("CONTEXT_IN_SYSTEM_PROMPT", "true").lower() == "true"

# ── Logging ──────────────────────────────────────────────────────────────────

# Log verbosity for context memory module
CONTEXT_MEMORY_DEBUG = os.getenv("CONTEXT_MEMORY_DEBUG", "false").lower() == "true"


def get_context_config() -> dict:
    """
    Get current context memory configuration.
    
    Returns:
        Dictionary with all context memory settings
    """
    return {
        "enabled": CONTEXT_MEMORY_ENABLED,
        "use_llm": CONTEXT_MEMORY_USE_LLM,
        "memory_window": CONTEXT_MEMORY_WINDOW,
        "overlap_threshold": CONTEXT_MEMORY_OVERLAP_THRESHOLD,
        "include_in_system_prompt": CONTEXT_IN_SYSTEM_PROMPT,
        "debug": CONTEXT_MEMORY_DEBUG,
    }
