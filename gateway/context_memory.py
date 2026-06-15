"""
gateway/context_memory.py
─────────────────────────
Context-aware memory manager for related question detection and response augmentation.

Features:
• Detects semantic relationship between current and previous queries
• Extracts and maintains context from related exchanges
• Augments prompts with relevant historical context
• Supports both LLM-based and similarity-based detection
"""

import logging
from typing import Optional, List, Dict, Any
import json

log = logging.getLogger("gateway.context_memory")


class ContextMemoryManager:
    """Manages context-aware memory for related questions."""

    def __init__(self, use_llm: bool = True):
        """
        Initialize the context memory manager.
        
        Args:
            use_llm: If True, use LLM for relation detection. If False, use similarity-based.
        """
        self.use_llm = use_llm

    def is_question_related(self, current_query: str, previous_query: str) -> bool:
        """
        Detect if current query is related to previous query.
        
        Uses LLM classification if available, falls back to keyword/similarity heuristics.
        
        Args:
            current_query: The new user query
            previous_query: The previous user query
            
        Returns:
            True if queries are related, False otherwise
        """
        if not previous_query:
            return False

        current_lower = current_query.lower()
        previous_lower = previous_query.lower()

        # Domain context terms that indicate we're in the stock market or procurement domain
        domain_terms = {
            "stock", "stocks", "market", "markets", "ticker", "company", "companies",
            "price", "prices", "volume", "close", "open", "high", "low", "adjusted",
            "cap", "marketcap", "pe", "ratio", "dividend", "yield", "sector", "exchange",
            "country", "currency", "trade", "trading", "return", "returns",
            "share", "shares", "investor", "investors", "regulation", "regulations",
            "contract", "contracts", "proposal", "proposals", "vendor", "vendors",
            "navy", "army", "air", "force", "defence", "defense", "procurement",
            "cost", "costs", "budget", "funding", "approved", "approval", "approve",
            "department", "departments", "agency", "service", "services",
            "bid", "bids", "tender", "tenders", "supplier", "suppliers",
            "payment", "invoice", "expenses", "fy", "q1", "q2", "q3", "q4"
        }

        # Check if both queries are in the domain
        current_domain_terms = set(current_lower.split()) & domain_terms
        previous_domain_terms = set(previous_lower.split()) & domain_terms

        # Generic off-domain question detection - STRICT CHECK
        off_domain_keywords = {"weather", "time", "joke", "music", "movie"}
        if any(keyword in current_lower for keyword in off_domain_keywords) and not current_domain_terms:
            log.info(f"📎 Related query: NO (off-domain query detected)")
            return False

        # Check for pronouns or continuation indicators first (high confidence)
        pronouns = {"this", "that", "it", "they", "those", "these", "them"}
        if pronouns & set(current_lower.split()):
            log.info("📎 Related query detected (contains pronouns indicating continuation)")
            return True

        # Check for continuation words
        continuation_words = {
            "also", "additionally", "furthermore", "moreover", "besides",
            "next", "then", "after", "later", "meanwhile", "while",
            "break", "breakdown", "details", "detail", "more", "same",
            "about", "regarding", "concerning", "show", "list"
        }
        for cont_word in continuation_words:
            if current_lower.startswith(cont_word) or f" {cont_word}" in current_lower:
                # If previous is in domain, likely related
                if previous_domain_terms:
                    log.info(f"📎 Related query detected (starts with continuation word: {cont_word})")
                    return True

        # Token-based overlap analysis
        current_tokens = set(current_lower.split())
        previous_tokens = set(previous_lower.split())

        # Common words to ignore (stop words)
        stop_words = {
            "the", "a", "an", "and", "or", "is", "are", "was", "were",
            "what", "which", "how", "why", "when", "where", "who",
            "show", "get", "find", "list", "tell", "give", "provide",
            "please", "can", "could", "would", "should", "be", "been",
            "about", "of", "for", "to", "from", "in", "on", "at", "by",
            "me", "you", "he", "she", "we", "they", "any", "all", "some",
            "there", "many", "more", "with", "that", "this", "as", "do", "does", "did",
            "have", "has", "had", "these", "those", "then", "it", "just", "your"
        }

        # Extract meaningful tokens
        current_meaningful = current_tokens - stop_words
        previous_meaningful = previous_tokens - stop_words

        # Calculate overlap for longer queries
        if current_meaningful and previous_meaningful:
            overlap = current_meaningful & previous_meaningful
            min_set = min(len(current_meaningful), len(previous_meaningful))
            overlap_ratio = len(overlap) / min_set if min_set > 0 else 0

            # If overlap > 15%, consider related
            if overlap_ratio > 0.15:
                log.info(f"📎 Related query detected (overlap={overlap_ratio:.2%}, tokens: {overlap})")
                return True
            
            # Domain term overlap check - only if previous had domain terms
            if previous_domain_terms:
                domain_overlap = (current_meaningful & previous_meaningful) & domain_terms
                if domain_overlap:
                    log.info(f"📎 Related query detected (domain term overlap: {domain_overlap})")
                    return True

        # Check if it's a simple follow-up to a domain query
        # "Show me X" followed by "What about Y?" pattern
        if previous_domain_terms and current_lower.startswith("what about"):
            log.info("📎 Related query detected (what about pattern in domain)")
            return True

        # Check if current is asking a domain question about previous results
        # "Show me X" followed by "How many are approved?" pattern
        if previous_domain_terms:
            # If current domain terms match previous, definitely related
            if current_domain_terms:
                log.info(f"📎 Related query detected (both in domain: {current_domain_terms})")
                return True
            
            # Also check for domain-specific inquiry patterns
            # e.g., "How many are approved?", "What is the cost?", etc.
            inquiry_patterns = {
                "how many", "how much", "what is the", "what are the",
                "how many are", "show me", "list", "get me"
            }
            for pattern in inquiry_patterns:
                if current_lower.startswith(pattern):
                    # If previous was domain query and current is an inquiry, related
                    log.info(f"📎 Related query detected (inquiry pattern: {pattern})")
                    return True

        return False

    def is_question_related_llm(self, current_query: str, previous_query: str) -> bool:
        """
        Use LLM to detect if current query is related to previous query.
        
        Args:
            current_query: The new user query
            previous_query: The previous user query
            
        Returns:
            True if queries are related, False otherwise
        """
        from gateway.llm_manager import (
            get_shared_model_path,
            get_shared_pipeline,
            is_shared_pipeline_ready,
        )

        model_path = get_shared_model_path()

        if not is_shared_pipeline_ready(model_path):
            log.warning("LLM not ready for relation detection, using heuristic")
            return self.is_question_related(current_query, previous_query)

        system_prompt = (
            "You are a query relationship classifier.\n"
            "Determine if the 'Current query' is a continuation, follow-up, or "
            "related to the 'Previous query'.\n"
            "Consider:\n"
            "- Same topic or entities\n"
            "- Follow-up questions (this, that, it, they pronouns)\n"
            "- Drill-down or drill-up queries\n"
            "- Rephrasing or clarifications\n"
            "Respond with ONLY 'YES' or 'NO'."
        )

        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Previous query: {previous_query}\n"
            f"Current query: {current_query}\n"
            f"Are they related?<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        config = {
            "max_new_tokens": 4,
            "temperature": 0.0,
            "top_k": 1,
            "do_sample": False,
        }

        try:
            pipe = get_shared_pipeline(model_path)
            out = pipe(prompt, **config)
            response = out[0]["generated_text"].strip().upper()
            is_related = response.startswith("YES")
            log.info(f"🤖 LLM relation check: {is_related} (raw={response})")
            return is_related
        except Exception as exc:
            log.warning(f"LLM relation detection failed: {exc}, using heuristic")
            return self.is_question_related(current_query, previous_query)

    def extract_context(
        self,
        memory: List[Dict[str, str]],
        max_context_msgs: int = 3
    ) -> Dict[str, Any]:
        """
        Extract relevant context from conversation memory.
        
        Returns the most recent exchanges that provide useful context.
        
        Args:
            memory: List of memory messages with 'role' and 'content' keys
            max_context_msgs: Maximum number of messages to include
            
        Returns:
            Dict with 'has_context', 'context_text', 'previous_query', 'previous_answer'
        """
        if not memory or len(memory) < 2:
            return {
                "has_context": False,
                "context_text": "",
                "previous_query": "",
                "previous_answer": "",
            }

        # Extract last user query and assistant response
        user_messages = [m for m in memory if m.get("role") == "user"]
        assistant_messages = [m for m in memory if m.get("role") == "assistant"]

        if not user_messages:
            return {
                "has_context": False,
                "context_text": "",
                "previous_query": "",
                "previous_answer": "",
            }

        previous_query = user_messages[-1]["content"] if user_messages else ""
        previous_answer = assistant_messages[-1]["content"] if assistant_messages else ""

        context_lines = [
            "Previous context:",
            f"  Q: {previous_query}",
            f"  A: {previous_answer}",
        ]

        # Include additional context messages if available
        if len(user_messages) > 1 and len(assistant_messages) > 1:
            for i in range(min(max_context_msgs - 1, len(user_messages) - 1)):
                idx = -(i + 2)
                if abs(idx) <= len(user_messages):
                    context_lines.append(f"  Earlier Q: {user_messages[idx]['content']}")

        return {
            "has_context": True,
            "context_text": "\n".join(context_lines),
            "previous_query": previous_query,
            "previous_answer": previous_answer,
        }

    def augment_query_with_context(
        self,
        current_query: str,
        memory: List[Dict[str, str]],
    ) -> tuple[str, bool]:
        """
        Augment current query with relevant context if a relationship is detected.
        
        Args:
            current_query: The new user query
            memory: List of memory messages
            
        Returns:
            Tuple of (augmented_query, was_augmented)
        """
        if not memory or len(memory) < 2:
            return current_query, False

        # Get previous query
        user_messages = [m for m in memory if m.get("role") == "user"]
        if not user_messages:
            return current_query, False

        previous_query = user_messages[-1]["content"]

        # Check if related
        if self.use_llm:
            is_related = self.is_question_related_llm(current_query, previous_query)
        else:
            is_related = self.is_question_related(current_query, previous_query)

        if not is_related:
            return current_query, False

        # Extract context
        context_info = self.extract_context(memory)
        if not context_info["has_context"]:
            return current_query, False

        # Augment query with context
        augmented = (
            f"{context_info['context_text']}\n\n"
            f"Now, in relation to the above, the user asks:\n"
            f"{current_query}"
        )

        log.info("✨ Query augmented with context from previous exchange")
        return augmented, True

    def build_context_prompt_section(
        self,
        memory: List[Dict[str, str]],
        max_context_items: int = 4,
    ) -> str:
        """
        Build a context section to include in system prompts.
        
        Useful for SQL/RAG generation tools to have conversation context.
        
        Args:
            memory: List of memory messages
            max_context_items: Maximum context items to include
            
        Returns:
            Formatted context string
        """
        if not memory:
            return ""

        context_lines = ["Conversation context:"]

        # Include recent exchanges
        user_messages = [m for m in memory if m.get("role") == "user"]
        assistant_messages = [m for m in memory if m.get("role") == "assistant"]

        for i in range(min(max_context_items, len(user_messages))):
            idx = -(i + 1)
            if abs(idx) <= len(user_messages) and abs(idx) <= len(assistant_messages):
                q = user_messages[idx]["content"][:100]  # Truncate for brevity
                a = assistant_messages[idx]["content"][:100]
                context_lines.append(f"  Q: {q}{'...' if len(user_messages[idx]['content']) > 100 else ''}")
                context_lines.append(f"  A: {a}{'...' if len(assistant_messages[idx]['content']) > 100 else ''}")

        return "\n".join(context_lines)


# Global singleton instance
_context_manager: Optional[ContextMemoryManager] = None


def get_context_manager(use_llm: bool = True) -> ContextMemoryManager:
    """Get or create the global context memory manager."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextMemoryManager(use_llm=use_llm)
    return _context_manager


def is_related_question(current_query: str, previous_query: str, use_llm: bool = False) -> bool:
    """
    Convenience function to check if current query is related to previous query.
    
    Args:
        current_query: The new user query
        previous_query: The previous user query
        use_llm: Use LLM-based detection (slower but more accurate)
        
    Returns:
        True if related, False otherwise
    """
    manager = get_context_manager(use_llm=use_llm)
    if use_llm:
        return manager.is_question_related_llm(current_query, previous_query)
    return manager.is_question_related(current_query, previous_query)


def augment_with_memory_context(
    current_query: str,
    memory: List[Dict[str, str]],
    use_llm: bool = False,
) -> tuple[str, bool]:
    """
    Convenience function to augment a query with memory context.
    
    Args:
        current_query: The new user query
        memory: List of memory messages
        use_llm: Use LLM-based detection
        
    Returns:
        Tuple of (augmented_query, was_augmented)
    """
    manager = get_context_manager(use_llm=use_llm)
    return manager.augment_query_with_context(current_query, memory)
