from __future__ import annotations

from fastapi import FastAPI
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

# --- Custom Metrics ---

LLM_TOKEN_USAGE = Counter(
    "llm_token_usage_total",
    "Total LLM token usage",
    ["type"],  # prompt_tokens, completion_tokens, total_tokens
)

RETRIEVAL_DURATION = Histogram(
    "knowledge_retrieval_duration_seconds",
    "Time spent retrieving from knowledge base",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

RERANK_DURATION = Histogram(
    "rerank_duration_seconds",
    "Time spent on cross-encoder reranking",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

KNOWLEDGE_REBUILD_TOTAL = Counter(
    "knowledge_rebuild_total",
    "Total number of knowledge base rebuilds",
    ["trigger"],  # manual, auto
)

REQUEST_TOTAL = Counter(
    "chat_request_total",
    "Total number of chat requests",
    ["endpoint"],  # /chat, /chat/stream
)


def setup_metrics_middleware(app: FastAPI) -> None:
    """Setup Prometheus metrics instrumentation for FastAPI.

    This adds default HTTP metrics (request count, duration, size).
    """
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/health", "/metrics"],
    ).instrument(app)
