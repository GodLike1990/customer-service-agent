"""
Prometheus 监控指标定义

定义客服系统的核心业务指标，供 Grafana 面板展示：
  - LLM Token 消耗量
  - 知识库检索耗时
  - 精排耗时
  - 知识库重建次数
  - 请求总量
"""
from __future__ import annotations

from fastapi import FastAPI
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

# --- 自定义业务指标 ---

LLM_TOKEN_USAGE = Counter(
    "llm_token_usage_total",
    "LLM token 使用总量",
    ["type"],  # prompt_tokens, completion_tokens, total_tokens
)

RETRIEVAL_DURATION = Histogram(
    "knowledge_retrieval_duration_seconds",
    "知识库检索耗时（秒）",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

RERANK_DURATION = Histogram(
    "rerank_duration_seconds",
    "Cross-Encoder 精排耗时（秒）",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

KNOWLEDGE_REBUILD_TOTAL = Counter(
    "knowledge_rebuild_total",
    "知识库重建次数",
    ["trigger"],  # manual（手动触发）, auto（自动触发）
)

REQUEST_TOTAL = Counter(
    "chat_request_total",
    "问答请求总量",
    ["endpoint"],  # /chat, /chat/stream
)


def setup_metrics_middleware(app: FastAPI) -> None:
    """为 FastAPI 注册 Prometheus 指标中间件。

    自动记录 HTTP 请求的计数、耗时和响应大小等标准指标。
    """
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/health", "/metrics"],
    ).instrument(app)
