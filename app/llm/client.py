"""
LLM 客户端模块

基于 LangChain 的 ChatOpenAI 封装，通过 OpenAI 兼容协议接入各种大模型。
只需在 settings.yaml 中配置 base_url/api_key/model_name 即可切换模型厂商。

支持的厂商（只要提供 OpenAI 兼容 API）：
  - OpenAI (GPT-4, GPT-3.5)
  - 百度文心一言
  - 阿里通义千问
  - 智谱 GLM
  - DeepSeek
  - 本地部署的 vLLM / Ollama 等

关键设计：
  - get_llm(): 基础 LLM 实例
  - get_llm_with_callbacks(): 带 Token 用量统计的 LLM 实例
  - TokenUsageCallback: 将 token 消耗上报到 Prometheus metrics
"""
from __future__ import annotations

from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler

from app.config import settings
from app.observability.metrics import LLM_TOKEN_USAGE


class TokenUsageCallback(BaseCallbackHandler):
    """LLM 回调：每次调用完成后记录 token 使用量到 Prometheus。

    记录三个维度：prompt_tokens、completion_tokens、total_tokens
    用于 Grafana 面板展示 token 消耗趋势和成本估算。
    """

    def on_llm_end(self, response, **kwargs) -> None:
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            if "prompt_tokens" in usage:
                LLM_TOKEN_USAGE.labels(type="prompt_tokens").inc(usage["prompt_tokens"])
            if "completion_tokens" in usage:
                LLM_TOKEN_USAGE.labels(type="completion_tokens").inc(usage["completion_tokens"])
            if "total_tokens" in usage:
                LLM_TOKEN_USAGE.labels(type="total_tokens").inc(usage["total_tokens"])


def get_llm(callbacks: Optional[list[BaseCallbackHandler]] = None) -> ChatOpenAI:
    """创建 ChatOpenAI 实例。

    所有 OpenAI 兼容的模型厂商都可以通过修改 base_url 接入，
    无需修改代码逻辑。
    """
    return ChatOpenAI(
        base_url=settings.llm.base_url,
        api_key=settings.llm.api_key,
        model=settings.llm.model_name,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
        callbacks=callbacks,
    )


def get_llm_with_callbacks() -> ChatOpenAI:
    """创建带 Token 用量监控的 ChatOpenAI 实例（业务层默认使用此方法）。"""
    return get_llm(callbacks=[TokenUsageCallback()])
