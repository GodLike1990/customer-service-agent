"""
核心业务逻辑层 - ChatService

串联整个问答流程：
  用户问题 → 知识库检索(Top20) → Cross-Encoder精排(Top5) → 构建Prompt → LLM生成回答

降级策略：
  - 知识库为空/无相关结果 → 使用 LLM 通用能力回答，提示用户联系人工客服
  - 精排超时 → 使用粗排结果（由 Reranker 内部处理）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncGenerator

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.knowledge.retriever import KnowledgeRetriever
from app.retrieval.reranker import Reranker
from app.llm.client import get_llm_with_callbacks
from app.observability.metrics import LLM_TOKEN_USAGE

logger = structlog.get_logger()

# 有知识库上下文时的系统提示词
SYSTEM_PROMPT = """你是一个专业的智能客服助手。请根据以下参考资料回答用户的问题。
如果参考资料中没有相关信息，请诚实地告知用户你不确定，并建议他们联系人工客服。
回答要求：简洁、准确、有帮助。

参考资料：
{context}
"""

# 无知识库上下文时的降级提示词
SYSTEM_PROMPT_NO_CONTEXT = """你是一个专业的智能客服助手。
当前知识库暂无相关资料，请基于你的通用知识尽力回答用户问题。
如果无法确定答案，请建议用户联系人工客服。
"""


@dataclass
class ChatResponse:
    """问答响应结构体"""
    answer: str                                    # LLM 生成的回答
    sources: list[dict] = field(default_factory=list)  # 引用的知识库来源
    has_knowledge_context: bool = False             # 是否使用了知识库上下文


class ChatService:
    """智能客服核心服务，编排 检索→精排→生成 的完整 RAG 流程。"""

    def __init__(self):
        self.retriever = KnowledgeRetriever()       # Milvus 向量检索
        self.reranker = Reranker()                  # CrossEncoder 精排
        self.llm = get_llm_with_callbacks()         # LLM（带 token 统计）

    async def answer(self, question: str) -> ChatResponse:
        """同步问答：返回完整回答。

        完整流程：检索 → 精排 → 拼接上下文 → LLM 生成
        """
        logger.info("chat_request", question_length=len(question))

        # Step 1: 从 Milvus 检索 Top20 候选文档
        retrieved_docs = self.retriever.retrieve(question)

        # Step 2: 无检索结果时走降级逻辑
        if not retrieved_docs:
            logger.info("no_knowledge_context", question=question[:100])
            return await self._generate_without_context(question)

        # Step 3: CrossEncoder 精排，从 Top20 中选出 Top5
        reranked_docs = self.reranker.rerank(question, retrieved_docs)

        # Step 4: 将精排文档拼接为上下文，送入 LLM 生成回答
        context = self._build_context(reranked_docs)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
            HumanMessage(content=question),
        ]

        response = await self.llm.ainvoke(messages)

        # 记录 token 用量到 Prometheus
        self._record_token_usage(response)

        # 构建来源信息（返回给前端展示引用）
        sources = [
            {"text": doc["text"][:200], "score": doc.get("rerank_score", doc.get("score"))}
            for doc in reranked_docs
        ]

        logger.info("chat_response_generated", sources_count=len(sources))
        return ChatResponse(
            answer=response.content,
            sources=sources,
            has_knowledge_context=True,
        )

    async def answer_stream(self, question: str) -> AsyncGenerator[str, None]:
        """流式问答：通过 SSE 逐字返回 LLM 生成内容。

        前端可实时展示打字效果，提升用户体验。
        """
        logger.info("chat_stream_request", question_length=len(question))

        # 检索与精排（同步完成，流式仅在 LLM 生成阶段）
        retrieved_docs = self.retriever.retrieve(question)

        if not retrieved_docs:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT_NO_CONTEXT),
                HumanMessage(content=question),
            ]
        else:
            reranked_docs = self.reranker.rerank(question, retrieved_docs)
            context = self._build_context(reranked_docs)
            messages = [
                SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
                HumanMessage(content=question),
            ]

        # 流式输出 LLM 响应（SSE 格式）
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield f"data: {chunk.content}\n\n"

        yield "data: [DONE]\n\n"

    async def _generate_without_context(self, question: str) -> ChatResponse:
        """降级模式：知识库无结果时，仅依赖 LLM 通用能力回答。"""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT_NO_CONTEXT),
            HumanMessage(content=question),
        ]
        response = await self.llm.ainvoke(messages)
        self._record_token_usage(response)
        return ChatResponse(
            answer=response.content,
            sources=[],
            has_knowledge_context=False,
        )

    @staticmethod
    def _build_context(documents: list[dict]) -> str:
        """将精排后的文档列表拼接为带编号的上下文字符串。"""
        parts = []
        for i, doc in enumerate(documents, 1):
            parts.append(f"[{i}] {doc['text']}")
        return "\n\n".join(parts)

    @staticmethod
    def _record_token_usage(response) -> None:
        """从 LLM 响应中提取 token 用量并记录到 Prometheus。

        LangChain ChatOpenAI 的响应中 token 信息可能在：
          - response.usage_metadata (dict, 新版，含 input_tokens/output_tokens)
          - response.response_metadata["token_usage"] (dict, 旧版，含 prompt_tokens/completion_tokens)
        """
        prompt = 0
        completion = 0
        total = 0

        # 优先从 usage_metadata 获取（新版 LangChain）
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            prompt = um.get("input_tokens", 0) if isinstance(um, dict) else getattr(um, "input_tokens", 0)
            completion = um.get("output_tokens", 0) if isinstance(um, dict) else getattr(um, "output_tokens", 0)
            total = um.get("total_tokens", 0) if isinstance(um, dict) else getattr(um, "total_tokens", 0)
        # 回退到 response_metadata（旧版）
        elif hasattr(response, "response_metadata") and response.response_metadata:
            token_usage = response.response_metadata.get("token_usage", {})
            prompt = token_usage.get("prompt_tokens", 0)
            completion = token_usage.get("completion_tokens", 0)
            total = token_usage.get("total_tokens", 0)

        if not total:
            total = prompt + completion

        if total > 0:
            LLM_TOKEN_USAGE.labels(type="prompt_tokens").inc(prompt)
            LLM_TOKEN_USAGE.labels(type="completion_tokens").inc(completion)
            LLM_TOKEN_USAGE.labels(type="total_tokens").inc(total)
            logger.info("token_usage_recorded", prompt=prompt, completion=completion, total=total)
        else:
            logger.warning("token_usage_not_available")
