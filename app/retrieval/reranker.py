"""
Cross-Encoder 精排模块

职责：对 Milvus 粗排返回的 Top20 候选文档进行二次精排。

工作原理：
  - 粗排（Milvus）：使用 bi-encoder（双塔模型），query 和 doc 分别编码后计算余弦相似度，速度快但精度有限
  - 精排（CrossEncoder）：将 [query, doc] 拼接后一起输入模型，计算交互式相似度，精度高但速度慢

设计考虑：
  - 单例模式：避免重复加载模型（约 1GB）
  - 超时降级：如果精排超时，自动使用粗排结果（保证可用性）
  - Prometheus 埋点：记录精排耗时用于监控
"""
from __future__ import annotations

import concurrent.futures
from typing import Optional

import structlog
from sentence_transformers import CrossEncoder

from app.config import settings
from app.observability.metrics import RERANK_DURATION

logger = structlog.get_logger()


class Reranker:
    """Cross-Encoder 精排器，对检索结果做二次排序。

    使用 BAAI/bge-reranker-base 模型，在 CPU 上运行。
    不是 LLM，只是一个轻量的文本对相似度打分模型。
    """

    _instance: Optional["Reranker"] = None
    _model: Optional[CrossEncoder] = None

    def __new__(cls) -> "Reranker":
        """单例：整个应用生命周期只加载一次模型。"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            logger.info("loading_reranker_model", model=settings.reranker.model_name)
            self.__class__._model = CrossEncoder(settings.reranker.model_name)
            logger.info("reranker_model_loaded")

    @RERANK_DURATION.time()
    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """对候选文档进行精排。

        流程：构造 [query, doc] 对 → CrossEncoder 打分 → 按分数降序取 Top-K

        Args:
            query: 用户问题
            documents: 粗排返回的文档列表（需包含 'text' 字段）
            top_k: 精排后保留的文档数

        Returns:
            精排后的文档列表，新增 'rerank_score' 字段
        """
        if not documents:
            return []

        top_k = top_k or settings.reranker.top_k
        texts = [doc["text"] for doc in documents]

        try:
            scores = self._predict_with_timeout(query, texts)
        except concurrent.futures.TimeoutError:
            # 降级策略：精排超时则直接返回粗排结果的前 top_k 条
            logger.warning(
                "reranker_timeout",
                timeout=settings.reranker.timeout_seconds,
                falling_back_to="coarse_ranking",
            )
            return documents[:top_k]

        # 将精排分数附加到文档上
        scored_docs = []
        for doc, score in zip(documents, scores):
            scored_docs.append({
                **doc,
                "rerank_score": float(score),
            })

        # 按精排分数降序排列
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        logger.info(
            "rerank_complete",
            input_count=len(documents),
            output_count=min(top_k, len(scored_docs)),
        )
        return scored_docs[:top_k]

    def _predict_with_timeout(self, query: str, texts: list[str]) -> list[float]:
        """带超时保护的 CrossEncoder 推理。

        使用线程池实现超时控制，避免模型推理时间过长阻塞请求。
        """
        pairs = [[query, text] for text in texts]

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._model.predict, pairs)
            scores = future.result(timeout=settings.reranker.timeout_seconds)

        return scores.tolist()
