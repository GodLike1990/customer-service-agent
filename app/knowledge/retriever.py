"""
知识库检索模块

封装 Milvus 向量相似度检索逻辑，返回 Top-K 相似文档。
检索结果将传递给 Reranker 做精排。

调用链路：用户问题 → Embedding → Milvus 相似度检索 → 返回候选文档列表
"""
from __future__ import annotations

from typing import Optional

import structlog
from llama_index.core import VectorStoreIndex

from app.config import settings
from app.knowledge.indexer import get_current_index, build_index
from app.observability.metrics import RETRIEVAL_DURATION

logger = structlog.get_logger()


class KnowledgeRetriever:
    """知识库检索器：从 Milvus 中检索与 query 最相似的文档块。"""

    def __init__(self, top_k: Optional[int] = None):
        self.top_k = top_k or settings.retrieval.top_k  # 默认 Top20

    def _get_index(self) -> VectorStoreIndex:
        """获取或构建向量索引（懒加载）。"""
        index = get_current_index()
        if index is None:
            index = build_index()
        return index

    @RETRIEVAL_DURATION.time()
    def retrieve(self, query: str) -> list[dict]:
        """检索与 query 最相似的 top_k 个文档。

        Args:
            query: 用户提出的问题

        Returns:
            文档列表，每项包含 text（文本内容）、score（相似度分数）、metadata
        """
        index = self._get_index()
        retriever = index.as_retriever(similarity_top_k=self.top_k)
        nodes = retriever.retrieve(query)

        results = []
        for node in nodes:
            results.append({
                "text": node.get_content(),
                "score": node.get_score(),
                "metadata": node.metadata,
            })

        logger.info("retrieval_complete", query_length=len(query), results_count=len(results))
        return results
