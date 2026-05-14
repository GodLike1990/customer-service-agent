from __future__ import annotations

import structlog
from fastapi import APIRouter, BackgroundTasks

from app.api.schemas import KnowledgeRebuildResponse, KnowledgeStatusResponse
from app.config import settings
from app.knowledge.indexer import rebuild_index, get_current_index
from app.observability.metrics import KNOWLEDGE_REBUILD_TOTAL

logger = structlog.get_logger()

knowledge_router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])

# Track rebuild state
_rebuild_in_progress = False


async def _do_rebuild():
    """Background task to rebuild the knowledge index."""
    global _rebuild_in_progress
    try:
        _rebuild_in_progress = True
        rebuild_index()
        KNOWLEDGE_REBUILD_TOTAL.labels(trigger="manual").inc()
        logger.info("knowledge_rebuild_complete", trigger="manual")
    except Exception as e:
        logger.error("knowledge_rebuild_failed", error=str(e))
    finally:
        _rebuild_in_progress = False


@knowledge_router.post("/rebuild", response_model=KnowledgeRebuildResponse)
async def trigger_rebuild(background_tasks: BackgroundTasks):
    """手动触发知识库重建。"""
    global _rebuild_in_progress

    if _rebuild_in_progress:
        return KnowledgeRebuildResponse(
            status="in_progress",
            message="知识库重建正在进行中，请稍后再试。",
        )

    background_tasks.add_task(_do_rebuild)
    return KnowledgeRebuildResponse(
        status="accepted",
        message="知识库重建任务已提交，将在后台执行。",
    )


@knowledge_router.get("/status", response_model=KnowledgeStatusResponse)
async def get_status():
    """查询知识库当前状态。"""
    index = get_current_index()
    status = "ready" if index is not None else "not_initialized"

    if _rebuild_in_progress:
        status = "rebuilding"

    return KnowledgeStatusResponse(
        status=status,
        collection_name=settings.milvus.collection_name,
        docs_dir=settings.knowledge_base.docs_dir,
    )
