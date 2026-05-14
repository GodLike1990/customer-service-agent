from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.schemas import ChatRequest, HealthResponse
from app.service.chat_service import ChatService

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat")
async def chat(request: ChatRequest):
    """问答接口：检索知识库并生成回答。"""
    service = ChatService()
    response = await service.answer(request.question)
    return {
        "answer": response.answer,
        "sources": response.sources,
        "has_knowledge_context": response.has_knowledge_context,
    }


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE 流式问答接口。"""
    service = ChatService()
    return StreamingResponse(
        service.answer_stream(request.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """健康检查端点。"""
    return HealthResponse()
