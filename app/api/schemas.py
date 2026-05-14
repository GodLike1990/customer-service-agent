from __future__ import annotations

from pydantic import BaseModel, Field


class SourceItem(BaseModel):
    text: str = Field(..., description="来源文本片段")
    score: float | None = Field(None, description="相关性分数")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4096, description="用户问题")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="AI 回答")
    sources: list[SourceItem] = Field(default_factory=list, description="引用来源")
    has_knowledge_context: bool = Field(default=False, description="是否使用了知识库上下文")


class KnowledgeRebuildResponse(BaseModel):
    status: str = Field(..., description="重建状态")
    message: str = Field(..., description="详细信息")


class KnowledgeStatusResponse(BaseModel):
    status: str = Field(..., description="知识库状态")
    collection_name: str = Field(..., description="Milvus collection 名称")
    docs_dir: str = Field(..., description="文档目录路径")


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
