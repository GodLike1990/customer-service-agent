"""
知识库索引模块

职责：
  1. 使用 LlamaIndex 的 SimpleDirectoryReader 加载 knowledge_docs/ 目录下的文档
  2. 使用 SentenceSplitter 将文档切分为固定大小的文本块（chunk）
  3. 使用 HuggingFace Embedding 模型（BGE）将文本块向量化
  4. 将向量存入 Milvus 向量数据库

核心函数：
  - build_index(): 首次构建索引（增量添加到已有 collection）
  - rebuild_index(): 强制重建索引（清空 collection 后重新导入）
  - incremental_update(): 增量更新（仅处理变更文件）
  - get_current_index(): 获取当前内存中的索引引用

注意：Embedding 和 Reranker 是轻量模型，在 CPU 上运行；LLM 通过远程 API 调用。
"""
from __future__ import annotations

import structlog
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import Collection, connections

from app.config import settings
from app.knowledge.file_registry import FileRegistry

logger = structlog.get_logger()

# 模块级索引引用，避免重复构建
_current_index: VectorStoreIndex | None = None


def _get_vector_store() -> MilvusVectorStore:
    """创建 Milvus 向量存储连接（增量模式，不覆盖已有数据）。"""
    return MilvusVectorStore(
        uri=f"http://{settings.milvus.host}:{settings.milvus.port}",
        collection_name=settings.milvus.collection_name,
        dim=1024,  # bge-large-zh-v1.5 的输出向量维度
        overwrite=False,
    )


def _configure_settings() -> None:
    """配置 LlamaIndex 全局设置：Embedding 模型和文本切分器。

    Settings 是 LlamaIndex 的全局单例，设置后所有索引操作自动使用。
    """
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=settings.embedding.model_name,  # 本地路径或 HF 模型名
        device=settings.embedding.device,
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=settings.knowledge_base.chunk_size,
        chunk_overlap=settings.knowledge_base.chunk_overlap,
    )
    # 索引阶段不需要 LLM（只做 embedding），显式禁用
    Settings.llm = None


def _get_registry() -> FileRegistry:
    """获取文件注册表实例，存储在知识库目录下。"""
    docs_dir = Path(settings.knowledge_base.docs_dir)
    registry_path = docs_dir / ".file_registry.json"
    return FileRegistry(registry_path)


def _delete_chunks_by_filepath(collection_name: str, filepath: str) -> int:
    """通过 file_path metadata 删除指定文件的所有 chunk。

    LlamaIndex 存储 node 时会在 metadata JSON 字段中包含 file_path。
    Milvus 支持通过表达式删除匹配的记录。

    Returns:
        删除的记录数
    """
    try:
        connections.connect(
            alias="default",
            host=settings.milvus.host,
            port=settings.milvus.port,
        )
        col = Collection(collection_name)
        col.load()
        # LlamaIndex MilvusVectorStore 将 file_path 存储在 metadata 的 JSON 中
        # 但也可能作为独立字段，需要根据实际 schema 调整
        # 尝试通过 doc_id 前缀或 metadata 匹配
        expr = f'metadata["file_path"] == "{filepath}"'
        result = col.delete(expr)
        col.flush()
        logger.info("chunks_deleted", filepath=filepath, expr=expr)
        return result.delete_count if hasattr(result, "delete_count") else 0
    except Exception as e:
        logger.warning("delete_chunks_failed", filepath=filepath, error=str(e))
        return 0


def build_index(docs_dir: str | None = None) -> VectorStoreIndex:
    """构建向量索引：加载文档 → 切分 → 向量化 → 存入 Milvus。

    如果 Milvus 中已有数据，新文档会增量追加。
    如果文档目录为空，返回一个连接到已有 collection 的空索引。
    构建完成后同步更新文件注册表。
    """
    global _current_index

    if docs_dir is None:
        docs_dir = settings.knowledge_base.docs_dir

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        docs_path.mkdir(parents=True, exist_ok=True)
        logger.warning("knowledge_docs_dir_created", path=str(docs_path))

    _configure_settings()
    vector_store = _get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 扫描目录中的有效文件
    supported_files = [f for f in docs_path.rglob("*") if f.is_file() and not f.name.startswith(".")]

    if not supported_files:
        logger.warning("knowledge_docs_empty", path=str(docs_path))
        # 连接到已有 Milvus collection（可能之前已有数据）
        _current_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        return _current_index

    logger.info("building_index", docs_dir=str(docs_path), file_count=len(supported_files))

    documents = SimpleDirectoryReader(str(docs_path)).load_data()
    _current_index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    # 同步更新注册表
    registry = _get_registry()
    for f in supported_files:
        rel_path = str(f.relative_to(docs_path))
        file_hash = FileRegistry.compute_hash(f)
        registry.update(rel_path, file_hash)
    registry.save()

    logger.info("index_built", document_count=len(documents))
    return _current_index


def rebuild_index() -> VectorStoreIndex:
    """强制重建索引：清空 Milvus collection 后重新导入所有文档。

    用于手动触发 (API) 的全量更新。重建完成后清空并重建注册表。
    """
    global _current_index

    _configure_settings()

    # overwrite=True 会清空已有 collection 重新创建
    vector_store = MilvusVectorStore(
        uri=f"http://{settings.milvus.host}:{settings.milvus.port}",
        collection_name=settings.milvus.collection_name,
        dim=1024,
        overwrite=True,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    docs_path = Path(settings.knowledge_base.docs_dir)
    if not docs_path.exists() or not any(
        f for f in docs_path.rglob("*") if f.is_file() and not f.name.startswith(".")
    ):
        logger.warning("rebuild_skipped_no_docs")
        _current_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        # 清空注册表
        registry = _get_registry()
        registry.clear()
        registry.save()
        return _current_index

    documents = SimpleDirectoryReader(str(docs_path)).load_data()
    _current_index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    # 重建注册表
    registry = _get_registry()
    registry.clear()
    supported_files = [f for f in docs_path.rglob("*") if f.is_file() and not f.name.startswith(".")]
    for f in supported_files:
        rel_path = str(f.relative_to(docs_path))
        file_hash = FileRegistry.compute_hash(f)
        registry.update(rel_path, file_hash)
    registry.save()

    logger.info("index_rebuilt", document_count=len(documents))
    return _current_index


def incremental_update() -> VectorStoreIndex:
    """增量更新索引：仅处理变更文件（新增/修改/删除）。

    流程：
      1. 通过 FileRegistry 检测变更
      2. 删除修改/删除文件的旧 chunk
      3. 重新索引新增/修改的文件
      4. 更新注册表

    如果注册表为空（首次运行或损坏），fallback 到全量重建。
    """
    global _current_index

    docs_path = Path(settings.knowledge_base.docs_dir)
    registry = _get_registry()

    # 注册表为空时 fallback 到全量重建
    if registry.is_empty:
        logger.info("registry_empty_fallback_to_rebuild")
        return rebuild_index()

    changes = registry.detect_changes(docs_path)

    # 无变更则直接返回当前索引
    if not changes["added"] and not changes["modified"] and not changes["deleted"]:
        logger.info("no_changes_detected")
        if _current_index is None:
            _configure_settings()
            vector_store = _get_vector_store()
            _current_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        return _current_index

    _configure_settings()
    collection_name = settings.milvus.collection_name

    # 删除已修改和已删除文件的旧 chunk
    for rel_path in changes["modified"] + changes["deleted"]:
        abs_path = str(docs_path / rel_path)
        _delete_chunks_by_filepath(collection_name, abs_path)
        registry.remove(rel_path)

    # 重新索引新增和修改的文件
    files_to_index = changes["added"] + changes["modified"]
    if files_to_index:
        abs_paths = [str(docs_path / rel_path) for rel_path in files_to_index]
        vector_store = _get_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        documents = SimpleDirectoryReader(input_files=abs_paths).load_data()
        if _current_index is None:
            _current_index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context
            )
        else:
            # 插入新文档到现有索引
            for doc in documents:
                _current_index.insert(doc)

        # 更新注册表
        for rel_path in files_to_index:
            abs_path = docs_path / rel_path
            file_hash = FileRegistry.compute_hash(abs_path)
            registry.update(rel_path, file_hash)

    registry.save()
    logger.info(
        "incremental_update_complete",
        added=len(changes["added"]),
        modified=len(changes["modified"]),
        deleted=len(changes["deleted"]),
    )
    return _current_index


def get_current_index() -> VectorStoreIndex | None:
    """获取当前内存中的索引实例（可能为 None，表示未初始化）。"""
    return _current_index
