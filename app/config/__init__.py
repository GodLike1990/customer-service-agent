"""
配置管理模块

通过 Pydantic BaseModel 定义所有配置项的类型和默认值，
从 conf/settings.yaml 加载配置，支持通过环境变量 CONFIG_PATH 指定配置文件路径。

使用方式：
    from app.config import settings
    print(settings.llm.base_url)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class LLMConfig(BaseModel):
    """大语言模型配置，通过 OpenAI 兼容协议接入各厂商模型"""
    base_url: str = "https://api.openai.com/v1"
    api_key: str = "sk-xxx"
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2048


class EmbeddingConfig(BaseModel):
    """Embedding 向量化模型配置，支持本地路径或 HuggingFace 模型名"""
    model_name: str = "BAAI/bge-large-zh-v1.5"
    device: str = "cpu"


class KnowledgeBaseConfig(BaseModel):
    """知识库文档配置：目录路径和文本切分参数"""
    docs_dir: str = "./knowledge_docs"
    chunk_size: int = 512       # 每个 chunk 的最大字符数
    chunk_overlap: int = 50     # 相邻 chunk 的重叠字符数


class MilvusConfig(BaseModel):
    """Milvus 向量数据库连接配置"""
    host: str = "milvus"        # Docker 服务名
    port: int = 19530
    collection_name: str = "customer_service_kb"


class RerankerConfig(BaseModel):
    """Cross-Encoder 精排模型配置"""
    model_name: str = "BAAI/bge-reranker-base"
    top_k: int = 5              # 精排后保留的文档数
    timeout_seconds: int = 10   # 超时自动降级


class RetrievalConfig(BaseModel):
    """向量检索配置"""
    top_k: int = 20             # Milvus 粗排返回的候选数


class ServerConfig(BaseModel):
    """服务端口配置：API 和 Metrics 端口分离"""
    host: str = "0.0.0.0"
    port: int = 8000            # 业务 API 端口
    metrics_port: int = 9100    # Prometheus 指标端口（仅内部）


class LoggingConfig(BaseModel):
    """日志配置：structlog JSON 格式输出"""
    level: str = "INFO"
    file: str = "./logs/app.log"


class WatcherConfig(BaseModel):
    """文件监听配置：自动检测知识库目录变化并触发重建"""
    enabled: bool = True
    debounce_seconds: int = 30  # 防抖：多次变更只触发一次


class Settings(BaseModel):
    """全局配置聚合，各子配置均有合理默认值"""
    llm: LLMConfig = LLMConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    knowledge_base: KnowledgeBaseConfig = KnowledgeBaseConfig()
    milvus: MilvusConfig = MilvusConfig()
    reranker: RerankerConfig = RerankerConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    server: ServerConfig = ServerConfig()
    logging: LoggingConfig = LoggingConfig()
    watcher: WatcherConfig = WatcherConfig()


def load_settings(config_path: Optional[str] = None) -> Settings:
    """从 YAML 文件加载配置。

    优先级：参数传入路径 > 环境变量 CONFIG_PATH > 默认路径 conf/settings.yaml
    如果配置文件不存在，则使用所有默认值。
    """
    if config_path is None:
        config_path = os.environ.get(
            "CONFIG_PATH", str(Path(__file__).parent.parent.parent / "conf" / "settings.yaml")
        )

    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return Settings(**data)

    return Settings()


# 模块级单例，应用启动时加载一次
settings = load_settings()
