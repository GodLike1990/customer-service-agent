"""
应用启动引导模块 (Bootstrap)

职责：统一管理应用的初始化流程，包括：
  - 日志系统初始化（structlog JSON 格式）
  - 全局异常处理注册
  - Prometheus 指标中间件注册
  - API 路由注册
  - Metrics 独立端口启动（与业务端口分离）
  - 知识库文件监听器启动

入口：create_app() 工厂函数，由 main.py 调用
"""
from __future__ import annotations

import logging
from pathlib import Path

import structlog
from fastapi import FastAPI
from prometheus_client import start_http_server

from app.config import settings


def create_app() -> FastAPI:
    """应用工厂函数：创建并完整配置 FastAPI 应用实例。

    初始化顺序：
      1. 日志 → 2. FastAPI → 3. 异常处理 → 4. Metrics 中间件
      → 5. 路由 → 6. Metrics 端口 → 7. 文件监听
    """
    # 1. 确保日志目录存在
    log_file = Path(settings.logging.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # 2. 初始化结构化日志（必须在其他模块使用 logger 之前完成）
    configure_logging()

    # 3. 创建 FastAPI 实例
    app = FastAPI(
        title="Customer Service Agent",
        version="1.0.0",
        description="Intelligent customer service system powered by LangChain + LlamaIndex",
    )

    # 4. 注册全局异常处理器（捕获未处理异常，统一返回格式）
    from app.api.error_handlers import register_handlers
    register_handlers(app)

    # 5. 注册 Prometheus 指标中间件（自动记录 HTTP 请求指标）
    from app.observability.metrics import setup_metrics_middleware
    setup_metrics_middleware(app)

    # 6. 注册业务路由
    from app.api.routes import router
    from app.api.knowledge_routes import knowledge_router
    app.include_router(router)
    app.include_router(knowledge_router)

    # 7. 在独立端口启动 Prometheus metrics（与业务 API 端口分离）
    #    Prometheus 通过 Docker 内部网络抓取此端口，不对外暴露
    start_http_server(settings.server.metrics_port)

    # 8. 启动时构建知识库索引（确保容器重启后索引可用）
    from app.knowledge.indexer import build_index
    try:
        build_index()
    except Exception as e:
        structlog.get_logger().warning("startup_index_build_failed", error=str(e))

    # 9. 启动知识库文件监听器（watchdog），检测文档变更自动触发重建
    if settings.watcher.enabled:
        from app.knowledge.watcher import start_watcher
        start_watcher()

    logger = structlog.get_logger()
    logger.info(
        "application_started",
        api_port=settings.server.port,
        metrics_port=settings.server.metrics_port,
    )

    return app


def configure_logging() -> None:
    """配置 structlog 结构化日志。

    输出格式：JSON，包含 timestamp、level、event 等字段
    输出目标：settings.logging.file 指定的文件
    """
    log_level = getattr(logging, settings.logging.level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,  # 支持请求级上下文（如 trace_id）
            structlog.processors.TimeStamper(fmt="iso"),  # ISO 格式时间戳
            structlog.processors.add_log_level,       # 添加日志级别字段
            structlog.processors.StackInfoRenderer(), # 异常堆栈渲染
            structlog.processors.format_exc_info,     # 格式化异常信息
            structlog.processors.JSONRenderer(),      # 最终输出为 JSON
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.WriteLoggerFactory(
            file=open(settings.logging.file, "a", encoding="utf-8")
        ),
        cache_logger_on_first_use=True,
    )
