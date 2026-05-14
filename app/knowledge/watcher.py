"""
知识库文件监听模块

使用 watchdog 监听 knowledge_docs/ 目录的文件变化，
检测到变更后通过防抖机制触发增量索引更新。

核心组件：
  - KnowledgeFileHandler: 文件事件处理器，带防抖定时器
  - start_watcher(): 启动文件监听
  - stop_watcher(): 停止文件监听
"""
from __future__ import annotations

import threading
from pathlib import Path

import structlog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from app.config import settings
from app.observability.metrics import KNOWLEDGE_REBUILD_TOTAL

logger = structlog.get_logger()

_observer: Observer | None = None


class KnowledgeFileHandler(FileSystemEventHandler):
    """监听知识库文档目录，检测到文件变化后触发增量更新。"""

    def __init__(self, debounce_seconds: int = 30):
        super().__init__()
        self.debounce_seconds = debounce_seconds
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def on_any_event(self, event: FileSystemEvent) -> None:
        # 忽略目录事件和临时文件
        if event.is_directory:
            return
        if event.src_path.startswith("."):
            return

        logger.info(
            "knowledge_file_change_detected",
            event_type=event.event_type,
            path=event.src_path,
        )

        with self._lock:
            # 防抖：取消之前的定时器，重新计时
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(
                self.debounce_seconds,
                self._trigger_rebuild,
            )
            self._timer.daemon = True
            self._timer.start()

    def _trigger_rebuild(self) -> None:
        """执行增量知识库更新。"""
        logger.info("auto_incremental_update_triggered")
        try:
            from app.knowledge.indexer import incremental_update
            incremental_update()
            KNOWLEDGE_REBUILD_TOTAL.labels(trigger="auto").inc()
            logger.info("auto_incremental_update_complete")
        except Exception as e:
            logger.error("auto_incremental_update_failed", error=str(e))


def start_watcher() -> None:
    """启动文件系统监听器，监听知识库文档目录。"""
    global _observer

    docs_dir = Path(settings.knowledge_base.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    handler = KnowledgeFileHandler(
        debounce_seconds=settings.watcher.debounce_seconds,
    )

    _observer = Observer()
    _observer.schedule(handler, str(docs_dir), recursive=True)
    _observer.daemon = True
    _observer.start()

    logger.info("knowledge_watcher_started", path=str(docs_dir))


def stop_watcher() -> None:
    """停止文件系统监听器。"""
    global _observer
    if _observer is not None:
        _observer.stop()
        _observer.join(timeout=5)
        _observer = None
        logger.info("knowledge_watcher_stopped")
