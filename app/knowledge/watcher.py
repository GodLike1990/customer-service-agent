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
    """Watches knowledge docs directory and triggers rebuild on changes."""

    def __init__(self, debounce_seconds: int = 30):
        super().__init__()
        self.debounce_seconds = debounce_seconds
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def on_any_event(self, event: FileSystemEvent) -> None:
        # Ignore directory events and temporary files
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
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(
                self.debounce_seconds,
                self._trigger_rebuild,
            )
            self._timer.daemon = True
            self._timer.start()

    def _trigger_rebuild(self) -> None:
        """Execute incremental knowledge base update."""
        logger.info("auto_incremental_update_triggered")
        try:
            from app.knowledge.indexer import incremental_update
            incremental_update()
            KNOWLEDGE_REBUILD_TOTAL.labels(trigger="auto").inc()
            logger.info("auto_incremental_update_complete")
        except Exception as e:
            logger.error("auto_incremental_update_failed", error=str(e))


def start_watcher() -> None:
    """Start file system watcher for knowledge docs directory."""
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
    """Stop file system watcher."""
    global _observer
    if _observer is not None:
        _observer.stop()
        _observer.join(timeout=5)
        _observer = None
        logger.info("knowledge_watcher_stopped")
