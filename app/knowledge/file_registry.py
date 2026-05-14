"""
文件注册表模块

维护已索引文件的 SHA256 哈希注册表，用于增量更新时检测文件变更。
注册表以 JSON 格式持久化在知识库目录下（.file_registry.json）。

核心功能：
  - detect_changes(): 扫描目录对比 hash，返回新增/修改/删除的文件列表
  - compute_hash(): 计算文件 SHA256
  - update() / remove(): 管理注册表条目
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import structlog

logger = structlog.get_logger()


class FileEntry(TypedDict):
    hash: str
    chunk_count: int
    indexed_at: str


class ChangeSet(TypedDict):
    added: list[str]
    modified: list[str]
    deleted: list[str]


class FileRegistry:
    """管理已索引文件的 hash 注册表，用于检测文件变更。"""

    def __init__(self, registry_path: str | Path):
        self.path = Path(registry_path)
        self._data: dict[str, FileEntry] = self._load()

    def _load(self) -> dict[str, FileEntry]:
        """从 JSON 文件加载注册表，文件不存在或损坏时返回空字典。"""
        if not self.path.exists():
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return raw.get("files", {})
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("file_registry_corrupted", path=str(self.path), error=str(e))
            return {}

    def save(self) -> None:
        """持久化注册表到 JSON 文件。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"files": self._data}, f, ensure_ascii=False, indent=2)

    @staticmethod
    def compute_hash(filepath: str | Path) -> str:
        """计算文件内容的 SHA256 哈希值。"""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def get_file_hash(self, relative_path: str) -> str | None:
        """获取注册表中记录的文件 hash。"""
        entry = self._data.get(relative_path)
        return entry["hash"] if entry else None

    def detect_changes(self, docs_dir: str | Path) -> ChangeSet:
        """扫描目录，对比注册表，返回变更分类。

        Args:
            docs_dir: 知识库文档目录路径

        Returns:
            ChangeSet: {added: [...], modified: [...], deleted: [...]}
            列表中的路径为相对于 docs_dir 的相对路径
        """
        docs_path = Path(docs_dir)
        current_files: dict[str, str] = {}

        # 扫描目录中的所有有效文件
        for f in docs_path.rglob("*"):
            if f.is_file() and not f.name.startswith("."):
                rel_path = str(f.relative_to(docs_path))
                current_files[rel_path] = self.compute_hash(f)

        added: list[str] = []
        modified: list[str] = []
        deleted: list[str] = []

        # 检测新增和修改
        for rel_path, file_hash in current_files.items():
            old_hash = self.get_file_hash(rel_path)
            if old_hash is None:
                added.append(rel_path)
            elif old_hash != file_hash:
                modified.append(rel_path)

        # 检测删除
        for rel_path in self._data:
            if rel_path not in current_files:
                deleted.append(rel_path)

        logger.info(
            "changes_detected",
            added=len(added),
            modified=len(modified),
            deleted=len(deleted),
        )
        return {"added": added, "modified": modified, "deleted": deleted}

    def update(self, relative_path: str, file_hash: str, chunk_count: int = 0) -> None:
        """更新或新增注册表条目。"""
        self._data[relative_path] = {
            "hash": file_hash,
            "chunk_count": chunk_count,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }

    def remove(self, relative_path: str) -> None:
        """移除注册表条目。"""
        self._data.pop(relative_path, None)

    def clear(self) -> None:
        """清空注册表。"""
        self._data.clear()

    @property
    def is_empty(self) -> bool:
        return len(self._data) == 0
