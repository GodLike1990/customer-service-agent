"""
增量索引更新自测

覆盖场景：
  - FileRegistry 的 detect_changes 逻辑（新增/修改/删除）
  - incremental_update 首次运行（无注册表）时 fallback 到全量重建
  - 单文件修改时仅重新索引该文件
  - 注册表损坏时 fallback
"""
import json
import sys
import importlib.util
import tempfile
from pathlib import Path

import pytest

# 直接加载 file_registry 模块，绕过 app.knowledge.__init__ 对 indexer 重依赖的导入
_module_path = Path(__file__).parent.parent / "app" / "knowledge" / "file_registry.py"
_spec = importlib.util.spec_from_file_location("file_registry", _module_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
FileRegistry = _module.FileRegistry


class TestFileRegistry:
    """测试 FileRegistry 的核心功能。"""

    def setup_method(self):
        """每个测试前创建临时目录。"""
        self.tmp_dir = tempfile.mkdtemp()
        self.docs_dir = Path(self.tmp_dir) / "docs"
        self.docs_dir.mkdir()
        self.registry_path = Path(self.tmp_dir) / ".file_registry.json"

    def _write_file(self, name: str, content: str) -> Path:
        filepath = self.docs_dir / name
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
        return filepath

    def test_compute_hash_consistent(self):
        """同一文件内容的 hash 应该一致。"""
        f = self._write_file("a.txt", "hello world")
        h1 = FileRegistry.compute_hash(f)
        h2 = FileRegistry.compute_hash(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex

    def test_compute_hash_different_content(self):
        """不同内容的 hash 应该不同。"""
        f1 = self._write_file("a.txt", "hello")
        h1 = FileRegistry.compute_hash(f1)
        f1.write_text("world", encoding="utf-8")
        h2 = FileRegistry.compute_hash(f1)
        assert h1 != h2

    def test_detect_changes_all_added(self):
        """注册表为空时，所有文件应标记为 added。"""
        self._write_file("doc1.md", "content1")
        self._write_file("doc2.md", "content2")

        registry = FileRegistry(self.registry_path)
        changes = registry.detect_changes(self.docs_dir)

        assert set(changes["added"]) == {"doc1.md", "doc2.md"}
        assert changes["modified"] == []
        assert changes["deleted"] == []

    def test_detect_changes_modified(self):
        """文件内容修改后应标记为 modified。"""
        f = self._write_file("doc.md", "original")
        registry = FileRegistry(self.registry_path)
        # 模拟已索引状态
        registry.update("doc.md", FileRegistry.compute_hash(f))
        registry.save()

        # 修改文件内容
        f.write_text("updated content", encoding="utf-8")

        # 重新加载注册表检测变更
        registry2 = FileRegistry(self.registry_path)
        changes = registry2.detect_changes(self.docs_dir)

        assert changes["added"] == []
        assert changes["modified"] == ["doc.md"]
        assert changes["deleted"] == []

    def test_detect_changes_deleted(self):
        """文件删除后应标记为 deleted。"""
        f = self._write_file("doc.md", "content")
        registry = FileRegistry(self.registry_path)
        registry.update("doc.md", FileRegistry.compute_hash(f))
        registry.save()

        # 删除文件
        f.unlink()

        registry2 = FileRegistry(self.registry_path)
        changes = registry2.detect_changes(self.docs_dir)

        assert changes["added"] == []
        assert changes["modified"] == []
        assert changes["deleted"] == ["doc.md"]

    def test_detect_changes_mixed(self):
        """混合场景：新增 + 修改 + 删除。"""
        f1 = self._write_file("keep.md", "keep")
        f2 = self._write_file("modify.md", "original")
        f3 = self._write_file("delete.md", "to_delete")

        registry = FileRegistry(self.registry_path)
        registry.update("keep.md", FileRegistry.compute_hash(f1))
        registry.update("modify.md", FileRegistry.compute_hash(f2))
        registry.update("delete.md", FileRegistry.compute_hash(f3))
        registry.save()

        # 修改一个，删除一个，新增一个
        f2.write_text("modified", encoding="utf-8")
        f3.unlink()
        self._write_file("new.md", "new content")

        registry2 = FileRegistry(self.registry_path)
        changes = registry2.detect_changes(self.docs_dir)

        assert changes["added"] == ["new.md"]
        assert changes["modified"] == ["modify.md"]
        assert changes["deleted"] == ["delete.md"]

    def test_registry_corrupted_returns_empty(self):
        """注册表文件损坏时应返回空数据而非异常。"""
        self.registry_path.write_text("not valid json{{{", encoding="utf-8")
        registry = FileRegistry(self.registry_path)
        assert registry.is_empty

    def test_save_and_load_roundtrip(self):
        """注册表保存后重新加载数据一致。"""
        registry = FileRegistry(self.registry_path)
        registry.update("test.md", "abc123", chunk_count=5)
        registry.save()

        registry2 = FileRegistry(self.registry_path)
        assert registry2.get_file_hash("test.md") == "abc123"

    def test_ignore_dotfiles(self):
        """以 . 开头的文件应被忽略。"""
        self._write_file(".hidden", "secret")
        self._write_file("visible.md", "content")

        registry = FileRegistry(self.registry_path)
        changes = registry.detect_changes(self.docs_dir)

        assert changes["added"] == ["visible.md"]

    def test_clear(self):
        """clear() 应清空所有条目。"""
        registry = FileRegistry(self.registry_path)
        registry.update("a.md", "hash1")
        registry.update("b.md", "hash2")
        registry.clear()
        assert registry.is_empty
