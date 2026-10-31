"""Red-first tests for WP-05: engine_key(), unload() for all engine kinds, _meta_lock.

Verifies that ModelManager provides a canonical static engine_key(), that embedding
and vision engine eviction calls unload(), that unload_all() clears all three caches,
and that _meta_lock is a separate threading.Lock from _lock.
"""

import threading
import unittest.mock as mock

import pytest

from arc_runner.server.model_manager import ModelManager


class TestEngineKey:
    """ModelManager.engine_key() must return a deterministic cache-key string."""

    def test_engine_key_method_exists_ok(self):
        """ModelManager must expose a static or class method named engine_key."""
        assert hasattr(ModelManager, "engine_key"), (
            "ModelManager must have an engine_key() static/class method"
        )
        assert callable(ModelManager.engine_key), (
            "ModelManager.engine_key must be callable"
        )

    def test_engine_key_canonical_format_ok(self):
        """engine_key returns model_id:device:none:NPU:4 for a simple LLM request."""
        key = ModelManager.engine_key("my-model", "GPU")
        assert "my-model" in key, "engine_key must contain the model_id"
        assert "GPU" in key, "engine_key must contain the device"

    def test_engine_key_with_draft_differs_from_without_ok(self):
        """engine_key with draft_model_id must differ from the key without a draft."""
        key_plain = ModelManager.engine_key("model", "GPU")
        key_draft = ModelManager.engine_key("model", "GPU", draft_model_id="draft-model")
        assert key_plain != key_draft, (
            "engine_key must produce distinct keys for plain vs. speculative engines"
        )

    def test_engine_key_same_args_same_result_ok(self):
        """engine_key must be deterministic: same arguments always return the same key."""
        k1 = ModelManager.engine_key("my-model", "GPU", draft_model_id="d", draft_device="NPU", gamma=4)
        k2 = ModelManager.engine_key("my-model", "GPU", draft_model_id="d", draft_device="NPU", gamma=4)
        assert k1 == k2, "engine_key must be deterministic"


class TestMetaLock:
    """ModelManager must have a _meta_lock separate from _lock for metadata counters."""

    def test_meta_lock_exists_ok(self):
        """ModelManager must have a _meta_lock attribute."""
        mm = ModelManager()
        assert hasattr(mm, "_meta_lock"), (
            "ModelManager must have a _meta_lock attribute for metadata counter isolation"
        )

    def test_meta_lock_is_threading_lock_ok(self):
        """_meta_lock must be a threading.Lock or RLock instance."""
        mm = ModelManager()
        assert isinstance(mm._meta_lock, type(threading.Lock())), (
            "_meta_lock must be a threading.Lock instance"
        )

    def test_meta_lock_separate_from_cache_lock_ok(self):
        """_meta_lock must not be the same object as _lock."""
        mm = ModelManager()
        assert mm._meta_lock is not mm._lock, (
            "_meta_lock must be a separate Lock object from _lock"
        )


class TestEmbeddingEngineEvictionCallsUnload:
    """Evicting an EmbeddingEngine from the cache must call its unload() method."""

    def test_embedding_engine_eviction_calls_unload_ok(self, tmp_path):
        """Evicting an EmbeddingEngine must call unload() on it."""
        mm = ModelManager(model_dirs=[str(tmp_path)], max_loaded_models=1)
        mock_engine_a = mock.MagicMock()
        mock_engine_b = mock.MagicMock()

        with mm._lock:
            mm._embedding_cache["model-a:NPU"] = mock_engine_a

        model_b_dir = tmp_path / "model-b"
        model_b_dir.mkdir()
        (model_b_dir / "openvino_model.xml").write_text("")

        with mock.patch.object(mm, "resolve_model_path", return_value=model_b_dir), \
             mock.patch("arc_runner.server.model_manager.EmbeddingEngine", return_value=mock_engine_b):
            mm.get_or_load_embedding_engine("model-b", device="NPU")

        mock_engine_a.unload.assert_called_once(), (
            "Evicted EmbeddingEngine must have unload() called on it"
        )

    def test_vision_engine_eviction_calls_unload_ok(self, tmp_path):
        """Evicting a VisionEngine must call unload() on it."""
        mm = ModelManager(model_dirs=[str(tmp_path)], max_loaded_models=1)
        mock_engine_a = mock.MagicMock()
        mock_engine_b = mock.MagicMock()

        with mm._lock:
            mm._vision_cache["model-a:GPU"] = mock_engine_a

        model_b_dir = tmp_path / "model-b"
        model_b_dir.mkdir()

        with mock.patch.object(mm, "resolve_model_path", return_value=model_b_dir), \
             mock.patch("arc_runner.server.model_manager.VisionEngine", return_value=mock_engine_b):
            mm.get_or_load_vision_engine("model-b", device="GPU")

        mock_engine_a.unload.assert_called_once(), (
            "Evicted VisionEngine must have unload() called on it"
        )


class TestUnloadAllClearsAllCaches:
    """unload_all() must clear the embedding and vision caches in addition to the main LLM cache."""

    def test_unload_all_clears_embedding_cache_ok(self):
        """unload_all() must call unload() on and remove all EmbeddingEngine entries."""
        mm = ModelManager()
        mock_emb = mock.MagicMock()
        with mm._lock:
            mm._embedding_cache["emb-key"] = mock_emb

        mm.unload_all()

        assert len(mm._embedding_cache) == 0, (
            "unload_all() must clear the _embedding_cache"
        )
        mock_emb.unload.assert_called_once(), (
            "unload_all() must call unload() on each EmbeddingEngine"
        )

    def test_unload_all_clears_vision_cache_ok(self):
        """unload_all() must call unload() on and remove all VisionEngine entries."""
        mm = ModelManager()
        mock_vis = mock.MagicMock()
        with mm._lock:
            mm._vision_cache["vis-key"] = mock_vis

        mm.unload_all()

        assert len(mm._vision_cache) == 0, (
            "unload_all() must clear the _vision_cache"
        )
        mock_vis.unload.assert_called_once(), (
            "unload_all() must call unload() on each VisionEngine"
        )

    def test_unload_model_clears_embedding_and_vision_for_model_id_ok(self):
        """unload_model() must evict embedding and vision cache entries that start with model_id."""
        mm = ModelManager()
        mock_emb = mock.MagicMock()
        mock_vis = mock.MagicMock()
        with mm._lock:
            mm._embedding_cache["my-model:NPU"] = mock_emb
            mm._vision_cache["my-model:GPU"] = mock_vis

        evicted = mm.unload_model("my-model")

        assert evicted is True, "unload_model must return True when any engine was evicted"
        assert "my-model:NPU" not in mm._embedding_cache, (
            "unload_model must remove embedding cache entry for model_id"
        )
        assert "my-model:GPU" not in mm._vision_cache, (
            "unload_model must remove vision cache entry for model_id"
        )
