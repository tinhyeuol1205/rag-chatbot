"""Regression tests cho cold-start concurrency (bug P2-14).

Trước fix: get_retriever/get_llm_service/embedding/reranker dùng check-then-create
không lock → 2 request đầu cùng thấy None rồi cùng khởi tạo model (RAM nhân đôi
hoặc OOM lúc deploy). Fix: lock + double-checked — single-flight.

Fixtures reset globals sau test để không phụ thuộc thứ tự chạy suite.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import api.chat as chat_module
import core.llm as llm_module
import ingestion.embeddings as embeddings_module
import retrieval.reranking.cross_encoder as reranker_module


def _fire(loader_fn, n_threads: int = 20, n_calls: int = 100):
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        return list(pool.map(lambda _: loader_fn(), range(n_calls)))


class _SlowFake:
    """Giả lập construction chậm — mở rộng cửa sổ race để test chắc ăn."""

    _created = 0
    _count_lock = Lock()

    def __init__(self, *args, **kwargs):
        time.sleep(0.01)  # Widen race window — pattern cũ sẽ fail chắc chắn
        with _SlowFake._count_lock:
            _SlowFake._created += 1


def test_retriever_is_single_flight(monkeypatch):
    _SlowFake._created = 0
    monkeypatch.setattr(chat_module, "RAGRetriever", _SlowFake)
    monkeypatch.setattr(chat_module, "_retriever", None)

    instances = _fire(chat_module.get_retriever)

    assert _SlowFake._created == 1
    assert len({id(inst) for inst in instances}) == 1


def test_llm_service_is_single_flight(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "LLM_PROVIDER", "openai")
    _SlowFake._created = 0
    monkeypatch.setattr(llm_module, "OpenAILLMService", _SlowFake)
    monkeypatch.setattr(llm_module, "_llm_instance", None)

    instances = _fire(llm_module.get_llm_service)

    assert _SlowFake._created == 1
    assert len({id(inst) for inst in instances}) == 1


def test_embedding_model_loads_once(monkeypatch):
    _SlowFake._created = 0
    monkeypatch.setattr(embeddings_module, "SentenceTransformer", _SlowFake)
    monkeypatch.setattr(embeddings_module, "_embedding_model", None)

    instances = _fire(embeddings_module._load_model)

    assert _SlowFake._created == 1
    assert len({id(inst) for inst in instances}) == 1


def test_reranker_model_loads_once(monkeypatch):
    _SlowFake._created = 0
    monkeypatch.setattr(reranker_module, "CrossEncoder", _SlowFake)
    monkeypatch.setattr(reranker_module, "_reranker_model", None)

    instances = _fire(reranker_module._load_reranker)

    assert _SlowFake._created == 1
    assert len({id(inst) for inst in instances}) == 1
