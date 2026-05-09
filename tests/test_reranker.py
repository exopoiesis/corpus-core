"""Reranker tests — config + lazy load + score-and-sort logic.

The CrossEncoder is monkeypatched so we don't pull weights at test time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from corpus_core.reranker import Reranker, RerankerConfig


def test_config_defaults():
    cfg = RerankerConfig()
    assert cfg.enabled is True
    assert cfg.model == "BAAI/bge-reranker-base"
    assert cfg.top_k_candidates == 50


def test_config_overrides():
    cfg = RerankerConfig(enabled=False, model="custom", top_k_candidates=10)
    assert cfg.enabled is False
    assert cfg.model == "custom"
    assert cfg.top_k_candidates == 10


def test_reranker_lazy_load_not_triggered_until_use():
    cfg = RerankerConfig()
    r = Reranker(cfg)
    assert r._model is None  # not loaded just by constructing


def test_reranker_model_name_property():
    cfg = RerankerConfig(model="some/cross-encoder")
    r = Reranker(cfg)
    assert r.model_name == "some/cross-encoder"


# ----- rerank() logic with stubbed CrossEncoder -----------------------------

@dataclass
class _StubPaper:
    """Implements the _RerankablePaper Protocol (`.search_text`)."""
    paper_id: str
    search_text: str


class _StubCrossEncoder:
    """Returns the score the test asks for via the predict scores list."""

    def __init__(self, scores):
        self._scores = scores

    def predict(self, pairs, show_progress_bar=False):
        # Caller passes (query, search_text) tuples in the same order the
        # papers came in; we hand back a parallel score array.
        return np.array(self._scores, dtype=np.float32)


def test_rerank_orders_by_score_desc():
    cfg = RerankerConfig()
    r = Reranker(cfg)
    r._model = _StubCrossEncoder(scores=[0.1, 0.9, 0.5])

    candidates = [
        _StubPaper("a", "first"),
        _StubPaper("b", "second"),
        _StubPaper("c", "third"),
    ]
    out = r.rerank("query", candidates, k=3)

    ids = [p.paper_id for p, _ in out]
    assert ids == ["b", "c", "a"]


def test_rerank_respects_k():
    cfg = RerankerConfig()
    r = Reranker(cfg)
    r._model = _StubCrossEncoder(scores=[0.3, 0.9, 0.5])

    candidates = [_StubPaper(f"p{i}", f"txt{i}") for i in range(3)]
    out = r.rerank("query", candidates, k=2)
    assert len(out) == 2
    # Top two by score: 0.9 → p1, 0.5 → p2.
    assert [p.paper_id for p, _ in out] == ["p1", "p2"]


def test_rerank_empty_candidates_returns_empty():
    r = Reranker(RerankerConfig())
    # _model intentionally not set — empty input must short-circuit
    # before any model interaction.
    assert r.rerank("any query", [], k=10) == []


def test_rerank_calls_model_with_query_and_search_text():
    cfg = RerankerConfig()
    r = Reranker(cfg)

    captured_pairs = []

    class _Spy(_StubCrossEncoder):
        def predict(self, pairs, show_progress_bar=False):
            captured_pairs.extend(pairs)
            return np.zeros(len(pairs), dtype=np.float32)

    r._model = _Spy(scores=[])
    candidates = [_StubPaper("a", "alpha body"), _StubPaper("b", "beta body")]
    r.rerank("dft methods", candidates, k=2)

    assert captured_pairs == [
        ("dft methods", "alpha body"),
        ("dft methods", "beta body"),
    ]


def test_rerank_returns_float_scores_not_numpy():
    """JSON-friendly: the caller serializes results, so scores must be
    Python floats not numpy.float32."""
    cfg = RerankerConfig()
    r = Reranker(cfg)
    r._model = _StubCrossEncoder(scores=[0.5])
    out = r.rerank("q", [_StubPaper("a", "t")], k=1)
    _, score = out[0]
    assert type(score) is float
