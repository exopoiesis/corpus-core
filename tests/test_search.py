"""Search primitives — exercised with a Protocol-conforming stub Paper.

Phase 3 invariant: corpus_core.search must NOT touch
arxiv_radar_mcp.corpus.Paper. The `_StubPaper` here proves the
Protocol duck-typing works against any host record schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from corpus_core.embeddings import EmbeddingIndex
from corpus_core.search import (search_hybrid, search_semantic, search_text,
                                similar_to)


@dataclass
class _StubPaper:
    """Implements the `_PaperLike` Protocol from corpus_core.search."""
    arxiv_id: str
    title: str
    abstract: str
    tags: list[str] = field(default_factory=list)
    domain: str = "test"


@pytest.fixture
def papers() -> list[_StubPaper]:
    return [
        _StubPaper("p001",
                   title="Density functional theory of mackinawite",
                   abstract="We compute lattice parameters of FeS via DFT.",
                   tags=["dft", "ab-initio"], domain="chemistry"),
        _StubPaper("p002",
                   title="MLIP for organic crystals",
                   abstract="Equivariant graph neural network potential.",
                   tags=["mlip", "gnn"], domain="chemistry"),
        _StubPaper("p003",
                   title="Generative model for catalysts",
                   abstract="Diffusion model proposes new transition-metal complexes.",
                   tags=["generative-model", "catalysis"], domain="chemistry,physics"),
    ]


@pytest.fixture
def papers_dict(papers) -> dict[str, _StubPaper]:
    return {p.arxiv_id: p for p in papers}


# ----- text search ----------------------------------------------------------

def test_search_text_finds_dft(papers):
    out = search_text(papers, "dft", k=5)
    ids = [p.arxiv_id for p, _ in out]
    assert ids == ["p001"]


def test_search_text_title_boosted_over_abstract(papers):
    """Token in title should rank higher than only-in-abstract."""
    out = search_text(papers, "model", k=5)
    # 'Generative model for catalysts' has it in title (boosted),
    # 'Diffusion model proposes...' is also in abstract of p003 — same
    # paper. Ensure non-empty result.
    assert any(p.arxiv_id == "p003" for p, _ in out)


def test_search_text_and_semantics(papers):
    """Multi-token query must be AND across tokens."""
    out = search_text(papers, "dft mackinawite", k=5)
    ids = [p.arxiv_id for p, _ in out]
    assert ids == ["p001"]


def test_search_text_no_match(papers):
    assert search_text(papers, "completely-unrelated-token", k=5) == []


def test_search_text_filters_by_domain(papers):
    # 'physics' only matches p003 (it's in the comma-split list).
    out = search_text(papers, "model", k=5, domain="physics")
    ids = [p.arxiv_id for p, _ in out]
    assert ids == ["p003"]


def test_search_text_filters_by_tag(papers):
    out = search_text(papers, "model", k=5, tag="generative-model")
    ids = [p.arxiv_id for p, _ in out]
    assert ids == ["p003"]


def test_search_text_empty_query(papers):
    assert search_text(papers, "   ", k=5) == []


def test_search_text_respects_k(papers):
    # All three contain a vowel — but k caps the result.
    out = search_text(papers, "a", k=2)
    assert len(out) <= 2


# ----- semantic + similar_to ------------------------------------------------

def _make_index(papers, vectors) -> EmbeddingIndex:
    matrix = np.asarray(vectors, dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    row_for = {p.arxiv_id: i for i, p in enumerate(papers)}
    return EmbeddingIndex(matrix=matrix, row_for=row_for,
                          model_name="test/dummy", dims=matrix.shape[1],
                          metadata={})


def test_search_semantic_returns_nearest(papers, papers_dict):
    """Crafted vectors so p002 is the closest match to query."""
    vectors = [
        [1.0, 0.0, 0.0],   # p001
        [0.9, 0.1, 0.0],   # p002 — closest to query [1, 0, 0]
        [0.0, 1.0, 0.0],   # p003
    ]
    idx = _make_index(papers, vectors)
    qvec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    out = search_semantic(papers_dict, qvec, idx, k=2)
    ids = [p.arxiv_id for p, _ in out]
    assert ids[0] == "p001"  # exact match wins
    assert ids[1] == "p002"


def test_search_semantic_filter_excludes_paper(papers, papers_dict):
    vectors = [[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0]]
    idx = _make_index(papers, vectors)
    qvec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Restrict to physics → only p003.
    out = search_semantic(papers_dict, qvec, idx, k=5, domain="physics")
    ids = [p.arxiv_id for p, _ in out]
    assert ids == ["p003"]


def test_similar_to_excludes_self(papers, papers_dict):
    vectors = [[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0]]
    idx = _make_index(papers, vectors)

    out = similar_to(papers_dict, "p001", idx, k=2)
    ids = [p.arxiv_id for p, _ in out]
    assert "p001" not in ids
    # p002 should rank above p003 because it's closer in cosine.
    assert ids[0] == "p002"


def test_similar_to_unknown_id(papers_dict):
    matrix = np.eye(3, dtype=np.float32)
    idx = EmbeddingIndex(matrix=matrix, row_for={"p001": 0, "p002": 1, "p003": 2},
                         model_name="test/dummy", dims=3, metadata={})
    out = similar_to(papers_dict, "phantom", idx, k=5)
    assert out == []


# ----- hybrid (RRF) ---------------------------------------------------------

def test_search_hybrid_rrf_combines_lists(papers):
    """When a paper appears in both lists, its rank score should add up."""
    text_results = [(papers[0], 1.0), (papers[1], 0.5)]
    sem_results = [(papers[1], 0.9), (papers[2], 0.7)]
    out = search_hybrid(text_results, sem_results, k=3, rrf_k=60)
    ids = [p.arxiv_id for p, _ in out]
    # p002 in BOTH → top.
    assert ids[0] == "p002"


def test_search_hybrid_respects_k(papers):
    text_results = [(papers[0], 1.0), (papers[1], 0.5), (papers[2], 0.2)]
    sem_results = []
    out = search_hybrid(text_results, sem_results, k=2)
    assert len(out) == 2
