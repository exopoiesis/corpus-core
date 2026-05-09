"""Search primitives: text, semantic, hybrid (RRF), similar-to.

All functions take an iterable of Paper-shaped records (already loaded
in memory) and return a list of (paper, score) tuples sorted desc by
score, top-k.

The `Paper` type is duck-typed via `_PaperLike` Protocol — any object
with `.arxiv_id`, `.title`, `.abstract`, `.tags`, `.domain` works, so
corpus_core stays decoupled from any one host project's record schema.

Filtering by domain / tag is applied BEFORE ranking — no point ranking
records the caller doesn't want.
"""
from __future__ import annotations

import re
from typing import Iterable, Protocol

import numpy as np

from corpus_core.embeddings import EmbeddingIndex


class _PaperLike(Protocol):
    """Minimal Paper interface used by the search primitives. Implemented
    by `arxiv_radar_mcp.corpus.Paper` and any compatible host record."""
    arxiv_id: str
    title: str
    abstract: str
    tags: list[str]
    domain: str


# Backwards-compat alias for type hints downstream — older callers used
# `from corpus_core.search import Paper`. New code should use
# `_PaperLike` or its own concrete type.
Paper = _PaperLike


def _filter(papers: Iterable[_PaperLike],
            domain: str | None = None,
            tag: str | None = None) -> list[_PaperLike]:
    out = []
    for p in papers:
        if domain and domain not in p.domain.split(","):
            continue
        if tag and tag not in p.tags:
            continue
        out.append(p)
    return out


# ----- text search -----------------------------------------------------------

def search_text(papers: Iterable[_PaperLike], query: str, k: int = 10,
                domain: str | None = None, tag: str | None = None
                ) -> list[tuple[_PaperLike, float]]:
    """Naive multi-token AND with title-boost.

    For MVP this is whole-corpus substring scan — fine up to a few hundred
    thousand papers. For larger corpora swap in BM25 (rank_bm25 lib) keeping
    the same return shape.
    """
    pool = _filter(papers, domain=domain, tag=tag)
    tokens = [t for t in re.split(r"\s+", query.lower().strip()) if t]
    if not tokens:
        return []

    scored: list[tuple[_PaperLike, float]] = []
    for p in pool:
        title_l = p.title.lower()
        abstract_l = p.abstract.lower()
        score = 0.0
        all_match = True
        for t in tokens:
            in_title = t in title_l
            in_abstract = t in abstract_l
            if not (in_title or in_abstract):
                all_match = False
                break
            score += 3.0 if in_title else 1.0
        if all_match:
            scored.append((p, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


# ----- semantic search -------------------------------------------------------

def search_semantic(papers: dict[str, _PaperLike], query_vec: np.ndarray,
                    index: EmbeddingIndex, k: int = 10,
                    domain: str | None = None, tag: str | None = None
                    ) -> list[tuple[_PaperLike, float]]:
    """Cosine similarity in embedding space. query_vec must be L2-normalized."""
    # Cosine = dot product because matrix is already L2-normalized.
    sims = index.matrix @ query_vec  # (N,)

    # Build mask of allowed rows after domain/tag filter, then take top-k of those.
    pool_ids = {p.arxiv_id for p in _filter(papers.values(), domain=domain, tag=tag)}
    if not pool_ids:
        return []
    rows = np.array([index.row_for[pid] for pid in pool_ids
                     if pid in index.row_for], dtype=np.int64)
    if rows.size == 0:
        return []
    candidate_sims = sims[rows]
    top = np.argpartition(-candidate_sims, min(k, len(rows) - 1))[:k]
    top_sorted = top[np.argsort(-candidate_sims[top])]

    id_for_row = {i: pid for pid, i in index.row_for.items()}
    return [(papers[id_for_row[int(rows[idx])]], float(candidate_sims[idx]))
            for idx in top_sorted]


# ----- hybrid (RRF) ----------------------------------------------------------

def search_hybrid(text_results: list[tuple[_PaperLike, float]],
                  semantic_results: list[tuple[_PaperLike, float]],
                  k: int = 10, rrf_k: int = 60) -> list[tuple[_PaperLike, float]]:
    """Reciprocal Rank Fusion: score(p) = Σ_lists 1 / (rrf_k + rank).

    Robust to score-scale mismatch between text and semantic. rrf_k=60 is
    the canonical default from the original RRF paper.
    """
    rank_text = {p.arxiv_id: i for i, (p, _) in enumerate(text_results)}
    rank_sem  = {p.arxiv_id: i for i, (p, _) in enumerate(semantic_results)}

    by_id: dict[str, _PaperLike] = {p.arxiv_id: p for p, _ in text_results}
    by_id.update({p.arxiv_id: p for p, _ in semantic_results})

    fused: list[tuple[_PaperLike, float]] = []
    for pid, p in by_id.items():
        score = 0.0
        if pid in rank_text:
            score += 1.0 / (rrf_k + rank_text[pid])
        if pid in rank_sem:
            score += 1.0 / (rrf_k + rank_sem[pid])
        fused.append((p, score))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:k]


# ----- similar-to ------------------------------------------------------------

def similar_to(papers: dict[str, _PaperLike], arxiv_id: str,
               index: EmbeddingIndex, k: int = 10
               ) -> list[tuple[_PaperLike, float]]:
    """Nearest neighbours of a known paper. Self-match excluded."""
    vec = index.vector(arxiv_id)
    if vec is None:
        return []
    sims = index.matrix @ vec
    own_row = index.row_for[arxiv_id]
    sims[own_row] = -1.0  # exclude self
    top = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
    top_sorted = top[np.argsort(-sims[top])]
    id_for_row = {i: pid for pid, i in index.row_for.items()}
    return [(papers[id_for_row[int(r)]], float(sims[r])) for r in top_sorted
            if id_for_row[int(r)] in papers]
