"""Cross-encoder reranker.

Cross-encoders score (query, passage) pairs jointly — they're slower than
bi-encoders (no caching, must re-run per pair) but the precision boost
over dense retrieval is large (≈ +5–15 nDCG@10 in published benchmarks).

We use them in `search_hybrid` as a final pass: pull top-K from RRF,
rerank, return top-k. Lazy-loaded so unit tests and the --build-cache
path don't pay the model-download cost.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

LOG = logging.getLogger(__name__)


@dataclass
class RerankerConfig:
    """Standalone reranker config — lives in corpus_core so the package
    has no upward dependency on a particular host project.

    Downstream projects can either use this directly or wrap it in a
    larger config dataclass and pass `cfg.reranker` here.
    """
    enabled: bool = True
    model: str = "BAAI/bge-reranker-base"
    top_k_candidates: int = 50  # how many hybrid candidates feed into the cross-encoder


class _RerankablePaper(Protocol):
    """Duck-typed view of a Paper used for reranking — anything with a
    `search_text` attribute (string the cross-encoder scores against the
    query). Both `arxiv_radar_mcp.corpus.Paper` and any custom record
    type that exposes the same attribute work."""

    @property
    def search_text(self) -> str: ...


class Reranker:
    """Lazy CrossEncoder wrapper. One instance per server lifetime."""

    def __init__(self, config: RerankerConfig) -> None:
        self.config = config
        self._model = None  # type: ignore[var-annotated]

    @property
    def model_name(self) -> str:
        return self.config.model

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            LOG.info(f"loading cross-encoder {self.model_name}...")
            self._model = CrossEncoder(self.model_name)

    def unload(self) -> bool:
        """Drop the underlying CrossEncoder + free GPU/host memory.

        Idempotent — returns True if a model was released, False if
        nothing was loaded. The next `rerank` call lazily re-loads.

        Mirrors `corpus_core.embeddings.Encoder.unload()` so callers
        that hold both can release them together after a heavy pass.
        """
        import gc

        if self._model is None:
            return False
        model = self._model
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                try:
                    # CrossEncoder wraps a HF model under `.model`.
                    inner = getattr(model, "model", None)
                    if inner is not None:
                        inner.to("cpu")
                except Exception as e:  # noqa: BLE001
                    LOG.debug(f"unload: cross-encoder.to('cpu') ignored: {e}")
        except ImportError:
            pass

        del model
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass

        LOG.info(f"unloaded cross-encoder {self.model_name}")
        return True

    def rerank(
        self, query: str, candidates: list[_RerankablePaper], k: int,
    ) -> list[tuple[_RerankablePaper, float]]:
        """Re-score (query, paper.search_text) pairs and return top-k by score."""
        if not candidates:
            return []
        self._ensure_loaded()
        pairs = [(query, p.search_text) for p in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)  # type: ignore[union-attr]
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        return [(p, float(s)) for p, s in ranked[:k]]
