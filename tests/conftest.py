"""Shared fixtures for corpus-core's standalone test suite.

corpus-core duck-types its consumers' Config — `Encoder.__init__(cfg)`
reads `cfg.embeddings.{model, batch_size, target_dim, cache_dir}`.
The `embeddings_config` / `corpus_config` fixtures here provide a
minimal stand-in so tests don't need arxiv-radar-mcp's full schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest


@dataclass
class _EmbeddingsCfg:
    model: str = "test/dummy"
    cache_dir: Path = field(default_factory=lambda: Path("./.tmp-cache"))
    batch_size: int = 4
    target_dim: int | None = None


@dataclass
class _CorpusCfg:
    """Minimal duck-typed Config — anything Encoder reads off `cfg.*`."""
    embeddings: _EmbeddingsCfg = field(default_factory=_EmbeddingsCfg)


@pytest.fixture
def embeddings_config(tmp_path: Path) -> _EmbeddingsCfg:
    """Embeddings-only stand-in. Tests that build Encoder directly use this."""
    return _EmbeddingsCfg(cache_dir=tmp_path / "cache")


@pytest.fixture
def corpus_config(tmp_path: Path) -> _CorpusCfg:
    """Full corpus_core duck-typed Config rooted in tmp_path."""
    return _CorpusCfg(embeddings=_EmbeddingsCfg(cache_dir=tmp_path / "cache"))
