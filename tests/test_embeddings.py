"""Encoder / EmbeddingIndex tests — no real model load.

The lazy SentenceTransformer is monkeypatched away (`enc._model = stub`)
so we only verify prefix logic, dispatch, and matryoshka truncation.
Real-weight tests live in arxiv-radar-mcp's `--build-cache` integration.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from corpus_core.embeddings import (Encoder, EmbeddingIndex, _maybe_truncate,
                                    passage_prefix, query_prefix)


# ----- prefix registry ------------------------------------------------------

@pytest.mark.parametrize("model,expected", [
    ("mixedbread-ai/mxbai-embed-large-v1",
     "Represent this sentence for searching relevant passages: "),
    ("BAAI/bge-large-en-v1.5",
     "Represent this sentence for searching relevant passages: "),
    ("intfloat/e5-large-v2", "query: "),
    ("sentence-transformers/all-MiniLM-L6-v2", ""),  # unknown → no-op
    ("totally/made-up-model", ""),
])
def test_query_prefix_registry(model, expected):
    assert query_prefix(model) == expected


@pytest.mark.parametrize("model,expected", [
    ("intfloat/e5-large-v2", "passage: "),
    ("intfloat/e5-small-v2", "passage: "),
    ("mixedbread-ai/mxbai-embed-large-v1", ""),
    ("BAAI/bge-large-en-v1.5", ""),
])
def test_passage_prefix_registry(model, expected):
    assert passage_prefix(model) == expected


# ----- _maybe_truncate (matryoshka) -----------------------------------------

def test_maybe_truncate_keeps_native_when_target_none():
    vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    out = _maybe_truncate(vec, target_dim=None)
    assert np.array_equal(out, vec)


def test_maybe_truncate_truncates_and_renormalizes():
    vec = np.array([3.0, 4.0, 1.0, 1.0], dtype=np.float32)
    out = _maybe_truncate(vec, target_dim=2)
    assert out.shape == (2,)
    # L2 norm should be 1 (renormalized).
    assert np.linalg.norm(out) == pytest.approx(1.0, rel=1e-5)


def test_maybe_truncate_passthrough_when_target_geq_native():
    vec = np.array([0.6, 0.8], dtype=np.float32)  # already L2-normalized
    out = _maybe_truncate(vec, target_dim=4)
    # Asking for more dims than we have → returns unchanged.
    assert np.array_equal(out, vec)


def test_maybe_truncate_handles_2d_matrix():
    matrix = np.array([[3.0, 4.0, 1.0, 1.0],
                       [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    out = _maybe_truncate(matrix, target_dim=2)
    assert out.shape == (2, 2)
    norms = np.linalg.norm(out, axis=1)
    # First row truncates to (3,4) → norm 5 → renormalize → unit.
    assert norms[0] == pytest.approx(1.0, rel=1e-5)
    # Second row truncates to (0,0) → all-zero stays zero (no NaN from 0/0).
    assert not np.any(np.isnan(out))


# ----- Encoder dispatch (no real model load) -------------------------------

class _FakeST:
    """SentenceTransformer stand-in. Records what it was called with."""

    def __init__(self):
        self.last_call: list[str] | None = None
        self.last_kwargs: dict | None = None
        self.max_seq_length = 0

    def encode(self, texts, **kwargs) -> np.ndarray:
        self.last_call = list(texts)
        self.last_kwargs = kwargs
        return np.ones((len(texts), 4), dtype=np.float32)


def _patch(enc: Encoder, fake: _FakeST) -> None:
    enc._model = fake


def test_encoder_query_applies_mxbai_prefix(embeddings_config):
    embeddings_config.model = "mixedbread-ai/mxbai-embed-large-v1"
    enc = Encoder(_CfgWrapper(embeddings_config))
    fake = _FakeST()
    _patch(enc, fake)

    enc.encode_query("what is dft?")

    assert fake.last_call == [
        "Represent this sentence for searching relevant passages: what is dft?"
    ]


def test_encoder_passages_applies_e5_prefix(embeddings_config):
    embeddings_config.model = "intfloat/e5-large-v2"
    enc = Encoder(_CfgWrapper(embeddings_config))
    fake = _FakeST()
    _patch(enc, fake)

    enc.encode_passages(["dft basics", "mlip overview"], show_progress=False)

    assert fake.last_call == [
        "passage: dft basics",
        "passage: mlip overview",
    ]


def test_encoder_unknown_model_no_prefix(embeddings_config):
    embeddings_config.model = "totally/made-up"
    enc = Encoder(_CfgWrapper(embeddings_config))
    fake = _FakeST()
    _patch(enc, fake)

    enc.encode_query("hello")
    assert fake.last_call == ["hello"]


def test_encoder_target_dim_truncates(embeddings_config):
    embeddings_config.model = "totally/made-up"
    embeddings_config.target_dim = 2
    enc = Encoder(_CfgWrapper(embeddings_config))
    fake = _FakeST()
    _patch(enc, fake)

    out = enc.encode_query("x")
    assert out.shape == (2,)


# Wrap a bare _EmbeddingsCfg in something with `.embeddings.*` so Encoder's
# `config.embeddings.model` access path matches the duck-typed Protocol.
class _CfgWrapper:
    def __init__(self, embeddings):
        self.embeddings = embeddings


# ----- EmbeddingIndex roundtrip --------------------------------------------

def test_embedding_index_load_roundtrip(tmp_path):
    """EmbeddingIndex has no `save()` — corpus_index.reindex writes the
    on-disk format. Verify load() picks up a hand-rolled cache dir and
    surfaces matrix + row_for + model + metadata correctly."""
    matrix = np.random.RandomState(0).randn(5, 8).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)  # L2-normalized
    row_for = {f"id-{i}": i for i in range(5)}
    metadata_chunks = [{"arxiv_id": f"id-{i}", "section": "x", "chunk_idx": 0}
                       for i in range(5)]

    np.save(tmp_path / "embeddings.npy", matrix, allow_pickle=False)
    (tmp_path / "index.json").write_text(json.dumps({
        "model": "test/dummy",
        "dims": 8,
        "n": 5,
        "row_for": row_for,
        "chunks": metadata_chunks,
    }), encoding="utf-8")

    loaded = EmbeddingIndex.load(tmp_path)
    assert loaded.matrix.shape == (5, 8)
    assert loaded.model_name == "test/dummy"
    assert loaded.dims == 8
    assert loaded.row_for == row_for
    # Metadata is the index.json minus the core fields.
    assert (loaded.metadata or {}).get("chunks") == metadata_chunks


def test_embedding_index_load_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        EmbeddingIndex.load(tmp_path / "nope")
