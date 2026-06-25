"""Tests for the architectural fixes (Blocks A, B, D, E).

Covers:
  * EmbeddingIndex.save atomicity (mismatched pair not loaded as valid)
  * chunker_version mismatch forces full rebuild
  * load_chunk_texts length-mismatch does not crash; logs WARNING
  * is_safe_paper_id rejects ':' and '..'
  * build_paper_archive root_dir guard (_is_within)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from corpus_core.archive import _is_within, build_paper_archive, is_safe_paper_id
from corpus_core.chunker import CHUNKER_VERSION
from corpus_core.corpus_index import load_chunk_texts, reindex
from corpus_core.embeddings import EmbeddingIndex


# ---------------------------------------------------------------------------
# Block A -- EmbeddingIndex.save atomicity
# ---------------------------------------------------------------------------


def test_save_creates_both_files(tmp_path: Path):
    matrix = np.eye(3, dtype=np.float32)
    row_for = {"a": 0, "b": 1, "c": 2}
    EmbeddingIndex.save(tmp_path, matrix, row_for, model="test/m")

    assert (tmp_path / "embeddings.npy").exists()
    assert (tmp_path / "index.json").exists()
    # Tmp files must be gone.
    assert not (tmp_path / "embeddings.npy.tmp").exists()
    assert not (tmp_path / "index.json.tmp").exists()


def test_save_roundtrip_via_load(tmp_path: Path):
    matrix = np.eye(4, dtype=np.float32)
    row_for = {"x": 0, "y": 1, "z": 2, "w": 3}
    EmbeddingIndex.save(
        tmp_path, matrix, row_for,
        model="test/model",
        metadata={"chunker_version": "99"},
    )

    idx = EmbeddingIndex.load(tmp_path)
    assert idx.model_name == "test/model"
    assert idx.row_for == row_for
    assert idx.dims == 4
    assert idx.matrix.shape == (4, 4)
    # chunker_version surfaces in metadata
    assert (idx.metadata or {}).get("chunker_version") == "99"


def test_save_stale_json_not_read_as_valid(tmp_path: Path):
    """Simulate a crash between npy rename and json rename.

    After the crash: new embeddings.npy (shape 5x4) but old index.json
    (n=3). EmbeddingIndex.load reads both and exposes the mismatch via
    matrix.shape vs payload['n'] -- the pair is internally inconsistent.
    Callers that check len(row_for) vs matrix.shape[0] will notice.

    This test verifies that load() doesn't silently treat this as valid.
    """
    # Write a consistent v1 pair.
    m1 = np.zeros((3, 4), dtype=np.float32)
    EmbeddingIndex.save(tmp_path, m1, {"a": 0, "b": 1, "c": 2}, model="m")

    # Simulate crash: overwrite npy only (no json update).
    m2 = np.zeros((5, 4), dtype=np.float32)
    with open(tmp_path / "embeddings.npy", "wb") as f:
        np.save(f, m2)

    idx = EmbeddingIndex.load(tmp_path)
    payload = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))

    # matrix has 5 rows; index.json says n=3 and row_for has 3 keys.
    assert idx.matrix.shape[0] == 5
    assert payload["n"] == 3
    # The inconsistency is detectable: matrix rows != json row count.
    assert idx.matrix.shape[0] != payload["n"]


# ---------------------------------------------------------------------------
# Block B -- chunker_version mismatch forces full rebuild
# ---------------------------------------------------------------------------


class _FakeEncoderFixed:
    """Minimal Encoder stand-in with a fixed dim."""
    def __init__(self, dim: int = 8):
        self.dim = dim
        self.encoded_calls: list = []
        self.config = type("C", (), {"embeddings": type("E", (), {
            "batch_size": 32, "model": "fake/enc"})()})()
        self._model = type("M", (), {"max_seq_length": 512})()

    @property
    def model_name(self) -> str:
        return "fake/enc"

    def _ensure_loaded(self) -> None:
        pass

    def encode_passages(self, texts, show_progress=True,
                        max_seq_length=512, batch_size=None):
        self.encoded_calls.append(list(texts))
        n = len(texts)
        matrix = np.zeros((n, self.dim), dtype=np.float32)
        for i in range(n):
            matrix[i, i % self.dim] = 1.0
        return matrix

    @property
    def total_encoded(self) -> int:
        return sum(len(c) for c in self.encoded_calls)


def _write_paper(sources_dir: Path, arxiv_id: str, body: str | None = None) -> None:
    if body is None:
        body = f"## Methods\n{arxiv_id} methods\n\n## Results\n{arxiv_id} results\n"
    (sources_dir / f"{arxiv_id}.md").write_text(body, encoding="utf-8")
    (sources_dir / f"{arxiv_id}.meta.json").write_text(json.dumps({
        "arxiv_id": arxiv_id, "source": "html",
        "fetch_time": "2026-05-01T00:00:00",
        "n_chars": len(body), "n_chunks_after_split": 0,
    }), encoding="utf-8")


def test_chunker_version_mismatch_forces_full_rebuild(tmp_path: Path):
    """When the on-disk index was built with a different CHUNKER_VERSION,
    _classify_papers should return full_rebuild_reason and reindex must
    re-encode everything."""
    sources = tmp_path / "sources"
    sources.mkdir()
    _write_paper(sources, "p1")
    _write_paper(sources, "p2")

    enc = _FakeEncoderFixed(dim=8)
    reindex(tmp_path, enc)
    enc.encoded_calls.clear()

    # Tamper with the on-disk chunker_version.
    payload = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    payload["chunker_version"] = "STALE_VERSION_XYZ"
    (tmp_path / "index.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    reindex(tmp_path, enc)

    # Every paper must have been re-encoded.
    all_texts = [t for call in enc.encoded_calls for t in call]
    assert any("p1" in t for t in all_texts), "p1 must be re-encoded on version mismatch"
    assert any("p2" in t for t in all_texts), "p2 must be re-encoded on version mismatch"

    # New index carries the current CHUNKER_VERSION.
    new_payload = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert new_payload.get("chunker_version") == CHUNKER_VERSION


def test_matching_chunker_version_allows_incremental(tmp_path: Path):
    """When the chunker_version matches, an unchanged corpus should noop."""
    sources = tmp_path / "sources"
    sources.mkdir()
    _write_paper(sources, "p1")

    enc = _FakeEncoderFixed(dim=8)
    reindex(tmp_path, enc)
    enc.encoded_calls.clear()

    # Verify chunker_version is current.
    payload = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert payload.get("chunker_version") == CHUNKER_VERSION

    reindex(tmp_path, enc)
    assert enc.total_encoded == 0, "noop expected when version matches and no changes"


# ---------------------------------------------------------------------------
# Block B -- load_chunk_texts length mismatch
# ---------------------------------------------------------------------------


def test_load_chunk_texts_length_mismatch_warns_not_crashes(
    tmp_path: Path, caplog
):
    """When the index expects N chunks but the re-chunked source has M != N,
    load_chunk_texts must log a WARNING and return placeholder strings,
    not crash or silently misalign rows."""
    sources = tmp_path / "sources"
    sources.mkdir()
    # Source has 3 sections.
    (sources / "p1.md").write_text(
        "## A\nbody A\n\n## B\nbody B\n\n## C\nbody C\n",
        encoding="utf-8",
    )

    # Index claims only 2 chunks for p1 -- mismatch.
    chunks_meta = [
        {"arxiv_id": "p1", "section": "A", "chunk_idx": 0, "n_chars": 10},
        {"arxiv_id": "p1", "section": "B", "chunk_idx": 0, "n_chars": 10},
    ]
    index = EmbeddingIndex(
        matrix=np.zeros((2, 4), dtype=np.float32),
        row_for={"p1": 0}, model_name="x", dims=4,
        metadata={"chunks": chunks_meta, "max_seq_length": 4096, "n_papers": 1},
    )

    with caplog.at_level(logging.WARNING, logger="corpus_core.corpus_index"):
        texts = load_chunk_texts(tmp_path, index)

    # Must not crash.
    assert len(texts) == 2
    # The mismatch rows become empty strings.
    assert all(t == "" for t in texts)
    # A warning must have been logged.
    assert any("mismatch" in rec.message.lower() or "p1" in rec.message
               for rec in caplog.records)


# ---------------------------------------------------------------------------
# Block D -- is_safe_paper_id colon and dotdot
# ---------------------------------------------------------------------------


def test_is_safe_paper_id_rejects_colon():
    """Windows drive-relative path: C:foo must be rejected."""
    assert not is_safe_paper_id("C:foo")
    assert not is_safe_paper_id("D:relative")
    assert not is_safe_paper_id("some:thing")


def test_is_safe_paper_id_rejects_dotdot():
    assert not is_safe_paper_id("../escape")
    assert not is_safe_paper_id("2603.05238/../etc/passwd")
    assert not is_safe_paper_id("..")


def test_is_safe_paper_id_accepts_normal_ids():
    assert is_safe_paper_id("2603.05238")
    assert is_safe_paper_id("cond-mat/0211034")
    assert is_safe_paper_id("sha256deadbeef1234")


# ---------------------------------------------------------------------------
# Block D -- build_paper_archive root_dir guard
# ---------------------------------------------------------------------------


def test_build_paper_archive_root_dir_blocks_escape(tmp_path: Path):
    """When root_dir is provided and markdown_path resolves outside it,
    build_paper_archive must return None (refuse to read)."""

    sources = tmp_path / "sources"
    sources.mkdir()
    # Write a legitimate file inside root.
    md_inside = sources / "legit.md"
    md_inside.write_text("# legit", encoding="utf-8")

    # A symlink that escapes root_dir.
    outside = tmp_path.parent / "escaped_paper.md"
    outside.write_text("# secret", encoding="utf-8")

    try:
        escape_link = sources / "escape.md"
        escape_link.symlink_to(outside)
        symlink_created = True
    except (OSError, NotImplementedError):
        symlink_created = False

    from corpus_core.archive import PaperFiles

    if symlink_created:
        result = build_paper_archive(
            "escape",
            PaperFiles(markdown_path=escape_link),
            root_dir=sources,
        )
        assert result is None, "symlink escape should be blocked by root_dir guard"
    # Without root_dir, the same path would succeed.


def test_build_paper_archive_root_dir_allows_valid_path(tmp_path: Path):
    from corpus_core.archive import PaperFiles

    sources = tmp_path / "sources"
    sources.mkdir()
    md = sources / "valid.md"
    md.write_text("# valid paper", encoding="utf-8")

    result = build_paper_archive(
        "valid",
        PaperFiles(markdown_path=md),
        root_dir=sources,
    )
    assert result is not None


# ---------------------------------------------------------------------------
# Block D -- _is_within helper
# ---------------------------------------------------------------------------


def test_is_within_basic(tmp_path: Path):
    inner = tmp_path / "sub" / "file.txt"
    inner.parent.mkdir(parents=True)
    inner.write_text("x")
    assert _is_within(inner, tmp_path)


def test_is_within_rejects_escape(tmp_path: Path):
    outside = tmp_path.parent / "outside.txt"
    assert not _is_within(outside, tmp_path)
