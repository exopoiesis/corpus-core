"""Unit tests for Group A architectural fixes.

Covers:
  Fix #2 -- stale reindex-lock: NOT removed when owner pid alive; removed when dead.
  Fix #3 -- load_chunk_texts fills stats["stale_papers"]; mismatch logs ERROR.
  Fix #6 -- _persist_index delegates to EmbeddingIndex.save; roundtrip loads OK.
  Fix #8 -- _call_tool wraps tool exceptions in {"error":...}, session stays alive.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from corpus_core.corpus_index import _persist_index, load_chunk_texts
from corpus_core.embeddings import EmbeddingIndex
from corpus_core.jobs import (
    JobRegistry,
    _parse_lock_file,
    _pid_is_alive,
)
from corpus_core.mcp_scaffold import build_mcp_app, make_method_dispatcher


# ---------------------------------------------------------------------------
# Fix #2 -- stale reindex-lock liveness checks
# ---------------------------------------------------------------------------


def _write_lock(path: Path, pid: int, hostname: str | None = None) -> None:
    """Write a properly-formatted new-style lockfile."""
    if hostname is None:
        hostname = socket.gethostname()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"pid={pid}\nhostname={hostname}\nstart_time=2026-06-25T10:00:00+00:00\n",
        encoding="utf-8",
    )


def test_stale_lock_not_removed_when_owner_pid_alive(tmp_path: Path):
    """Lockfile must NOT be deleted when the recorded pid is still alive."""
    fulltext = tmp_path / "fulltext"
    lock_path = fulltext / ".reindex.lock"
    # Write a lock with OUR own pid -- we are definitely alive.
    _write_lock(lock_path, pid=os.getpid())

    reg = JobRegistry(cache_dir=tmp_path)
    # Lock must survive init because pid is alive.
    assert lock_path.exists(), "lock must remain when owner is alive"
    reg.shutdown()


def test_stale_lock_removed_when_owner_pid_dead(tmp_path: Path):
    """Lockfile MUST be deleted when the recorded pid is provably dead."""
    fulltext = tmp_path / "fulltext"
    lock_path = fulltext / ".reindex.lock"
    # Use pid=1 on all platforms; on non-init-namespace Linux it's alive but
    # we want a truly dead pid. Use a pid that cannot exist: on POSIX, os.getpid()
    # + 1_000_000 is virtually guaranteed dead; wrap in _pid_is_alive check.
    # For a deterministic test, mock _pid_is_alive to return False.
    _write_lock(lock_path, pid=99999999)

    with patch("corpus_core.jobs._pid_is_alive", return_value=False):
        reg = JobRegistry(cache_dir=tmp_path)

    assert not lock_path.exists(), "stale lock (dead pid) must be removed on init"
    reg.shutdown()


def test_stale_lock_not_removed_when_foreign_host(tmp_path: Path, caplog):
    """Lockfile from a different hostname must NOT be deleted (cannot verify pid)."""
    fulltext = tmp_path / "fulltext"
    lock_path = fulltext / ".reindex.lock"
    _write_lock(lock_path, pid=12345, hostname="other-machine.example.com")

    with caplog.at_level(logging.WARNING, logger="corpus_core.jobs"):
        reg = JobRegistry(cache_dir=tmp_path)

    assert lock_path.exists(), "foreign-host lock must not be removed"
    # A warning must be logged.
    assert any("foreign" in r.message.lower() or "other-machine" in r.message
               for r in caplog.records)
    reg.shutdown()


def test_stale_lock_unparseable_format_removed(tmp_path: Path):
    """Lockfile in old/unknown format (e.g. just 'stale') must be removed."""
    fulltext = tmp_path / "fulltext"
    fulltext.mkdir()
    lock_path = fulltext / ".reindex.lock"
    lock_path.write_text("stale", encoding="utf-8")

    reg = JobRegistry(cache_dir=tmp_path)
    assert not lock_path.exists(), "unparseable lock must be removed"
    reg.shutdown()


def test_parse_lock_file_valid(tmp_path: Path):
    lock = tmp_path / ".reindex.lock"
    _write_lock(lock, pid=42)
    info = _parse_lock_file(lock)
    assert info is not None
    assert info["pid"] == "42"
    assert info["hostname"] == socket.gethostname()


def test_parse_lock_file_missing(tmp_path: Path):
    assert _parse_lock_file(tmp_path / "nonexistent") is None


def test_parse_lock_file_old_format(tmp_path: Path):
    lock = tmp_path / ".reindex.lock"
    lock.write_text("pid=12345 at=2026-01-01\n", encoding="utf-8")
    # Old single-line format has no hostname key -> should return None.
    result = _parse_lock_file(lock)
    assert result is None


def test_pid_is_alive_self():
    """Our own pid must always be alive."""
    assert _pid_is_alive(os.getpid()) is True


def test_pid_is_alive_invalid():
    """Clearly invalid pid values should return False."""
    assert _pid_is_alive(-1) is False
    assert _pid_is_alive(0) is False


# ---------------------------------------------------------------------------
# Fix #3 -- load_chunk_texts: stats["stale_papers"] + ERROR log
# ---------------------------------------------------------------------------


def test_load_chunk_texts_stats_on_mismatch(tmp_path: Path, caplog):
    """When chunk count mismatches, stats['stale_papers'] must contain the id
    and an ERROR must be logged."""
    sources = tmp_path / "sources"
    sources.mkdir()
    # Source that will produce 3 chunks (3 sections).
    (sources / "p1.md").write_text(
        "## A\nbody A\n\n## B\nbody B\n\n## C\nbody C\n",
        encoding="utf-8",
    )

    # Index that claims only 2 chunks for p1.
    chunks_meta = [
        {"arxiv_id": "p1", "section": "A", "chunk_idx": 0, "n_chars": 10},
        {"arxiv_id": "p1", "section": "B", "chunk_idx": 0, "n_chars": 10},
    ]
    index = EmbeddingIndex(
        matrix=np.zeros((2, 4), dtype=np.float32),
        row_for={"p1": 0},
        model_name="x",
        dims=4,
        metadata={"chunks": chunks_meta, "max_seq_length": 4096, "n_papers": 1},
    )

    stats: dict = {}
    with caplog.at_level(logging.ERROR, logger="corpus_core.corpus_index"):
        texts = load_chunk_texts(tmp_path, index, stats=stats)

    assert "stale_papers" in stats
    assert "p1" in stats["stale_papers"]
    # Rows become empty strings.
    assert all(t == "" for t in texts)
    # An ERROR must have been logged (not just WARNING).
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


def test_load_chunk_texts_stats_empty_when_no_mismatch(tmp_path: Path):
    """No mismatch = stats['stale_papers'] should be empty."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "p1.md").write_text(
        "## A\nbody A\n",
        encoding="utf-8",
    )

    # Build a real index via reindex so chunk counts match.
    from corpus_core.corpus_index import FULLTEXT_MAX_SEQ_LENGTH
    from corpus_core.chunker import chunk_markdown

    chunks = chunk_markdown((sources / "p1.md").read_text(encoding="utf-8"),
                             max_tokens=FULLTEXT_MAX_SEQ_LENGTH)
    chunks_meta = [
        {
            "arxiv_id": "p1",
            "section": c.section,
            "chunk_idx": c.chunk_idx,
            "n_chars": c.n_chars,
            "n_tokens_est": c.n_tokens_est,
        }
        for c in chunks
    ]
    n = len(chunks)
    index = EmbeddingIndex(
        matrix=np.zeros((n, 4), dtype=np.float32),
        row_for={"p1": 0},
        model_name="x",
        dims=4,
        metadata={"chunks": chunks_meta, "max_seq_length": FULLTEXT_MAX_SEQ_LENGTH,
                  "n_papers": 1},
    )

    stats: dict = {}
    texts = load_chunk_texts(tmp_path, index, stats=stats)
    assert stats.get("stale_papers") == []
    assert len(texts) == n
    assert all(t != "" for t in texts)


def test_load_chunk_texts_without_stats_still_works(tmp_path: Path):
    """Calling without stats= must not raise."""
    index = EmbeddingIndex(
        matrix=np.zeros((0, 4), dtype=np.float32),
        row_for={},
        model_name="x",
        dims=4,
        metadata={"chunks": [], "max_seq_length": 4096, "n_papers": 0},
    )
    result = load_chunk_texts(tmp_path, index)
    assert result == []


# ---------------------------------------------------------------------------
# Fix #6 -- _persist_index delegates to EmbeddingIndex.save
# ---------------------------------------------------------------------------


def test_persist_index_produces_loadable_embedding_index(tmp_path: Path):
    """_persist_index must write files that EmbeddingIndex.load can read back
    with the same keys as before the refactor."""
    from corpus_core.chunker import CHUNKER_VERSION
    from corpus_core.corpus_index import FULLTEXT_MAX_SEQ_LENGTH

    matrix = np.eye(3, dtype=np.float32)
    row_for = {"a": 0, "b": 1, "c": 2}
    chunk_meta = [
        {"arxiv_id": "a", "section": "Methods", "chunk_idx": 0, "n_chars": 100,
         "n_tokens_est": 20},
        {"arxiv_id": "b", "section": "Results", "chunk_idx": 0, "n_chars": 80,
         "n_tokens_est": 15},
        {"arxiv_id": "c", "section": "Conclusion", "chunk_idx": 0, "n_chars": 60,
         "n_tokens_est": 10},
    ]

    _persist_index(
        tmp_path,
        matrix,
        row_for,
        chunk_meta,
        model_name="test/model",
        n_papers=3,
        encode_seconds=1.23,
    )

    # Both files must exist and no .tmp leftovers.
    assert (tmp_path / "embeddings.npy").exists()
    assert (tmp_path / "index.json").exists()
    assert not (tmp_path / "embeddings.npy.tmp").exists()
    assert not (tmp_path / "index.json.tmp").exists()

    # Reload via EmbeddingIndex.load and verify all expected keys are present.
    idx = EmbeddingIndex.load(tmp_path)
    assert idx.model_name == "test/model"
    assert idx.dims == 3
    assert idx.row_for == row_for
    assert idx.matrix.shape == (3, 3)

    meta = idx.metadata or {}
    assert meta.get("chunker_version") == CHUNKER_VERSION
    assert meta.get("max_seq_length") == FULLTEXT_MAX_SEQ_LENGTH
    assert meta.get("chunks") == chunk_meta
    assert meta.get("n_papers") == 3
    assert abs(meta.get("encode_seconds", 0) - 1.23) < 0.01


def test_persist_index_raw_json_keys(tmp_path: Path):
    """The on-disk index.json must carry the same top-level keys as before,
    so existing lab-corpus / arxiv-radar index files remain readable."""
    matrix = np.zeros((2, 4), dtype=np.float32)
    _persist_index(
        tmp_path,
        matrix,
        {"x": 0, "y": 1},
        [
            {"arxiv_id": "x", "section": "S", "chunk_idx": 0, "n_chars": 5,
             "n_tokens_est": 1},
            {"arxiv_id": "y", "section": "S", "chunk_idx": 0, "n_chars": 5,
             "n_tokens_est": 1},
        ],
        model_name="m",
        n_papers=2,
        encode_seconds=0.5,
    )
    payload = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    required_keys = {
        "model", "dims", "n", "row_for",
        "chunker_version", "max_seq_length", "chunks",
        "n_papers", "encode_seconds",
    }
    missing = required_keys - payload.keys()
    assert not missing, f"index.json missing keys: {missing}"


# ---------------------------------------------------------------------------
# Fix #8 -- _call_tool catches tool exceptions, session stays alive
# ---------------------------------------------------------------------------


class _BrokenHandler:
    def explode(self) -> None:
        raise RuntimeError("tool crashed")

    def value_err(self) -> None:
        raise ValueError("bad value")

    def ok(self) -> dict:
        return {"status": "ok"}


def _run_sync(coro):
    """Run a coroutine synchronously in a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _call_via_handler(app, tool_name: str, arguments: dict) -> dict:
    """Invoke the MCP app's call_tool handler via a real CallToolRequest and
    return the parsed JSON content of the first TextContent item.

    Uses request_handlers[CallToolRequest] which is the SDK-internal path
    that our _call_tool async function is registered on. Returns the parsed
    payload dict from result.root.content[0].text.
    """
    from mcp import types as mcp_types

    handler = app.request_handlers[mcp_types.CallToolRequest]
    req = mcp_types.CallToolRequest(
        method="tools/call",
        params=mcp_types.CallToolRequestParams(name=tool_name, arguments=arguments),
    )
    server_result = _run_sync(handler(req))
    text = server_result.root.content[0].text
    return json.loads(text)


def test_call_tool_wraps_runtime_error_in_error_dict():
    """An exception inside the tool method must be caught and returned as
    {"error": "RuntimeError: tool crashed"}, not propagate to the caller."""
    handler = _BrokenHandler()
    specs = [
        {"name": "explode", "description": ".",
         "inputSchema": {"type": "object", "properties": {}}},
    ]
    dispatcher = make_method_dispatcher(handler, ["explode"])
    app = build_mcp_app(server_name="test", tool_specs=specs, dispatcher=dispatcher)

    data = _call_via_handler(app, "explode", {})
    assert "error" in data
    assert "RuntimeError" in data["error"]
    assert "tool crashed" in data["error"]


def test_call_tool_wraps_value_error():
    handler = _BrokenHandler()
    specs = [
        {"name": "value_err", "description": ".",
         "inputSchema": {"type": "object", "properties": {}}},
    ]
    dispatcher = make_method_dispatcher(handler, ["value_err"])
    app = build_mcp_app(server_name="test", tool_specs=specs, dispatcher=dispatcher)

    data = _call_via_handler(app, "value_err", {})
    assert "error" in data
    assert "ValueError" in data["error"]


def test_call_tool_success_path_unaffected():
    """The happy path (no exception) must still work correctly."""
    handler = _BrokenHandler()
    specs = [
        {"name": "ok", "description": ".",
         "inputSchema": {"type": "object", "properties": {}}},
    ]
    dispatcher = make_method_dispatcher(handler, ["ok"])
    app = build_mcp_app(server_name="test", tool_specs=specs, dispatcher=dispatcher)

    data = _call_via_handler(app, "ok", {})
    assert data == {"status": "ok"}
