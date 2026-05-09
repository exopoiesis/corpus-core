"""Fulltext index lifecycle: reindex (full + incremental) + search primitives.

Reindex flow (incremental by default, falls back to full when needed):
  1. Walk <cache_dir>/fulltext/sources/*.md
  2. Compare against the existing fulltext/index.json: detect new /
     changed (mtime > meta.indexed_at) / deleted papers.
  3. If model name in index doesn't match current encoder, or there is no
     existing index, fall back to full rebuild.
  4. Else encode only new + changed papers, drop rows of deleted/changed
     ones, np.concatenate to existing matrix → atomic write of
     embeddings.npy + index.json.
  5. row_for maps arxiv_id → first chunk row of that paper (used by
     similar_to_paper to seed the mean-of-chunks).

Why incremental matters: a 50-paper enriched corpus takes 5-30 min to
reindex on GPU when re-encoding everything. After fetch_papers([new_id]),
only the new paper's chunks need encoding — incremental cuts that to
seconds. Full rebuild is still available via `incremental=False` for
post-model-change reseeding or recovery from a corrupted index.

Search:
  * search_paper_text  — substring scan over chunk texts
  * search_paper_semantic — cosine over chunk embeddings, returns
                             {arxiv_id, section, snippet, score} payloads
  * similar_to_paper   — mean-of-chunks → cosine over chunk matrix

This module is fulltext-only; abstract semantics live in `search.py`.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from corpus_core.chunker import Chunk, chunk_markdown
from corpus_core.embeddings import EmbeddingIndex, Encoder

LOG = logging.getLogger(__name__)


# Encoder seq window for fulltext chunks. Matches the chunker's default
# `max_tokens` — the chunker won't emit anything bigger than this, so we
# never need to encode at a longer window.
FULLTEXT_MAX_SEQ_LENGTH = 4_096

# Adaptive bucketing for reindex performance. Without bucketing, a 500-token
# References chunk would still pad to FULLTEXT_MAX_SEQ_LENGTH (encoder's
# seq window), wasting compute per row. Buckets group chunks by estimated
# token count so each pass uses a tighter seq window.
#
# Token thresholds reflect the 2026-05-02 chunker max=4096 default — the
# previous "long" 12288 bucket disappeared once chunker started splitting
# big sections into sub-chunks of ≤4096 tokens with overlap. On RTX 4070
# with bf16 Qwen3-4B:
#   short  (≤512)  : ~0.26 s/chunk @ batch=64
#   medium (≤2048) : ~0.17 s/chunk @ batch=16
#   long   (≤4096) : ~1-2 s/chunk @ batch=8
#
# Each entry: (max_tokens_inclusive, encode_seq_length, batch_size_on_12gb).
# Anything over the largest threshold reuses the long bucket (truncated
# to encode_seq_length by the encoder).
_REINDEX_BUCKETS = [
    # (token_threshold, encode_seq_length, batch_size)
    (   512,    512, 64),   # short: Headers, References, captions
    ( 2_048,  2_048, 16),   # medium: typical paragraphs
    ( 4_096,  4_096,  8),   # long: full sub-chunks of a methods/results section
]


@dataclass
class _PaperChunks:
    arxiv_id: str
    chunks: list[Chunk]


@dataclass
class _ReindexPlan:
    """Decision and per-paper partition for one reindex pass.

    `full_rebuild_reason` set → caller falls back to full rebuild and the
    other lists are advisory only. Otherwise the four id-lists partition
    the union of (papers in sources_dir) ∪ (papers in existing index).
    """
    full_rebuild_reason: str | None
    new_ids: list[str] = field(default_factory=list)
    changed_ids: list[str] = field(default_factory=list)
    deleted_ids: list[str] = field(default_factory=list)
    unchanged_ids: list[str] = field(default_factory=list)


# Tolerance for filesystem mtime resolution and clock skew when comparing
# `<id>.md.mtime` against `<id>.meta.json.indexed_at` to detect changes.
# 1 second is enough on every common filesystem (ext4, NTFS, APFS).
_MTIME_TOLERANCE_S = 1.0


def reindex(
    fulltext_dir: Path,
    encoder: Encoder,
    *,
    incremental: bool = True,
    progress_cb=None,
) -> EmbeddingIndex:
    """Reindex dispatcher: incremental by default, full when forced or required.

    Incremental encodes only papers that are new or whose source markdown
    has changed since last index, dropping rows for deleted ones. Falls
    back to full rebuild if the existing index is missing, unparseable,
    or was built with a different encoder model.

    `progress_cb(n_done, n_total)` is called once at the end with the
    final paper count (matches the legacy semantics — JobHandle gets a
    completion update). Returns the freshly-built index.

    Pass `incremental=False` to force a full rebuild even when an index
    exists — useful as recovery after a corrupted swap.
    """
    sources_dir = fulltext_dir / "sources"
    if not sources_dir.exists():
        raise FileNotFoundError(f"no sources directory at {sources_dir}")

    if not incremental:
        return _reindex_full(fulltext_dir, encoder, progress_cb=progress_cb)

    existing_payload = _load_existing_payload(fulltext_dir)
    plan = _classify_papers(sources_dir, existing_payload, encoder.model_name)

    if plan.full_rebuild_reason is not None:
        LOG.info(f"reindex: full rebuild ({plan.full_rebuild_reason})")
        return _reindex_full(fulltext_dir, encoder, progress_cb=progress_cb)

    return _reindex_incremental(
        fulltext_dir, encoder, existing_payload, plan,
        progress_cb=progress_cb,
    )


def _reindex_full(
    fulltext_dir: Path,
    encoder: Encoder,
    *,
    progress_cb=None,
) -> EmbeddingIndex:
    """Rebuild fulltext index from scratch, encoding every cached source."""
    sources_dir = fulltext_dir / "sources"
    paper_chunks = _collect_chunks(sources_dir)
    if not paper_chunks:
        raise FileNotFoundError(f"no enriched papers under {sources_dir}")

    LOG.info(f"reindex: {len(paper_chunks)} papers, "
             f"{sum(len(p.chunks) for p in paper_chunks)} chunks total")

    prior_max_seq = _get_max_seq_length(encoder)
    prior_batch = encoder.config.embeddings.batch_size

    try:
        all_chunks: list[Chunk] = []
        chunk_meta: list[dict] = []
        row_for: dict[str, int] = {}

        for pc in paper_chunks:
            row_for[pc.arxiv_id] = len(all_chunks)
            for c in pc.chunks:
                all_chunks.append(c)
                chunk_meta.append({
                    "arxiv_id": pc.arxiv_id,
                    "section": c.section,
                    "chunk_idx": c.chunk_idx,
                    "n_chars": c.n_chars,
                    "n_tokens_est": c.n_tokens_est,
                })

        t0 = time.time()
        matrix, bucket_stats = _encode_bucketed(encoder, all_chunks)
        encode_seconds = time.time() - t0

        for bucket_label, bucket_n, bucket_seconds in bucket_stats:
            if bucket_n:
                LOG.info(f"  bucket {bucket_label}: {bucket_n} chunks "
                         f"in {bucket_seconds:.1f}s "
                         f"({bucket_seconds / bucket_n:.2f}s/chunk)")

        if progress_cb is not None:
            progress_cb(len(paper_chunks), len(paper_chunks))

        _persist_index(
            fulltext_dir, matrix, row_for, chunk_meta,
            model_name=encoder.model_name,
            n_papers=len(paper_chunks),
            encode_seconds=encode_seconds,
        )
        _stamp_meta(sources_dir, paper_chunks)

        LOG.info(f"reindex done: {matrix.shape[0]} chunks → "
                 f"{matrix.nbytes / 1024 / 1024:.1f} MB in {encode_seconds:.1f}s")

        return EmbeddingIndex(
            matrix=matrix,
            row_for=row_for,
            model_name=encoder.model_name,
            dims=int(matrix.shape[1]),
            metadata={
                "max_seq_length": FULLTEXT_MAX_SEQ_LENGTH,
                "chunks": chunk_meta,
                "n_papers": len(paper_chunks),
            },
        )
    finally:
        _set_max_seq_length(encoder, prior_max_seq)
        encoder.config.embeddings.batch_size = prior_batch


def _reindex_incremental(
    fulltext_dir: Path,
    encoder: Encoder,
    existing_payload: dict,
    plan: _ReindexPlan,
    *,
    progress_cb=None,
) -> EmbeddingIndex:
    """Apply the partition from `plan` to the existing index.

    Three sub-paths:
      * noop — nothing new/changed/deleted. Reload from disk and return.
      * append-only — only new papers. Encode them, np.concatenate.
      * mixed — drop rows of deleted+changed, encode new+changed, concat.
    """
    sources_dir = fulltext_dir / "sources"

    n_papers_before = existing_payload.get("n_papers", 0)

    # Noop: nothing to do.
    if not plan.new_ids and not plan.changed_ids and not plan.deleted_ids:
        LOG.info(f"reindex incremental: noop (={len(plan.unchanged_ids)} "
                 f"papers unchanged)")
        if progress_cb is not None:
            progress_cb(n_papers_before, n_papers_before)
        # Re-load to give the caller a fresh EmbeddingIndex instance — the
        # server's _do_reindex assigns this back to self.fulltext_index.
        return EmbeddingIndex.load(fulltext_dir)

    drop_ids = set(plan.deleted_ids) | set(plan.changed_ids)
    old_chunks: list[dict] = list(existing_payload.get("chunks", []))
    survive_rows: list[int] = [i for i, c in enumerate(old_chunks)
                                if c["arxiv_id"] not in drop_ids]
    survive_chunks: list[dict] = [old_chunks[i] for i in survive_rows]

    # Load existing matrix and copy survivors out before any write — fancy
    # indexing always copies into a new ndarray, so the mmap is no longer
    # referenced by `kept_matrix` after this line.
    old_npy = np.load(fulltext_dir / "embeddings.npy", mmap_mode="r")
    if survive_rows:
        kept_matrix = np.array(old_npy[survive_rows])
    else:
        kept_matrix = np.zeros((0, int(old_npy.shape[1])), dtype=np.float32)
    del old_npy  # release mmap before atomic-rename below.

    # Re-chunk new + changed papers. Order: new first (sorted), then changed.
    encode_paper_chunks = _chunk_paper_ids(
        sources_dir, plan.new_ids + plan.changed_ids,
    )

    flat_chunks: list[Chunk] = []
    new_chunk_meta: list[dict] = []
    for pc in encode_paper_chunks:
        for c in pc.chunks:
            flat_chunks.append(c)
            new_chunk_meta.append({
                "arxiv_id": pc.arxiv_id,
                "section": c.section,
                "chunk_idx": c.chunk_idx,
                "n_chars": c.n_chars,
                "n_tokens_est": c.n_tokens_est,
            })

    encode_seconds = 0.0
    if flat_chunks:
        prior_max_seq = _get_max_seq_length(encoder)
        prior_batch = encoder.config.embeddings.batch_size
        try:
            t0 = time.time()
            new_matrix, bucket_stats = _encode_bucketed(encoder, flat_chunks)
            encode_seconds = time.time() - t0
            for bucket_label, bucket_n, bucket_seconds in bucket_stats:
                if bucket_n:
                    LOG.info(f"  bucket {bucket_label}: {bucket_n} chunks "
                             f"in {bucket_seconds:.1f}s "
                             f"({bucket_seconds / bucket_n:.2f}s/chunk)")
        finally:
            _set_max_seq_length(encoder, prior_max_seq)
            encoder.config.embeddings.batch_size = prior_batch

        if kept_matrix.shape[0] == 0:
            # Started from a fully-rebuilt set — adopt new_matrix's dim.
            kept_matrix = np.zeros((0, new_matrix.shape[1]), dtype=np.float32)
        elif kept_matrix.shape[1] != new_matrix.shape[1]:
            # Belt-and-suspenders: classification already enforces model
            # match, but reject obvious dim drift before persisting.
            raise RuntimeError(
                f"reindex incremental: dim mismatch — kept matrix "
                f"{kept_matrix.shape[1]} vs newly encoded "
                f"{new_matrix.shape[1]}"
            )
        matrix = np.concatenate([kept_matrix, new_matrix], axis=0)
    else:
        matrix = kept_matrix.astype(np.float32, copy=False)

    combined_chunks = survive_chunks + new_chunk_meta

    # row_for: first row index for each arxiv_id, in combined order.
    row_for: dict[str, int] = {}
    for i, c in enumerate(combined_chunks):
        pid = c["arxiv_id"]
        if pid not in row_for:
            row_for[pid] = i

    n_papers = len({c["arxiv_id"] for c in combined_chunks})

    _persist_index(
        fulltext_dir, matrix, row_for, combined_chunks,
        model_name=encoder.model_name,
        n_papers=n_papers,
        encode_seconds=encode_seconds,
    )
    _stamp_meta(sources_dir, encode_paper_chunks)

    LOG.info(
        f"reindex incremental: +{len(plan.new_ids)} new "
        f"~{len(plan.changed_ids)} changed -{len(plan.deleted_ids)} deleted "
        f"={len(plan.unchanged_ids)} unchanged → "
        f"{matrix.shape[0]} chunks across {n_papers} papers "
        f"in {encode_seconds:.1f}s"
    )

    if progress_cb is not None:
        progress_cb(n_papers, n_papers)

    return EmbeddingIndex(
        matrix=matrix,
        row_for=row_for,
        model_name=encoder.model_name,
        dims=int(matrix.shape[1]),
        metadata={
            "max_seq_length": FULLTEXT_MAX_SEQ_LENGTH,
            "chunks": combined_chunks,
            "n_papers": n_papers,
        },
    )


def _classify_papers(
    sources_dir: Path,
    existing_payload: dict | None,
    current_model_name: str,
) -> _ReindexPlan:
    """Decide which papers are new / changed / deleted / unchanged.

    Returns `full_rebuild_reason` set when the existing index can't be
    extended in place (missing, model mismatch). Otherwise the four
    id-lists partition (sources_dir ∪ indexed_ids).

    A paper is "changed" when its `<id>.md.mtime` is greater than the
    `indexed_at` timestamp recorded in `<id>.meta.json` (with a 1-second
    tolerance for filesystem clock granularity). Papers with no readable
    `indexed_at` are treated as changed so they get re-encoded once,
    after which the meta gets stamped and they settle into "unchanged".
    """
    if existing_payload is None:
        return _ReindexPlan(full_rebuild_reason="no existing index")

    indexed_model = existing_payload.get("model")
    if indexed_model != current_model_name:
        return _ReindexPlan(
            full_rebuild_reason=(
                f"model mismatch (index was built with {indexed_model!r}, "
                f"current encoder is {current_model_name!r})"
            ),
        )

    indexed_ids = {c["arxiv_id"] for c in existing_payload.get("chunks", [])}
    source_ids = {p.stem for p in sources_dir.glob("*.md")}

    new_ids = sorted(source_ids - indexed_ids)
    deleted_ids = sorted(indexed_ids - source_ids)
    candidates = sorted(source_ids & indexed_ids)

    changed_ids: list[str] = []
    unchanged_ids: list[str] = []

    for pid in candidates:
        if _paper_changed_since_index(sources_dir, pid):
            changed_ids.append(pid)
        else:
            unchanged_ids.append(pid)

    return _ReindexPlan(
        full_rebuild_reason=None,
        new_ids=new_ids,
        changed_ids=changed_ids,
        deleted_ids=deleted_ids,
        unchanged_ids=unchanged_ids,
    )


def _paper_changed_since_index(sources_dir: Path, arxiv_id: str) -> bool:
    """True if `<id>.md.mtime` exceeds `<id>.meta.json.indexed_at`.

    Defensive defaults: when meta is missing or unparseable, return True
    so the paper gets re-encoded (after which meta gets a fresh stamp).
    """
    meta_path = sources_dir / f"{arxiv_id}.meta.json"
    md_path = sources_dir / f"{arxiv_id}.md"

    indexed_at_epoch = _read_indexed_at_epoch(meta_path)
    if indexed_at_epoch is None:
        return True

    try:
        md_mtime = md_path.stat().st_mtime
    except OSError:
        return True

    return md_mtime > indexed_at_epoch + _MTIME_TOLERANCE_S


def _read_indexed_at_epoch(meta_path: Path) -> float | None:
    """Parse `meta.indexed_at` (ISO 8601) into epoch seconds; None on miss."""
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    iso = meta.get("indexed_at")
    if not iso:
        return None
    try:
        # Python 3.10 fromisoformat doesn't accept trailing Z; substitute.
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt.timestamp()


def _load_existing_payload(fulltext_dir: Path) -> dict | None:
    """Read fulltext/index.json if both index.json and embeddings.npy exist."""
    index_path = fulltext_dir / "index.json"
    npy_path = fulltext_dir / "embeddings.npy"
    if not index_path.exists() or not npy_path.exists():
        return None
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _persist_index(
    fulltext_dir: Path,
    matrix: np.ndarray,
    row_for: dict[str, int],
    chunk_meta: list[dict],
    *,
    model_name: str,
    n_papers: int,
    encode_seconds: float,
) -> None:
    """Atomically write embeddings.npy + index.json (.tmp → rename)."""
    fulltext_dir.mkdir(parents=True, exist_ok=True)

    npy_final = fulltext_dir / "embeddings.npy"
    npy_tmp = fulltext_dir / "embeddings.npy.tmp"
    with open(npy_tmp, "wb") as f:
        np.save(f, matrix)
    npy_tmp.replace(npy_final)

    index_payload = {
        "model": model_name,
        "dims": int(matrix.shape[1]),
        "n": int(matrix.shape[0]),
        "row_for": row_for,
        "max_seq_length": FULLTEXT_MAX_SEQ_LENGTH,
        "chunks": chunk_meta,
        "n_papers": n_papers,
        "encode_seconds": round(encode_seconds, 2),
    }
    json_final = fulltext_dir / "index.json"
    json_tmp = fulltext_dir / "index.json.tmp"
    json_tmp.write_text(json.dumps(index_payload, indent=1), encoding="utf-8")
    json_tmp.replace(json_final)


def _stamp_meta(sources_dir: Path, paper_chunks: list[_PaperChunks]) -> None:
    """Backfill n_chunks_after_split + indexed_at on each paper's meta.json.

    Used by both full and incremental paths. For full reindex, stamps
    every paper. For incremental, stamps only the ones that were
    re-encoded — unchanged papers keep their existing indexed_at.
    """
    now = _utcnow_iso()
    for pc in paper_chunks:
        meta_path = sources_dir / f"{pc.arxiv_id}.meta.json"
        try:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                meta = {}
            meta["n_chunks_after_split"] = len(pc.chunks)
            meta["indexed_at"] = now
            tmp = meta_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
            tmp.replace(meta_path)
        except OSError as e:
            LOG.warning(f"could not update meta for {pc.arxiv_id}: {e}")


def _collect_chunks(sources_dir: Path) -> list[_PaperChunks]:
    """Walk sources_dir, run chunker on every .md file (used by full reindex)."""
    out: list[_PaperChunks] = []
    for md_path in sorted(sources_dir.glob("*.md")):
        arxiv_id = md_path.stem
        chunks = _chunk_one(md_path, arxiv_id)
        if chunks is not None:
            out.append(_PaperChunks(arxiv_id=arxiv_id, chunks=chunks))
    return out


def _chunk_paper_ids(sources_dir: Path, ids: list[str]) -> list[_PaperChunks]:
    """Run chunker only on the requested ids (used by incremental path)."""
    out: list[_PaperChunks] = []
    for pid in ids:
        chunks = _chunk_one(sources_dir / f"{pid}.md", pid)
        if chunks is not None:
            out.append(_PaperChunks(arxiv_id=pid, chunks=chunks))
    return out


def _chunk_one(md_path: Path, arxiv_id: str) -> list[Chunk] | None:
    """Read + chunk a single paper. None for read errors or empty results."""
    try:
        text = md_path.read_text(encoding="utf-8")
    except OSError as e:
        LOG.warning(f"[reindex] skip {arxiv_id}: read error {e}")
        return None
    chunks = chunk_markdown(text, max_tokens=FULLTEXT_MAX_SEQ_LENGTH)
    if not chunks:
        LOG.warning(f"[reindex] skip {arxiv_id}: chunker produced nothing")
        return None
    return chunks


def _set_max_seq_length(encoder: Encoder, target: int) -> int:
    """Set the underlying SentenceTransformer's max_seq_length, return prior.

    Encoder lazy-loads — to keep things simple we ensure the model is loaded
    here so the attribute exists.
    """
    encoder._ensure_loaded()  # noqa: SLF001 — internal API
    prior = getattr(encoder._model, "max_seq_length", -1)
    encoder._model.max_seq_length = target
    return prior


def _get_max_seq_length(encoder: Encoder) -> int:
    """Read current max_seq_length without forcing a load if not yet loaded."""
    if encoder._model is None:  # noqa: SLF001
        return -1
    return getattr(encoder._model, "max_seq_length", -1)


def _encode_bucketed(
    encoder: Encoder, chunks: list[Chunk],
) -> tuple[np.ndarray, list[tuple[str, int, float]]]:
    """Encode chunks in length-sorted buckets so short chunks don't pay the
    cost of the longest seq window.

    Returns:
        matrix      — (N, dim) embeddings in original chunk order
        bucket_stats — [(label, n_chunks, seconds), ...] for telemetry

    The bucketing pass is correctness-neutral (same model, same prefix, same
    L2 norm — embeddings are unchanged) and reduces wasted padding-pass
    compute by 5-30× on typical arxiv-paper chunk-length distributions.
    """
    if not chunks:
        # Encode nothing to a (0, dim) array — pull dim from a one-token probe.
        probe = encoder.encode_passages(["x"], show_progress=False)
        return np.zeros((0, probe.shape[-1]), dtype=np.float32), []

    n = len(chunks)
    # bucket_idx[i] = index into _REINDEX_BUCKETS for chunk i
    bucket_assignment: list[int] = []
    for c in chunks:
        for b_idx, (threshold, _seq, _bs) in enumerate(_REINDEX_BUCKETS):
            if c.n_tokens_est <= threshold:
                bucket_assignment.append(b_idx)
                break
        else:
            # token count exceeds the largest bucket — use the largest.
            bucket_assignment.append(len(_REINDEX_BUCKETS) - 1)

    # Encode bucket-by-bucket; record output rows by their original index.
    rows: list[np.ndarray | None] = [None] * n
    stats: list[tuple[str, int, float]] = []

    for b_idx, (threshold, seq_len, batch_size) in enumerate(_REINDEX_BUCKETS):
        original_idx_in_bucket = [i for i, b in enumerate(bucket_assignment) if b == b_idx]
        label = f"≤{threshold}t"
        if not original_idx_in_bucket:
            stats.append((label, 0, 0.0))
            continue
        texts = [chunks[i].text for i in original_idx_in_bucket]

        t0 = time.time()
        bucket_matrix = encoder.encode_passages(
            texts,
            show_progress=False,
            max_seq_length=seq_len,
            batch_size=batch_size,
        )
        elapsed = time.time() - t0

        for j, orig_i in enumerate(original_idx_in_bucket):
            rows[orig_i] = bucket_matrix[j]

        stats.append((label, len(original_idx_in_bucket), elapsed))

    # Stack — by construction every slot is filled.
    assert all(r is not None for r in rows), "bucketing left a chunk un-encoded"
    matrix = np.stack(rows, axis=0).astype(np.float32, copy=False)
    return matrix, stats


def _utcnow_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Search primitives over the fulltext index
# ---------------------------------------------------------------------------


def _snippet(text: str, query: str | None = None, length: int = 240) -> str:
    """Pull a representative window from the chunk text. If query is given,
    center on the first match; else take the head."""
    if query:
        m = re.search(re.escape(query.split()[0]), text, re.IGNORECASE) if query.split() else None
        if m:
            start = max(0, m.start() - length // 3)
            end = min(len(text), start + length)
            prefix = "…" if start > 0 else ""
            suffix = "…" if end < len(text) else ""
            return prefix + text[start:end].strip() + suffix
    head = text[:length].strip()
    return head + ("…" if len(text) > length else "")


# Section-name patterns we treat as "junk" — they're high-recall for any
# query (lots of keywords from cited papers, author affiliations) but
# they're rarely what a researcher actually wants to read. Header is
# explicitly NOT in this list — it carries title+abstract and is often
# the right hit.
_JUNK_SECTION_RE = re.compile(
    r"^(references?|"
    r"acknowledge?ments?|"
    r"bibliograph(y|ic)|"
    r"data\s+availab(ility|le)|"
    r"author\s+(contributions|information)|"
    r"funding|"
    r"competing\s+interests|"
    r"declarations?|"
    r"conflict\s+of\s+interest|"
    r"supplementary\s+(material|information)|"
    r"appendix\s+[a-z]\b)",   # "Appendix A", "Appendix B" etc — usually proofs/extras
    re.IGNORECASE,
)


def is_junk_section(section_name: str) -> bool:
    """True if section_name looks like a citations/admin section that
    we want to demote in search results."""
    if not section_name:
        return False
    name = section_name.strip()
    # Strip leading section numbers / labels — many shapes in the wild:
    #   "1 Introduction", "1.2.3 Foo", "S10 Methods", "A.1 Bar",
    #   "V Acknowledgements" (Roman numeral), "Section 4. Results",
    #   "Appendix A. References", "Chapter 3. Methods"
    prefix_pattern = (
        r"^("
        r"(?:Section|Chapter|Part|Appendix)\s+[A-Z0-9]+\.?\s+"   # "Section 4. ", "Appendix A. "
        r"|[IVXLCDM]+\b\.?\s+"                                    # Roman numerals: V, IV, III, …
        r"|[A-Z]?\d+(?:\.\d+)*\.?\s+"                             # 1, 1.2, S10, A1
        r")"
    )
    stripped = re.sub(prefix_pattern, "", name, count=1, flags=re.IGNORECASE)
    return bool(_JUNK_SECTION_RE.match(stripped))


def search_paper_text(
    chunk_texts: list[str],
    chunk_meta: list[dict],
    query: str,
    k: int = 10,
    snippet_chars: int = 240,
) -> list[dict]:
    """Substring AND-scan over chunk texts. Title-boost not applicable here —
    chunks already carry their section as a separate field. Junk sections
    (References, Acknowledgments, etc.) are filtered after ranking — see
    is_junk_section.

    `snippet_chars` controls the length of the returned `snippet` field
    (default 240). Use larger values when extracting recipes / numeric
    parameters from a paper.
    """
    tokens = [t for t in re.split(r"\s+", query.lower().strip()) if t]
    if not tokens or not chunk_texts:
        return []

    scored: list[tuple[float, int]] = []
    for i, text in enumerate(chunk_texts):
        text_l = text.lower()
        if all(t in text_l for t in tokens):
            # Score by token-occurrence count (cheap proxy for relevance).
            score = float(sum(text_l.count(t) for t in tokens))
            scored.append((score, i))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict] = []
    junk_fallback: list[dict] = []
    for score, idx in scored:
        meta = chunk_meta[idx]
        item = {
            "arxiv_id": meta["arxiv_id"],
            "section": meta["section"],
            "chunk_idx": meta.get("chunk_idx", 0),
            "snippet": _snippet(chunk_texts[idx], query=query,
                                length=snippet_chars),
            "score": round(score, 4),
        }
        if is_junk_section(meta["section"]):
            junk_fallback.append(item)
        else:
            out.append(item)
        if len(out) >= k:
            break

    # If filtering left us short, top up from junk so the response is
    # never silently empty.
    if len(out) < k:
        out.extend(junk_fallback[: k - len(out)])
    return out[:k]


def search_paper_semantic(
    index: EmbeddingIndex,
    chunk_texts: list[str] | None,
    query_vec: np.ndarray,
    k: int = 10,
    snippet_chars: int = 240,
) -> list[dict]:
    """Cosine over chunk embeddings; return per-chunk payloads.

    Junk sections (References / Acknowledgments / Data availability /
    Appendix [A-Z] etc.) are filtered after ranking — they have high
    keyword density but rarely match user intent. Filter is applied by
    oversampling top-k×4 candidates and skipping junk; if filtering
    leaves <k results, junk is appended at the tail so the response is
    never silently empty.

    `chunk_texts` is optional — if provided we include a snippet, else
    only the meta fields. `snippet_chars` controls snippet length
    (default 240; raise to ~600 when the LLM needs longer context for
    parameter extraction).
    """
    if index.metadata is None or "chunks" not in index.metadata:
        return []
    chunk_meta = index.metadata["chunks"]
    if not chunk_meta:
        return []

    sims = index.matrix @ query_vec
    # Oversample so that after junk-filter we still have k clean hits in
    # most realistic cases.
    oversample = min(max(k * 4, 20), len(sims))
    top = np.argpartition(-sims, oversample - 1)[:oversample]
    top_sorted = top[np.argsort(-sims[top])]

    clean: list[dict] = []
    junk: list[dict] = []
    for idx in top_sorted:
        meta = chunk_meta[int(idx)]
        snippet = ""
        if chunk_texts is not None and int(idx) < len(chunk_texts):
            snippet = _snippet(chunk_texts[int(idx)], length=snippet_chars)
        item = {
            "arxiv_id": meta["arxiv_id"],
            "section": meta["section"],
            "chunk_idx": meta.get("chunk_idx", 0),
            "snippet": snippet,
            "score": round(float(sims[int(idx)]), 4),
        }
        if is_junk_section(meta["section"]):
            junk.append(item)
        else:
            clean.append(item)
        if len(clean) >= k:
            break

    if len(clean) < k:
        clean.extend(junk[: k - len(clean)])
    return clean[:k]


def similar_to_paper(
    index: EmbeddingIndex,
    arxiv_id: str,
    k: int = 10,
) -> list[dict]:
    """Mean-of-chunks → cosine over the chunk matrix, group results by paper.

    Returns one row per paper (best-scoring chunk wins for ranking),
    excluding the source paper itself. Useful for "show me similar papers
    based on full content, not just abstract".
    """
    rows = [i for i, _ in index.chunks_for(arxiv_id)]
    if not rows:
        return []
    mean_vec = index.matrix[rows].mean(axis=0)
    n = float(np.linalg.norm(mean_vec))
    if n == 0:
        return []
    mean_vec = (mean_vec / n).astype(np.float32)

    sims = index.matrix @ mean_vec
    chunk_meta = index.metadata.get("chunks", []) if index.metadata else []
    if not chunk_meta:
        return []

    # Group by arxiv_id, take best chunk per paper, exclude source.
    best_per_paper: dict[str, tuple[float, int]] = {}
    for i, s in enumerate(sims):
        pid = chunk_meta[i].get("arxiv_id")
        if not pid or pid == arxiv_id:
            continue
        prev = best_per_paper.get(pid)
        if prev is None or s > prev[0]:
            best_per_paper[pid] = (float(s), i)

    ranked = sorted(best_per_paper.items(), key=lambda x: x[1][0], reverse=True)
    out: list[dict] = []
    for pid, (score, row) in ranked[:k]:
        meta = chunk_meta[row]
        out.append({
            "arxiv_id": pid,
            "section": meta["section"],
            "chunk_idx": meta.get("chunk_idx", 0),
            "score": round(score, 4),
        })
    return out


def load_chunk_texts(fulltext_dir: Path, index: EmbeddingIndex) -> list[str]:
    """Re-derive chunk texts from cached source markdowns + chunker.

    The index doesn't carry chunk text bodies (would inflate index.json
    by 100×); we re-chunk on demand. Cheap because chunker is O(N) regex
    + a few string concat passes, milliseconds per paper.
    """
    if index.metadata is None or "chunks" not in index.metadata:
        return []
    chunks_meta = index.metadata["chunks"]

    # Group chunk-meta by arxiv_id to know how many we expect per paper.
    by_id: dict[str, list[dict]] = {}
    for c in chunks_meta:
        by_id.setdefault(c["arxiv_id"], []).append(c)

    # Re-chunk each source and emit texts in the order matching index rows.
    sources_dir = fulltext_dir / "sources"
    text_for_row: list[str | None] = [None] * len(chunks_meta)

    for arxiv_id, paper_meta in by_id.items():
        md_path = sources_dir / f"{arxiv_id}.md"
        try:
            text = md_path.read_text(encoding="utf-8")
        except OSError:
            continue
        rebuilt = chunk_markdown(text, max_tokens=FULLTEXT_MAX_SEQ_LENGTH)
        # Find the index range for this paper in chunk_meta order.
        # We rely on chunker being deterministic — same input produces same
        # ordered chunks in the same order it did during reindex.
        rows_for_paper = [i for i, m in enumerate(chunks_meta)
                          if m["arxiv_id"] == arxiv_id]
        for row_i, chunk in zip(rows_for_paper, rebuilt):
            text_for_row[row_i] = chunk.text

    return [t or "" for t in text_for_row]
