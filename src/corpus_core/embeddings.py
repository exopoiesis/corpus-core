"""Sentence-transformers embedding cache.

On `--build-cache`:
  1. Load corpus
  2. For each paper: encode `paper.search_text` (title + abstract), prefixed
     with the model's passage-prefix when the model expects one (E5 family).
  3. Persist:
       <cache_dir>/embeddings.npy        — float32, shape (N, D), L2-normalized
       <cache_dir>/index.json            — {arxiv_id: row_idx, ...}, plus model + dims
  4. Subsequent runs: mmap-load embeddings + index for fast cosine similarity.

At query-time, the SentenceTransformer instance lives on the long-running
server (Encoder) — loaded once, reused for every query. The legacy
`encode_query(text, config)` shim instantiates fresh per-call and is kept
only for ad-hoc / one-shot scripts.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Encoder accepts any object with `.embeddings.{model, batch_size,
    # target_dim, cache_dir}` — i.e. a downstream project's Config.
    # Declared as a Protocol so corpus_core stays runtime-independent of
    # any specific project schema.
    from typing import Protocol

    class _EmbeddingsCfg(Protocol):
        model: str
        cache_dir: Path
        batch_size: int
        target_dim: int | None

    class Config(Protocol):
        embeddings: "_EmbeddingsCfg"

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Instruction-prefix registry
# ---------------------------------------------------------------------------
# Several embedding families were trained with explicit query / passage
# prefixes. Forgetting them silently costs 5–15% recall. The registry maps
# canonical model names to their prescribed prefixes; unknown models get
# empty strings (no-op).

_QWEN3_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery: "
)

_QUERY_PREFIX = {
    # mxbai (Mixedbread) — query-only prefix, passages plain.
    "mixedbread-ai/mxbai-embed-large-v1":
        "Represent this sentence for searching relevant passages: ",
    "mixedbread-ai/mxbai-embed-large-v2":
        "Represent this sentence for searching relevant passages: ",

    # BGE family (BAAI) — query-only prefix.
    "BAAI/bge-large-en-v1.5":
        "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5":
        "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-small-en-v1.5":
        "Represent this sentence for searching relevant passages: ",

    # E5 family (Microsoft) — symmetric prefixes on both sides.
    "intfloat/e5-large-v2": "query: ",
    "intfloat/e5-base-v2":  "query: ",
    "intfloat/e5-small-v2": "query: ",
    "intfloat/multilingual-e5-large": "query: ",
    "intfloat/multilingual-e5-base":  "query: ",

    # Qwen3-Embedding — instruction-style query template, no doc prefix.
    "Qwen/Qwen3-Embedding-0.6B": _QWEN3_QUERY_PREFIX,
    "Qwen/Qwen3-Embedding-4B":   _QWEN3_QUERY_PREFIX,
    "Qwen/Qwen3-Embedding-8B":   _QWEN3_QUERY_PREFIX,

    # Microsoft Harrier-OSS — same Instruct/Query template (verified on
    # the 27B model card). Last-token pooling + L2 norm handled by
    # sentence-transformers. No prefix on the document side.
    "microsoft/harrier-oss-v1-0.6b": _QWEN3_QUERY_PREFIX,
    "microsoft/harrier-oss-v1-27b":  _QWEN3_QUERY_PREFIX,
}

_PASSAGE_PREFIX = {
    # E5 expects "passage: " on the document side too.
    "intfloat/e5-large-v2": "passage: ",
    "intfloat/e5-base-v2":  "passage: ",
    "intfloat/e5-small-v2": "passage: ",
    "intfloat/multilingual-e5-large": "passage: ",
    "intfloat/multilingual-e5-base":  "passage: ",
}


def query_prefix(model: str) -> str:
    """Return the query-side prefix for a model, or '' if none is registered."""
    return _QUERY_PREFIX.get(model, "")


def passage_prefix(model: str) -> str:
    """Return the passage-side prefix for a model, or '' if none is registered."""
    return _PASSAGE_PREFIX.get(model, "")


# ---------------------------------------------------------------------------
# Index (read-only, mmap)
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingIndex:
    """In-memory view over a cached embedding matrix.

    Two flavours coexist:
      * abstract index: one row per paper, `row_for[arxiv_id] = row`,
        `metadata` empty.
      * fulltext index: rows are chunks (potentially many per paper).
        `row_for` maps arxiv_id → first row of that paper; `metadata`
        carries the per-row {arxiv_id, section, chunk_idx, n_chars}.

    Use .vector(arxiv_id) to fetch a single embedding (works for abstract
    index; for fulltext returns the first chunk's vector — query through
    `chunks_for(arxiv_id)` if you need them all).
    """
    matrix: np.ndarray                       # (N, D) float32, L2-normalized
    row_for: dict[str, int]
    model_name: str
    dims: int
    metadata: dict | None = None             # {chunks: [...], max_seq_length: int, ...}

    @classmethod
    def load(cls, cache_dir: Path) -> "EmbeddingIndex":
        idx = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
        matrix = np.load(cache_dir / "embeddings.npy", mmap_mode="r")
        # Strip core fields; everything else is metadata (forward-compat).
        core = {"row_for", "model", "dims", "n"}
        metadata = {k: v for k, v in idx.items() if k not in core}
        # chunker_version (if present) stays inside metadata; _classify_papers
        # reads it directly from the on-disk index.json payload.
        return cls(
            matrix=matrix,
            row_for=idx["row_for"],
            model_name=idx["model"],
            dims=idx["dims"],
            metadata=metadata or None,
        )

    @classmethod
    def save(
        cls,
        cache_dir: Path,
        matrix: "np.ndarray",
        row_for: "dict[str, int]",
        model: str,
        metadata: "dict | None" = None,
    ) -> None:
        """Atomically write embeddings.npy + index.json to cache_dir.

        Both files are written to .tmp siblings first; then .npy is renamed,
        then .json is renamed.  This ordering means that during a crash:
          * If only .npy.tmp exists  -> neither file was replaced; old pair valid.
          * If .npy was replaced but .json.tmp not yet renamed -> .json still
            points at the old matrix (stale but consistent).
          * If .npy AND .json were replaced -> new pair valid.
        The one edge case (npy replaced, json.tmp rename fails) leaves the new
        matrix with the old index.  `load()` will fail a coherence check
        (n_rows != matrix.shape[0]) or simply produce wrong results.  Full
        rebuild on next start recovers.  This is identical to the ordering
        used by _persist_index in corpus_index.py and better than no atomicity.

        `metadata` is merged into the JSON payload alongside `model`/`dims`/`n`;
        use it to carry `chunker_version`, `max_seq_length`, `chunks`, etc.
        """
        cache_dir.mkdir(parents=True, exist_ok=True)

        npy_final = cache_dir / "embeddings.npy"
        npy_tmp   = cache_dir / "embeddings.npy.tmp"
        json_final = cache_dir / "index.json"
        json_tmp   = cache_dir / "index.json.tmp"

        # Write .npy to temp first.
        with open(npy_tmp, "wb") as f:
            np.save(f, matrix)

        # Build JSON payload; metadata values are merged at top level.
        payload: dict = {
            "model": model,
            "dims": int(matrix.shape[1]),
            "n": int(matrix.shape[0]),
            "row_for": row_for,
        }
        if metadata:
            payload.update(metadata)

        json_tmp.write_text(
            json.dumps(payload, indent=1), encoding="utf-8",
        )

        # Atomic renames: npy first, then json.
        os.replace(npy_tmp, npy_final)
        os.replace(json_tmp, json_final)

    def vector(self, arxiv_id: str) -> np.ndarray | None:
        i = self.row_for.get(arxiv_id)
        return None if i is None else self.matrix[i]

    def chunks_for(self, arxiv_id: str) -> list[tuple[int, dict]]:
        """For a fulltext index: return [(row_idx, chunk_meta), ...] for one paper.

        Empty list for non-fulltext indexes or unknown arxiv_id.
        """
        if not self.metadata or "chunks" not in self.metadata:
            return []
        return [(i, c) for i, c in enumerate(self.metadata["chunks"])
                if c.get("arxiv_id") == arxiv_id]


# ---------------------------------------------------------------------------
# Encoder (lazy-load wrapper)
# ---------------------------------------------------------------------------

class Encoder:
    """Wraps SentenceTransformer with model-aware prefixes + lazy load.

    Designed to live on the long-running RadarServer: the heavy
    SentenceTransformer instance is loaded on first `encode_*` call and
    kept for the lifetime of the server, so subsequent queries pay no
    model-load cost.
    """

    def __init__(self, config: Config) -> None:
        import threading
        self.config = config
        self._model = None  # type: ignore[var-annotated]
        # Guard concurrent _ensure_loaded calls — refresh loop, warmup task,
        # and reindex jobs all live in different worker threads and can race
        # to load the model twice. Two SentenceTransformer instances ≈ 2×
        # GPU memory (16 GB for Qwen3-4B bf16 — exhausts a 12 GB card,
        # forcing slow CPU/host-memory spillover that drops every encode
        # speed by 3-50× depending on bucket).
        # RLock (reentrant): encode_* hold this lock around the actual
        # `.encode()` AND call _ensure_loaded() (which re-acquires it) inside —
        # needs reentrancy on the same thread. Cross-thread (idle-unload timer)
        # it still serializes: unload() blocks until an in-flight encode releases,
        # so the model can't be moved to CPU mid-forward-pass (was: device-mismatch
        # cpu/cuda crash during long reindex encodes >idle_unload_s).
        self._model_lock = threading.RLock()

        # Idle-unload timer. Explicit unload() is wired only to job
        # completion (reindex / bulk ingest); a plain search loads the
        # model and would otherwise pin ~7-8 GB of VRAM for the server's
        # whole lifetime. Re-arming this timer on every encode drops the
        # model after `idle_unload_s` seconds of no encodes, so on a host
        # that shares the GPU with DFT/MLIP the VRAM frees itself when the
        # corpus is quiet, while back-to-back queries in one session stay
        # warm (no per-query ~20 s cold reload). 0/None disables it.
        self._idle_lock = threading.Lock()
        self._idle_timer = None  # type: ignore[var-annotated]
        try:
            self._idle_unload_s = float(
                getattr(config.embeddings, "idle_unload_s", 0) or 0)
        except (TypeError, ValueError):
            self._idle_unload_s = 0.0

    @property
    def model_name(self) -> str:
        return self.config.embeddings.model

    def _ensure_loaded(self) -> None:
        # Fast path — no lock when already loaded.
        if self._model is not None:
            return
        with self._model_lock:
            # Re-check after acquiring the lock: another thread may have
            # finished loading while we were blocked.
            if self._model is not None:
                return
            import torch
            from sentence_transformers import SentenceTransformer
            LOG.info(f"loading bi-encoder {self.model_name}...")
            model = SentenceTransformer(self.model_name)
            # Cast to bf16 on CUDA. Qwen3-Embedding ships with bf16 weights
            # but sentence-transformers leaves activation dtype at fp32 by
            # default — causing `RuntimeError: expected mat1 and mat2 to
            # have the same dtype` inside Linear layers when the model
            # actually runs. Explicit cast unifies weight and activation
            # dtype so the matmul kernels match.
            # CPU path stays fp32 — bf16 on CPU is much slower than fp32.
            if torch.cuda.is_available():
                model = model.to(dtype=torch.bfloat16)
                LOG.info("  cast to bfloat16 on cuda")
            # Publish only after fully initialized so other threads see a
            # consistent state.
            self._model = model

    def unload(self) -> bool:
        """Drop the underlying SentenceTransformer + free GPU/host memory.

        Idempotent — calling on an already-unloaded encoder is a no-op.
        Returns True if a model was actually released, False if nothing
        was loaded. The next `encode_*` call lazily re-loads.

        Use after heavy one-shot work (reindex, bulk ingest) when the
        host's GPU is also used for unrelated jobs — Qwen3-Embedding-4B
        in bf16 takes ~7-8 GB on the device and that VRAM otherwise
        stays pinned for the server's lifetime.

        Concurrency: takes the same lock as `_ensure_loaded` so a
        parallel encode either runs to completion before the unload or
        triggers a re-load after. Don't call from inside an active
        `encode_*` (would deadlock).
        """
        import gc

        # Cancel any pending idle timer first — whoever called unload()
        # (explicit job-completion path, or the timer itself) takes
        # ownership; a stale timer would just re-fire a harmless no-op.
        self._cancel_idle_timer()

        with self._model_lock:
            if self._model is None:
                return False
            model = self._model
            self._model = None
            try:
                # Move to CPU first so the CUDA allocator can actually
                # release the VRAM blocks on the next empty_cache().
                import torch
                if torch.cuda.is_available():
                    try:
                        model.to("cpu")
                    except Exception as e:  # noqa: BLE001
                        LOG.debug(f"unload: model.to('cpu') ignored: {e}")
            except ImportError:
                torch = None  # type: ignore[assignment]

            del model
            gc.collect()

            try:
                import torch  # re-import in case the first one failed
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except ImportError:
                pass

            LOG.info(f"unloaded bi-encoder {self.model_name}")
            return True

    # -- idle-unload timer ---------------------------------------------------

    def _cancel_idle_timer(self) -> None:
        with self._idle_lock:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None

    def _touch_idle(self) -> None:
        """Re-arm the idle-unload countdown. Called after every encode.

        Each call cancels the prior timer and starts a fresh one, so the
        model is dropped only after a full `idle_unload_s` window with no
        encode activity. No-op when idle-unload is disabled (<= 0).
        """
        if self._idle_unload_s <= 0:
            return
        import threading
        with self._idle_lock:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
            t = threading.Timer(self._idle_unload_s, self._idle_unload_fire)
            t.daemon = True  # never block interpreter shutdown
            self._idle_timer = t
            t.start()

    def _idle_unload_fire(self) -> None:
        try:
            if self.unload():
                LOG.info(
                    f"idle-unload: released bi-encoder {self.model_name} "
                    f"after {self._idle_unload_s:.0f}s of inactivity")
        except Exception as e:  # noqa: BLE001
            LOG.warning(f"idle-unload failed (model stays resident): {e}")

    def encode_query(self, text: str, max_seq_length: int = 512) -> np.ndarray:
        """Encode a single query. L2-normalized, with model-specific prefix.

        Same `max_seq_length=512` default as encode_passages — see that
        method's docstring for why this is required for Qwen3.
        """
        with self._model_lock:               # hold through encode: idle-unload can't interleave
            self._cancel_idle_timer()        # no pending timer fires mid-encode
            self._ensure_loaded()
            self._model.max_seq_length = max_seq_length
            prefixed = query_prefix(self.model_name) + text
            vec = self._model.encode(  # type: ignore[union-attr]
                [prefixed],
                normalize_embeddings=True,
            ).astype(np.float32)[0]
        self._touch_idle()                   # re-arm only after encode completes
        return _maybe_truncate(vec, self.config.embeddings.target_dim)

    def encode_passages(
        self,
        texts: list[str],
        show_progress: bool = True,
        max_seq_length: int = 512,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Encode a batch of passages. L2-normalized, with model prefix if any.

        `max_seq_length` defaults to 512 — fits arxiv abstracts (~150-300
        tokens average) with headroom. Caller can pass a larger value (e.g.
        12288 for full-text section chunks); fulltext_index does this
        per-bucket. **Critical**: Qwen3 ships with max_seq_length=32768,
        and sentence-transformers `padding="max_length"` will pad every
        batch to that length on some configs, costing ~30-40× slowdown on
        short inputs. Setting this attribute on the loaded model before
        encode is the documented fix.

        `batch_size` overrides config.embeddings.batch_size for this call.
        """
        with self._model_lock:               # hold through encode: idle-unload/unload() can't
            self._cancel_idle_timer()        # interleave and move model to CPU mid-forward-pass
            self._ensure_loaded()
            # Set the per-call seq window (no-op when already at this length).
            self._model.max_seq_length = max_seq_length

            prefix = passage_prefix(self.model_name)
            prepared = [prefix + t for t in texts] if prefix else texts
            bs = batch_size if batch_size is not None else self.config.embeddings.batch_size
            matrix = self._model.encode(  # type: ignore[union-attr]
                prepared,
                batch_size=bs,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32, copy=False)
        self._touch_idle()                   # re-arm only after encode completes
        return _maybe_truncate(matrix, self.config.embeddings.target_dim)


def _maybe_truncate(arr: np.ndarray, target_dim: int | None) -> np.ndarray:
    """Matryoshka-style dim reduction: slice to target_dim, then re-L2-normalize.

    No-op when target_dim is None or already ≥ the model's native dim. Works on
    a single vector (1-D) or a batch (2-D, last axis = features).
    """
    if target_dim is None:
        return arr
    native = arr.shape[-1]
    if target_dim >= native:
        return arr
    sliced = arr[..., :target_dim]
    norms = np.linalg.norm(sliced, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (sliced / norms).astype(np.float32, copy=False)


# build_cache and the legacy `encode_query(text, config)` shim live in
# arxiv_radar_mcp/build_cache.py — both depend on the project-specific
# corpus loader, which corpus_core does not own.
