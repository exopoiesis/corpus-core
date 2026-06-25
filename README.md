# corpus-core

Shared infrastructure for corpus indexing + MCP search. Powers both
[arxiv-radar-mcp](https://github.com/exopoiesis/arxiv-radar-mcp) (the
public-data arXiv topical radar) and
[lab-corpus-mcp](https://github.com/exopoiesis/lab-corpus-mcp) (the
private multi-source PDF / video / lab-notes server).

What's inside:

| Module | Role |
|--------|------|
| `embeddings.py` | `Encoder` — lazy SentenceTransformer wrapper with model-aware query/passage prefixes, bf16 on CUDA, matryoshka truncation. `Encoder.unload()` drops the in-process model + frees CUDA VRAM (idempotent; next encode lazily re-loads). `EmbeddingIndex` — mmap'd float32 matrix + `row_for` mapping + metadata, atomic save/load. |
| `chunker.py` | `chunk_markdown(text, max_tokens) → list[Chunk]`. Section-aware split + paragraph overlap; rough but fast token estimator. |
| `corpus_index.py` | Chunk-level corpus search. `reindex(parse_dir, encoder, *, incremental)` — incremental encode + atomic swap. `search_paper_text` / `search_paper_semantic` / `similar_to_paper`. `is_junk_section` filter. |
| `search.py` | Abstract-level search primitives over `EmbeddingIndex`: `search_text` / `search_semantic` / `similar_to`. Paper-shaped records via `Protocol` — no host-project dep. |
| `jobs.py` | `JobRegistry` — ThreadPoolExecutor + persistent `jobs/<id>.json`. Disk-truth fallback in `get()` so a stuck-running cell doesn't lie about completed jobs. |
| `proxy.py` | Local stdio↔remote-HTTP bridge. `run_proxy(target, port, ssh_binary)` opens an SSH tunnel and forwards MCP traffic; `_bridge_loop` reconnects on backend disconnect. |
| `reranker.py` | `Reranker` — lazy CrossEncoder wrapper for hybrid-search re-scoring. `Reranker.unload()` mirrors `Encoder.unload()` for symmetric VRAM control after a batch. Local `RerankerConfig` dataclass. |
| `mcp_scaffold.py` | Generic MCP server scaffold: `make_method_dispatcher(handler, allowlist) → Dispatcher`, `build_mcp_app(server_name, tool_specs, dispatcher) → mcp.server.Server`, `serve_stdio` / `serve_streamable_http` transports with optional `BackgroundTaskFactory` list. |
| `http_fetch.py` | `fetch_url(url, dest_path) → FetchResult` — throttled GET with 429/503 retry + `Retry-After` + atomic file write. `fetch_arxiv_pdf(arxiv_id, dest_dir)` convenience wrapper. **`get_arxiv_throttle()` singleton** — process-wide 1 req / 3 sec budget shared by arxiv-radar's HTML/LaTeX fetcher and lab-corpus's `ingest_url` / `ingest_arxiv_pdf`, so the combined image never double-spams arxiv.org. |

## Install

```bash
pip install corpus-core            # once published to PyPI
# or, during dev:
pip install -e ../corpus-core
```

## Quick start

```python
from corpus_core import (
    Encoder, EmbeddingIndex,
    Chunk, chunk_markdown,
    search_text, search_semantic, similar_to,
    search_paper_text, search_paper_semantic, similar_to_paper,
    load_chunk_texts, reindex, is_junk_section,
    JobRegistry, JobHandle, JobError, Job,
    make_method_dispatcher, build_mcp_app,
    serve_stdio, serve_streamable_http,
    Dispatcher, BackgroundTaskFactory,
    fetch_url, fetch_arxiv_pdf,
    Throttle, get_arxiv_throttle,
    request_with_retry, FetchResult,
    ARXIV_RATE_LIMIT_S, DEFAULT_USER_AGENT,
)

# Submodule access also fine:
from corpus_core.embeddings import Encoder
from corpus_core.proxy import run_proxy
from corpus_core.reranker import Reranker, RerankerConfig
from corpus_core.http_fetch import fetch_url, get_arxiv_throttle
```

## Process-level singleton contract

**One cache_dir = one writing process.**
`corpus_core.jobs.JobRegistry` serialises reindex attempts within a process via
`acquire_reindex_lock()`. Running two separate processes against the same
`cache_dir` at the same time is not supported and will produce a corrupted
index. The lockfile (`<cache_dir>/fulltext/.reindex.lock`) records `pid` +
`hostname` + `start_time` so a crashed owner's lock can be recovered on the
next start (same-host pid dead = stale; foreign host = operator must remove
manually).

`corpus_core.embeddings.Encoder` and `corpus_core.http_fetch.get_arxiv_throttle()`
are process-level singletons -- construct one instance and inject it into both
arxiv-radar-mcp and lab-corpus-mcp via the `encoder=` / shared-throttle
parameters. Never instantiate two Encoders in the same process against the
same GPU (two Qwen3-4B bf16 models = ~16 GB, exhausts a 12 GB card).

## Invariants downstream packages must honour

* **Embedding cache layout**:
  * `<cache_dir>/embeddings.npy` — float32, L2-normalized, shape `(N, D)`.
  * `<cache_dir>/index.json` — `{model, dims, n, row_for, ...metadata}`.
  * Both written atomically (`*.tmp` → `os.replace`).
* **Job persistence schema**: `<cache_dir>/jobs/<job_id>.json` with
  fields `{job_id, kind, state, progress, n_total, n_done, started_at,
  finished_at, result, error, args}`. State ∈ {`pending`, `running`,
  `done`, `failed`, `orphaned`}.
* **Chunk metadata**: each chunk in `EmbeddingIndex.metadata["chunks"]`
  has `{arxiv_id, section, chunk_idx, n_chars, n_tokens_est}`. The
  `arxiv_id` field is the corpus-wide paper id — DOI / PMID / sha256 /
  any string the host project chooses.
* **Encoder config duck-type**: `Encoder.__init__(config)` reads
  `config.embeddings.{model, batch_size, target_dim, cache_dir}`.
  Pass any object with that shape. See `corpus_core.embeddings.Config`
  Protocol for the formal type.
* **HTTP fetch invariants** (`http_fetch.py`):
  * `fetch_url` writes atomically (`<dest>.tmp` → `os.replace`); on
    any failure (transport error, non-2xx, empty body) `dest_path`
    is not created or overwritten.
  * `Throttle` is one instance per source domain; all callers that
    share an instance share the budget. Use `get_arxiv_throttle()`
    for every arxiv.org GET so the combined image enforces 1 req /
    3 sec across both downstream servers.
  * `request_with_retry` retries only on 429/503; other status codes
    fall through after the first attempt. Honours `Retry-After` if
    present, else exponential backoff 3→6→12 sec.

## Used by

* [arxiv-radar-mcp](https://github.com/exopoiesis/arxiv-radar-mcp) --
  arxiv-only topical radar over the `daily-arxiv-*` fork family.
* [lab-corpus-mcp](https://github.com/exopoiesis/lab-corpus-mcp) --
  private PDF / DOCX / PPTX / image corpus parsed via MinerU; can
  also run combined with arxiv-radar-mcp on one Qwen instance to
  fit a 12 GB GPU.

> **Note: shared GPU hosting requires the combined-supervisor (DECISIONS-136).**
> arxiv-radar-mcp and lab-corpus-mcp each load Qwen3-Embedding-4B (~8 GB
> bf16). Running them as two independent standalone backends on the same GPU
> is **not supported** -- two copies total ~16 GB, causing OOM on 12 GB cards.
> The only supported topology for shared GPU hosting is the
> **combined-supervisor** in `lab-corpus-mcp`, which constructs one `Encoder`
> and injects it into both servers via the `encoder=` parameter on
> `RadarServer.__init__` and the equivalent in `lab-corpus-mcp`.

## Status

**v0.2.0, in production** as of 2026-05-24. Both downstream
projects (arxiv-radar-mcp and lab-corpus-mcp) install corpus-core
editable from the sibling repo. The combined `exopoiesis/lab-corpus-gpu`
image on gomer holds:

* 34,627 abstract embeddings (Qwen3-Embedding-4B native 2560 dims,
  L2-normalized) over 5 arxiv-radar fork sources.
* 466 fulltext chunks across 51 fetched arxiv papers + 54 chunks
  across 2 MinerU-parsed lab PDFs.
* All in one `corpus_core.embeddings.EmbeddingIndex` cache layout,
  encoded by a single shared Encoder instance.

Build-time `audit_image.py` in both downstream projects checks the
no-duplicate-distribution invariant — pip never ends up with two
copies of any package, including torch.

**HTTP fetch primitives added (2026-05-13)** to close
arxiv-radar-mcp's U14: `http_fetch.py` extracts the throttled GET +
429/503-retry pattern that previously lived inside arxiv-radar's
`fulltext.py`. Both servers now share the singleton arxiv throttle —
arxiv-radar uses it for HTML/LaTeX, lab-corpus uses it for
`ingest_url` / `ingest_arxiv_pdf` PDF downloads. Same module-global
lock across the whole combined image.

**VRAM unload added (2026-05-24, v0.2.0).** `Encoder.unload()` and
`Reranker.unload()` drop the in-process model and free CUDA VRAM via
`torch.cuda.empty_cache()` + `gc.collect()`. Both are idempotent and
guarded by the existing model-load lock so concurrent encodes either
complete first or re-load on next call. Downstream projects call
`Encoder.unload()` after heavy one-shot work (reindex, refresh, bulk
ingest) so a shared GPU host can use the freed VRAM for unrelated
compute. 119 corpus-core tests green.

Phase 3 extraction is complete; PyPI publication of `corpus-core`
deferred until the API stabilises through real-world ingest of more
than the current 53 papers.

## Tests

`pytest -q` from the repo root. The standalone test suite covers
`chunker`, `jobs`, `mcp_scaffold`, `proxy` invocation, `reranker`
config + lazy load, and basic embedding-index roundtrip with a
deterministic stub encoder. Heavier integration testing
(host-project Configs, real Qwen weights, real MCP sessions) lives
in the `arxiv-radar-mcp` and `lab-corpus-mcp` test suites.

## License

MIT.
