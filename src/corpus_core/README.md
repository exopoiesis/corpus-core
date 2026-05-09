# corpus-core

Shared infrastructure for corpus indexing + MCP search.

This package lives **inside** `arxiv-radar-mcp` for now (Phase 1 of the
extraction plan; see `arxiv-radar-mcp/docs/PLAN_CORE_EXTRACTION.md`).
Phase 3 will move it to its own repo and PyPI distribution `corpus-core`.

Designed to be the shared dependency of:
- **`arxiv-radar-mcp`** — arxiv-only topical radar (this repo).
- **`lab-corpus-mcp`** — multi-source PDF/video/PubMed/Scholar personal
  corpus (next session, separate repo).

Both downstream projects own their own ingestion (different paper
sources, different schema specifics) but reuse `corpus-core` for
embedding, chunking, vector search, async jobs, and MCP transport.

## Public API

```python
from corpus_core import (
    Encoder, EmbeddingIndex,             # embedding model + index
    Chunk, chunk_markdown,                # markdown → chunks
    FULLTEXT_MAX_SEQ_LENGTH,
    search_text, search_semantic, similar_to,
                                          # abstract-level search
    search_paper_text, search_paper_semantic, similar_to_paper,
                                          # chunk-level search
    load_chunk_texts, reindex,            # corpus index lifecycle
    is_junk_section,
    JobRegistry, JobHandle, JobError, Job,
                                          # async background jobs
    make_method_dispatcher,               # generic dispatcher builder
    build_mcp_app,                        # mcp.server.Server constructor
    serve_stdio, serve_streamable_http,   # transport entry points
    Dispatcher, BackgroundTaskFactory,    # type aliases
)

# Submodule access also fine:
from corpus_core.embeddings import Encoder
from corpus_core.proxy import run_proxy   # stdio↔HTTP bridge
from corpus_core.reranker import Reranker
from corpus_core.mcp_scaffold import serve_streamable_http
```

## Module map

| Module | Role |
|--------|------|
| `embeddings.py` | `Encoder` (lazy SentenceTransformer wrapper, model-aware prefixes, bf16 on CUDA), `EmbeddingIndex` (mmap matrix + row_for + metadata) |
| `chunker.py` | Markdown → `Chunk` with section-aware split + paragraph overlap; `chunk_markdown(text, max_tokens)` |
| `corpus_index.py` | Chunk-level corpus search (was `fulltext_index.py` in arxiv-radar-mcp); incremental reindex, junk-section filter, adaptive bucket encoding |
| `search.py` | Abstract-level search primitives: `search_text` / `search_semantic` / `similar_to` over `EmbeddingIndex` |
| `jobs.py` | `JobRegistry` — ThreadPoolExecutor + persistent `jobs/<id>.json`. Disk-truth fallback in `get()` (U1 fix) |
| `proxy.py` | Local stdio→remote-HTTP bridge with reconnect-loop (U8 Option B); `run_proxy(target, port, ssh_binary)` |
| `reranker.py` | Cross-encoder reranker class. Kept as utility but no longer wired into any tool (РЕШЕНИЕ-010 in arxiv-radar-mcp). |
| `mcp_scaffold.py` | Generic MCP server scaffold: `make_method_dispatcher` (handler+allowlist → dispatcher), `build_mcp_app` (server-name + tool-specs + dispatcher → `mcp.server.Server`), `serve_stdio` / `serve_streamable_http` transports, optional `BackgroundTaskFactory` list (warm-up, refresh, etc.). Phase 1.5 — extracted from `arxiv-radar-mcp/server.py` so `lab-corpus-mcp` can build its own MCP server on top of the same primitives without depending on the arxiv shell. |

## Invariants downstream projects must honour

- **Embedding cache layout**:
  `<cache_dir>/embeddings.npy` (float32, L2-normalized, shape `(N, D)`)
  + `<cache_dir>/index.json` (`{model, dims, n, row_for, ...metadata}`).
  Both written atomically (`*.tmp` → `rename`).
- **Job persistence schema**: `<cache_dir>/jobs/<job_id>.json` with
  fields `{job_id, kind, state, progress, n_total, n_done, started_at,
  finished_at, result, error, args}`. State ∈ {`pending`, `running`,
  `done`, `failed`, `orphaned`}.
- **Chunk metadata**: each chunk in `EmbeddingIndex.metadata["chunks"]`
  has `{arxiv_id, section, chunk_idx, n_chars, n_tokens_est}` — but
  `arxiv_id` is just the corpus-wide paper id (DOI / PMID / sha256
  also OK — downstream chooses).
- **Encoder config duck-type**: `Encoder.__init__` reads
  `config.embeddings.{model, batch_size, target_dim}`. Downstream's
  config dataclass needs those three fields; everything else is theirs.

## What is NOT in corpus-core (lives in downstream shells)

- arxiv-specific HTML/LaTeX cascade fetcher
  (→ `arxiv-radar-mcp/fulltext.py`)
- arxiv-radar-* fork loader
  (→ `arxiv-radar-mcp/corpus.py`)
- daily git-pull refresh
  (→ `arxiv-radar-mcp/refresh.py`)
- `relevance_filter` + canonical tags loaders
  (project-specific)
- TOOL_SPECS catalogue (each downstream owns its tool surface)

## Tests

For now, the test suite under `arxiv-radar-mcp/tests/` doubles as
`corpus-core` tests. When Phase 3 splits corpus-core to its own repo,
relevant tests move with it (`test_jobs.py`, `test_embeddings.py`,
`test_chunker.py`, `test_proxy.py`, `test_search_text.py`,
`test_fulltext_index.py`, `test_reranker.py`, plus the dispatcher /
`_build_mcp_app` cases from `test_server.py` + `test_server_http.py`
that exercise the scaffold via the arxiv-radar shell).
