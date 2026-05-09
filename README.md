# corpus-core

Shared infrastructure for corpus indexing + MCP search. Powers both
[arxiv-radar-mcp](https://github.com/exopoiesis/arxiv-radar-mcp) (the
public-data arXiv topical radar) and
[lab-corpus-mcp](https://github.com/exopoiesis/lab-corpus-mcp) (the
private multi-source PDF / video / lab-notes server).

What's inside:

| Module | Role |
|--------|------|
| `embeddings.py` | `Encoder` — lazy SentenceTransformer wrapper with model-aware query/passage prefixes, bf16 on CUDA, matryoshka truncation. `EmbeddingIndex` — mmap'd float32 matrix + `row_for` mapping + metadata, atomic save/load. |
| `chunker.py` | `chunk_markdown(text, max_tokens) → list[Chunk]`. Section-aware split + paragraph overlap; rough but fast token estimator. |
| `corpus_index.py` | Chunk-level corpus search. `reindex(parse_dir, encoder, *, incremental)` — incremental encode + atomic swap. `search_paper_text` / `search_paper_semantic` / `similar_to_paper`. `is_junk_section` filter. |
| `search.py` | Abstract-level search primitives over `EmbeddingIndex`: `search_text` / `search_semantic` / `similar_to`. Paper-shaped records via `Protocol` — no host-project dep. |
| `jobs.py` | `JobRegistry` — ThreadPoolExecutor + persistent `jobs/<id>.json`. Disk-truth fallback in `get()` so a stuck-running cell doesn't lie about completed jobs. |
| `proxy.py` | Local stdio↔remote-HTTP bridge. `run_proxy(target, port, ssh_binary)` opens an SSH tunnel and forwards MCP traffic; `_bridge_loop` reconnects on backend disconnect. |
| `reranker.py` | `Reranker` — lazy CrossEncoder wrapper for hybrid-search re-scoring. Local `RerankerConfig` dataclass. |
| `mcp_scaffold.py` | Generic MCP server scaffold: `make_method_dispatcher(handler, allowlist) → Dispatcher`, `build_mcp_app(server_name, tool_specs, dispatcher) → mcp.server.Server`, `serve_stdio` / `serve_streamable_http` transports with optional `BackgroundTaskFactory` list. |

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
)

# Submodule access also fine:
from corpus_core.embeddings import Encoder
from corpus_core.proxy import run_proxy
from corpus_core.reranker import Reranker, RerankerConfig
```

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

## Used by

* [arxiv-radar-mcp](https://github.com/exopoiesis/arxiv-radar-mcp) —
  arxiv-only topical radar over the `daily-arxiv-*` fork family.
* [lab-corpus-mcp](https://github.com/exopoiesis/lab-corpus-mcp) —
  private PDF / DOCX / PPTX / image corpus parsed via MinerU; can
  also run combined with arxiv-radar-mcp on one Qwen instance to
  fit a 12 GB GPU.

## Tests

`pytest -q` from the repo root. The standalone test suite covers
`chunker`, `jobs`, `mcp_scaffold`, `proxy` invocation, `reranker`
config + lazy load, and basic embedding-index roundtrip with a
deterministic stub encoder. Heavier integration testing
(host-project Configs, real Qwen weights, real MCP sessions) lives
in the `arxiv-radar-mcp` and `lab-corpus-mcp` test suites.

## License

MIT.
