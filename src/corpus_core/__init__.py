"""corpus-core — shared infrastructure for corpus indexing + MCP search.

Used by `arxiv-radar-mcp` (arxiv-only topical radar) and the
forthcoming `lab-corpus-mcp` (multi-source PDF/video/PubMed/Scholar
personal corpus). See `arxiv-radar-mcp/docs/PLAN_CORE_EXTRACTION.md`
for the architecture rationale and what stays in each project shell.

Phase 1 (in-place extraction inside arxiv-radar-mcp). Phase 3 will
break this out into its own repo + PyPI package.

Public API surface (re-exports for convenience):
    Encoder, EmbeddingIndex                — embedding model + index
    chunk_markdown, Chunk, FULLTEXT_MAX_SEQ_LENGTH
                                            — markdown → chunk splitter
    search_text, search_semantic, similar_to
                                            — abstract-level search primitives
    search_paper_text, search_paper_semantic, similar_to_paper
                                            — chunk-level search primitives
    load_chunk_texts, reindex, is_junk_section
                                            — corpus index lifecycle
    JobRegistry, JobHandle, JobError, Job   — async background jobs
    make_method_dispatcher, build_mcp_app, serve_stdio,
    serve_streamable_http, Dispatcher, BackgroundTaskFactory
                                            — generic MCP server scaffold

Submodule access (`from corpus_core.embeddings import ...`) is also
fully supported — the explicit re-exports here are convenience only.
"""
from corpus_core.chunker import (
    Chunk,
    chunk_markdown,
)
from corpus_core.corpus_index import (
    FULLTEXT_MAX_SEQ_LENGTH,
    is_junk_section,
    load_chunk_texts,
    reindex,
    search_paper_semantic,
    search_paper_text,
    similar_to_paper,
)
from corpus_core.embeddings import (
    EmbeddingIndex,
    Encoder,
    passage_prefix,
    query_prefix,
)
from corpus_core.jobs import (
    Job,
    JobError,
    JobHandle,
    JobRegistry,
)
from corpus_core.mcp_scaffold import (
    BackgroundTaskFactory,
    Dispatcher,
    build_mcp_app,
    make_method_dispatcher,
    serve_stdio,
    serve_streamable_http,
)
from corpus_core.search import (
    search_semantic,
    search_text,
    similar_to,
)

__all__ = [
    # embeddings
    "Encoder",
    "EmbeddingIndex",
    "passage_prefix",
    "query_prefix",
    # chunker
    "Chunk",
    "chunk_markdown",
    # search (abstract level)
    "search_text",
    "search_semantic",
    "similar_to",
    # corpus index (chunk level)
    "FULLTEXT_MAX_SEQ_LENGTH",
    "is_junk_section",
    "load_chunk_texts",
    "reindex",
    "search_paper_text",
    "search_paper_semantic",
    "similar_to_paper",
    # jobs
    "Job",
    "JobError",
    "JobHandle",
    "JobRegistry",
    # mcp scaffold
    "BackgroundTaskFactory",
    "Dispatcher",
    "build_mcp_app",
    "make_method_dispatcher",
    "serve_stdio",
    "serve_streamable_http",
]
