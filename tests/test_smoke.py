"""Smoke test — every submodule must import without arxiv_radar_mcp installed.

This is the load-bearing assertion for Phase 3 (corpus-core extracted
to its own repo): the moment any module gains a hidden import of
`arxiv_radar_mcp` or `lab_corpus_mcp`, this suite turns red.
"""
from __future__ import annotations


def test_top_level_imports():
    import corpus_core
    assert corpus_core is not None


def test_submodule_imports():
    # Every public module — failure here means the standalone install
    # depends on a host project we shouldn't.
    from corpus_core import chunker, corpus_index, embeddings, jobs
    from corpus_core import mcp_scaffold, proxy, reranker, search
    for mod in (chunker, corpus_index, embeddings, jobs,
                mcp_scaffold, proxy, reranker, search):
        assert mod is not None


def test_public_api_reexports():
    """The `__init__.py` advertises a curated public surface — verify
    none of the names accidentally vanished."""
    from corpus_core import (
        Encoder, EmbeddingIndex,
        Chunk, chunk_markdown,
        FULLTEXT_MAX_SEQ_LENGTH, is_junk_section,
        load_chunk_texts, reindex,
        search_paper_text, search_paper_semantic, similar_to_paper,
        search_text, search_semantic, similar_to,
        Job, JobError, JobHandle, JobRegistry,
        BackgroundTaskFactory, Dispatcher,
        build_mcp_app, make_method_dispatcher,
        serve_stdio, serve_streamable_http,
    )
    # Just touch each binding so flake8 / linters don't strip them.
    for obj in (
        Encoder, EmbeddingIndex,
        Chunk, chunk_markdown,
        FULLTEXT_MAX_SEQ_LENGTH, is_junk_section,
        load_chunk_texts, reindex,
        search_paper_text, search_paper_semantic, similar_to_paper,
        search_text, search_semantic, similar_to,
        Job, JobError, JobHandle, JobRegistry,
        BackgroundTaskFactory, Dispatcher,
        build_mcp_app, make_method_dispatcher,
        serve_stdio, serve_streamable_http,
    ):
        assert obj is not None


def test_no_host_project_imports():
    """Concrete check: corpus_core must NEVER import from arxiv_radar_mcp
    or lab_corpus_mcp. Phase 3 architectural invariant."""
    import sys

    # Force-load every corpus_core submodule.
    import corpus_core  # noqa: F401
    from corpus_core import (chunker, corpus_index, embeddings, jobs,  # noqa: F401
                             mcp_scaffold, proxy, reranker, search)  # noqa: F401

    leaked = [name for name in sys.modules
              if name.startswith("arxiv_radar_mcp")
              or name.startswith("lab_corpus_mcp")]
    assert leaked == [], (
        f"corpus_core leaked imports of host-project modules: {leaked}. "
        "Phase 3 invariant — corpus_core must stay standalone."
    )
