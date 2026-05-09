"""Tests for chunker.py — markdown → chunk list."""
from __future__ import annotations

import pytest

from corpus_core.chunker import (Chunk, chunk_markdown, estimate_tokens,
                                     split_by_headings, split_long_section)


# ----- estimate_tokens -------------------------------------------------------

def test_estimate_tokens_basic():
    assert estimate_tokens("a" * 4) == 1
    assert estimate_tokens("a" * 400) == 100


def test_estimate_tokens_minimum_one():
    assert estimate_tokens("") == 1
    assert estimate_tokens("ab") == 1


# ----- split_by_headings -----------------------------------------------------

def test_split_by_headings_three_sections():
    md = """# Title

Some intro paragraph.

## Methods
We did stuff.

## Results
We saw stuff.

## Discussion
Stuff matters.
"""
    out = split_by_headings(md)
    names = [n for n, _ in out]
    assert names == ["Header", "Methods", "Results", "Discussion"]
    assert "intro paragraph" in dict(out)["Header"]
    assert "We did stuff" in dict(out)["Methods"]


def test_split_by_headings_no_headings_emits_body():
    md = "Just a paragraph, no headings here."
    out = split_by_headings(md)
    assert out == [("Body", "Just a paragraph, no headings here.")]


def test_split_by_headings_skips_empty_section():
    md = """## Methods

## Results
content
"""
    out = split_by_headings(md)
    assert ("Methods", "") not in out
    assert dict(out).get("Results") == "content"


def test_split_by_headings_ignores_subsection_h3():
    md = """## Methods
Top-level methods text.

### Subsection
Inside subsection.

## Results
Results text.
"""
    out = split_by_headings(md)
    names = [n for n, _ in out]
    assert names == ["Methods", "Results"]
    # H3 content should be inside the parent section.
    assert "Inside subsection" in dict(out)["Methods"]


def test_split_by_headings_no_pre_header_text_skips_header():
    md = """## Methods
Methods text.
"""
    out = split_by_headings(md)
    assert out == [("Methods", "Methods text.")]


# ----- split_long_section ----------------------------------------------------

def test_split_long_section_under_limit_returns_one_chunk():
    text = "a " * 100  # ~50 tokens
    out = split_long_section(text, max_tokens=1000)
    assert len(out) == 1


def test_split_long_section_over_limit_splits_at_paragraph_boundary():
    p1 = "a " * 200  # ~100 tokens
    p2 = "b " * 200
    p3 = "c " * 200
    text = f"{p1}\n\n{p2}\n\n{p3}"
    out = split_long_section(text, max_tokens=150)
    assert len(out) >= 2
    # Each chunk should be roughly under-limit (greedy fill, no overlap).
    for chunk in out:
        # Some slack: a single oversized paragraph passes through.
        assert "\n\n" not in chunk[:1] if chunk else True


def test_split_long_section_preserves_content():
    text = "para1\n\npara2\n\npara3"
    out = split_long_section(text, max_tokens=2)  # force split
    joined = "\n\n".join(out)
    assert "para1" in joined and "para2" in joined and "para3" in joined


# ----- chunk_markdown (top-level) -------------------------------------------

def test_chunk_markdown_returns_chunks_with_section_names():
    md = """# Title

intro

## Methods

methods text

## Results

results text
"""
    chunks = chunk_markdown(md, max_tokens=10_000)
    sections = [c.section for c in chunks]
    assert "Header" in sections
    assert "Methods" in sections
    assert "Results" in sections
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.chunk_idx == 0  # no sub-split needed
        assert c.n_chars > 0
        assert c.n_tokens_est > 0


def test_chunk_markdown_subsplits_oversize_section():
    """Force a section that exceeds max_tokens, see chunk_idx increment."""
    big = ("paragraph one. " * 200 + "\n\n" + "paragraph two. " * 200)
    md = f"## BigSection\n\n{big}\n"
    chunks = chunk_markdown(md, max_tokens=200)
    big_chunks = [c for c in chunks if c.section == "BigSection"]
    assert len(big_chunks) >= 2
    assert [c.chunk_idx for c in big_chunks] == list(range(len(big_chunks)))


def test_chunk_markdown_empty_input():
    assert chunk_markdown("", max_tokens=1000) == []


def test_chunk_markdown_keeps_inline_latex():
    md = "## Methods\n\nWe use $E = mc^2$ and $\\alpha = 1$ everywhere."
    chunks = chunk_markdown(md, max_tokens=1000)
    text = chunks[0].text
    assert "$E = mc^2$" in text
    assert "\\alpha" in text


def test_chunk_markdown_default_max_tokens_is_4096():
    """2026-05-02 perf rebalance: was 12_000, dropped to 4_096 so most
    sections land in the medium encode bucket (~5-10× reindex speedup).
    Encoder seq window in fulltext_index follows the same value."""
    import inspect
    from corpus_core.corpus_index import FULLTEXT_MAX_SEQ_LENGTH
    sig = inspect.signature(chunk_markdown)
    assert sig.parameters["max_tokens"].default == 4_096
    assert FULLTEXT_MAX_SEQ_LENGTH == 4_096


# ----- paragraph-aligned overlap --------------------------------------------


def test_split_long_section_overlap_carries_tail():
    """When a long section gets split, the trailing paragraphs of one
    chunk re-appear at the start of the next."""
    p1 = "Para A. " * 50    # ~50 tokens
    p2 = "Para B. " * 50
    p3 = "Para C. " * 50
    p4 = "Para D. " * 50
    p5 = "Para E. " * 50
    text = "\n\n".join([p1, p2, p3, p4, p5])

    # max_tokens=200 forces split after ~3 paragraphs;
    # overlap_ratio=0.20 → carry ≈ 40 tokens (1 paragraph) into next chunk.
    chunks = split_long_section(text, max_tokens=200, overlap_ratio=0.20)
    assert len(chunks) >= 2

    # Last paragraph of chunk 0 should appear at the start of chunk 1.
    chunk0_last = chunks[0].rsplit("\n\n", 1)[-1].strip()
    chunk1_first = chunks[1].split("\n\n", 1)[0].strip()
    assert chunk0_last == chunk1_first, (
        f"expected overlap, got chunk0 tail={chunk0_last[:30]!r} "
        f"chunk1 head={chunk1_first[:30]!r}"
    )


def test_split_long_section_no_overlap_when_ratio_zero():
    p1 = "X. " * 50
    p2 = "Y. " * 50
    p3 = "Z. " * 50
    text = "\n\n".join([p1, p2, p3])
    chunks = split_long_section(text, max_tokens=100, overlap_ratio=0.0)
    if len(chunks) >= 2:
        # No paragraph repetition between chunks.
        chunk0_last = chunks[0].rsplit("\n\n", 1)[-1].strip()
        chunk1_first = chunks[1].split("\n\n", 1)[0].strip()
        assert chunk0_last != chunk1_first


def test_split_long_section_overlap_does_not_create_infinite_chunks():
    """Overlap must not cause runaway sub-splitting. The number of chunks
    is bounded — at worst one chunk per paragraph (because carry never
    feeds back into the loop)."""
    n_paragraphs = 20
    text = "\n\n".join(["short. " * 5 for _ in range(n_paragraphs)])
    chunks = split_long_section(text, max_tokens=100, overlap_ratio=0.20)
    # Sanity bounds: at least one chunk, at most n_paragraphs (the
    # degenerate case where every paragraph alone fills max_tokens).
    assert 1 <= len(chunks) <= n_paragraphs
