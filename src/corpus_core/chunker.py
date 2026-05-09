"""Markdown → chunk list, keyed by ## headings.

Used by the reindex flow: each enriched paper's full text (cached as
markdown under cache_dir/fulltext/sources/<id>.md) is split into chunks
that fit the encoder's max_seq_length window. Each chunk carries the
section name so search results can attribute "found in Methods of paper X".

Strategy:
  1. Scan for `## headings` (level-2 ATX). The arxiv HTML/LaTeX path
     emits these for top-level sections (Introduction, Methods, ...).
  2. Concatenate everything between two consecutive `##` lines into one
     section. The text before the first `##` is bundled as section "Header"
     (title, abstract, possibly authors).
  3. If a section's token count exceeds `max_tokens`, sub-split by
     paragraph boundaries (blank line). Sub-chunks try to maximize fit
     without splitting paragraphs in half.
  4. Empty / whitespace-only sections are dropped.

Token estimation here uses a fast char-based heuristic (~4 chars/token
for English with inline LaTeX). The exact tokenizer-aware count happens
later when sentence-transformers actually encodes — this is just to
decide where to split.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Roughly 4 characters per token for English + inline LaTeX. Empirically:
# - English prose ≈ 4 chars/token (BPE tokenizers)
# - LaTeX-heavy math ≈ 3 chars/token (more symbols → more tokens)
# Erring on the conservative side (4) means our chunks fit comfortably.
_CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    section: str       # heading text, e.g. "Methods" or "Header"
    chunk_idx: int     # index within section (0 if section fits whole)
    text: str          # the chunk body
    n_chars: int       # for telemetry
    n_tokens_est: int  # estimate, NOT exact tokenizer count


def estimate_tokens(text: str) -> int:
    """Cheap proxy for tokenizer length. ±20% of the real count, good enough
    to drive split decisions without paying a tokenizer-load cost."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


# Match level-2 ATX headings on their own line: `## Methods`. Captures the
# trailing heading text. Avoids matching `### subsection` (level-3+) and
# inline `# foo` (LaTeX comments inside fenced blocks).
_HEADING_RE = re.compile(r"^##[ \t]+(.+?)[ \t]*$", re.MULTILINE)


def split_by_headings(markdown: str) -> list[tuple[str, str]]:
    """Return [(section_name, body_text), ...] preserving order.

    Body text excludes the heading line itself. Whitespace at the section
    boundaries is stripped. The text before the first `##` is yielded as
    section name "Header" (typical: title + abstract pulled out of the
    fetcher).
    """
    matches = list(_HEADING_RE.finditer(markdown))
    if not matches:
        # No level-2 headings — treat the whole doc as one "Body" section.
        body = markdown.strip()
        return [("Body", body)] if body else []

    sections: list[tuple[str, str]] = []

    head_text = markdown[: matches[0].start()].strip()
    if head_text:
        sections.append(("Header", head_text))

    for i, m in enumerate(matches):
        section_name = m.group(1).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[body_start:body_end].strip()
        if body:
            sections.append((section_name, body))

    return sections


def split_long_section(
    text: str,
    max_tokens: int,
    overlap_ratio: float = 0.12,
) -> list[str]:
    """Sub-split a section that exceeds max_tokens into paragraph-aligned chunks.

    Greedy fill with paragraph-aligned overlap: walk paragraphs (separated
    by blank lines), accumulate while still under max_tokens, flush. When
    flushing, carry the trailing paragraphs of the just-emitted chunk into
    the start of the next one — up to roughly `overlap_ratio * max_tokens`
    tokens of overlap. This is the standard RAG "sliding window" technique
    adapted to paragraph boundaries, so we never split mid-word.

    `overlap_ratio=0.12` ≈ 12% overlap (e.g. ~500 tokens at max_tokens=4096).
    Set to 0 to disable.

    Single paragraphs that exceed max_tokens on their own are emitted as-is —
    the encoder will truncate them. We lose tail tokens but keep the chunk
    addressable; rare in practice with max_tokens >= 2048.
    """
    paragraphs = [p for p in re.split(r"\n[ \t]*\n", text) if p.strip()]
    if not paragraphs:
        return []

    overlap_target = int(max_tokens * overlap_ratio) if overlap_ratio > 0 else 0
    chunks: list[str] = []
    cur_parts: list[str] = []
    cur_tokens = 0

    for p in paragraphs:
        p_tokens = estimate_tokens(p)
        if cur_parts and cur_tokens + p_tokens > max_tokens:
            chunks.append("\n\n".join(cur_parts))
            # Carry tail paragraphs of the just-flushed chunk into the next
            # one. Walk backwards, take whole paragraphs while their total
            # is under overlap_target. Skip entirely when overlap_target is
            # 0 (caller asked for no overlap).
            carry: list[str] = []
            carry_tokens = 0
            if overlap_target > 0:
                for prev in reversed(cur_parts):
                    prev_tokens = estimate_tokens(prev)
                    # Stop if adding this would push us past the target,
                    # but always carry at least one paragraph (so single-
                    # oversize-paragraph edge cases still get context).
                    if carry and carry_tokens + prev_tokens > overlap_target:
                        break
                    carry.insert(0, prev)
                    carry_tokens += prev_tokens
            cur_parts = carry + [p]
            cur_tokens = carry_tokens + p_tokens
        else:
            cur_parts.append(p)
            cur_tokens += p_tokens

    if cur_parts:
        chunks.append("\n\n".join(cur_parts))

    return chunks


def chunk_markdown(
    markdown: str,
    max_tokens: int = 4_096,
    overlap_ratio: float = 0.12,
) -> list[Chunk]:
    """Top-level: markdown → list[Chunk] ready for embedding.

    `max_tokens` is the encoder's effective `max_seq_length` for the chunk —
    chunks larger than this get truncated by the encoder, so the chunker
    splits ahead of time. Default 4096 reflects 2026-05-02 perf rebalance:
      * empirically 12288 long-bucket chunks took 51 sec each on RTX 4070
        (96% of reindex wall time), even though they were only 12% of chunks
      * 4096 puts most sections in the medium bucket (~1 sec/chunk), giving
        ~5-10× reindex speedup
      * granularity bonus: top-k results now point at sub-section level,
        not "Methods as a whole"

    `overlap_ratio` controls the paragraph-aligned overlap between
    adjacent sub-chunks of the same section (see split_long_section).
    Default 0.12 ≈ ~500 tokens at max=4096 — last 1-2 paragraphs of the
    previous chunk are repeated at the start of the next.

    Returns chunks in document order — the section field on each chunk is
    enough to attribute results back to a paper region.
    """
    sections = split_by_headings(markdown)
    out: list[Chunk] = []

    for section_name, body in sections:
        body_tokens = estimate_tokens(body)
        if body_tokens <= max_tokens:
            out.append(Chunk(
                section=section_name,
                chunk_idx=0,
                text=body,
                n_chars=len(body),
                n_tokens_est=body_tokens,
            ))
        else:
            sub_texts = split_long_section(
                body, max_tokens=max_tokens, overlap_ratio=overlap_ratio,
            )
            for idx, sub in enumerate(sub_texts):
                out.append(Chunk(
                    section=section_name,
                    chunk_idx=idx,
                    text=sub,
                    n_chars=len(sub),
                    n_tokens_est=estimate_tokens(sub),
                ))

    return out
