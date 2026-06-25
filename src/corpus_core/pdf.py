"""PDF parsing primitives for corpus-core (optional extra [pdf]).

Provides a clean parse API on top of MinerU, independent of lab-corpus's
LabPaper schema / paper_id logic. Both arxiv-radar-mcp and lab-corpus-mcp
depend on this module; it must stay cheap to import (no MinerU loaded at
module import time -- only inside function bodies).

Public API
----------
DEFAULT_BACKEND : str
    MinerU backend constant ("pipeline"); override per-call if needed.
MineruRunner : TypeAlias
    (input_file, output_dir, timeout_seconds) -> Path to produced markdown.
    Test-injection seam -- identical contract to lab_corpus_mcp.ingest.MineruRunner.
PdfParseResult : dataclass
    Result of parse_pdf(). Fields: markdown, images, n_chars, backend,
    media_subdir_in_md.  The last field is the subdirectory name MinerU
    places images under (always "images" for MinerU pipeline); callers
    that need a different name in the final markdown must rewrite refs
    themselves.

parse_pdf(pdf_path, *, media_out_dir, backend, runner) -> PdfParseResult
    Parse one PDF.  Puts image files under media_out_dir.  Markdown refs
    point to media_subdir_in_md/<name> relative paths (NOT media_out_dir).

unload_pdf_models() -> bool
    Release MinerU VRAM singletons.  Idempotent.

is_pdf_parser_available() -> bool
    True iff MinerU is importable (lazy; safe to call without MinerU installed).

looks_like_pdf_stub(markdown) -> bool
    Heuristic: True when the markdown looks like a scan-only / failed parse
    (very short or almost entirely whitespace).
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

LOG = logging.getLogger(__name__)

# ---- Public constants --------------------------------------------------------

# Default MinerU backend.  "pipeline" (layout CNN + OCR + table) is
# validated on a 12 GB GPU when sharing VRAM with our Qwen3-Embedding-4B.
# "vlm-transformers" (1.2B Qwen2-VL) OOMs / wedges on the same setup.
DEFAULT_BACKEND = "pipeline"

# Name of the subdirectory MinerU writes images into, relative to the
# markdown file.  The value is part of the media-translation contract:
# callers that expose a different public media dir (e.g. arxiv-radar uses
# "<id>.media/") must rewrite refs from this constant to their target.
_MINERU_IMAGES_SUBDIR = "images"


# ---- Test-injection seam -----------------------------------------------------

MineruRunner = Callable[[Path, Path, int], Path]
"""Signature: (input_file, output_dir, timeout_seconds) -> path-to-produced-markdown.

Real production path uses _default_mineru_runner; tests inject a fake so
they never touch the 2 GB MinerU install.
"""


# ---- Public dataclasses ------------------------------------------------------

@dataclass
class PdfParseResult:
    """Result of a single parse_pdf() call.

    Fields
    ------
    markdown : str
        Full markdown text produced by MinerU.
    images : list[dict]
        Per-image dicts {name: str, abs_path: str}.  name is the bare
        filename (no directory prefix); abs_path is the absolute path
        where the file now lives inside media_out_dir.
    n_chars : int
        len(markdown).
    backend : str
        The MinerU backend that was used.
    media_subdir_in_md : str
        The subdir name the markdown refs use for images (always "images"
        for MinerU pipeline output).  Rewrite this in refs if you need a
        different public name.
    """
    markdown: str
    images: list[dict] = field(default_factory=list)
    n_chars: int = 0
    backend: str = DEFAULT_BACKEND
    media_subdir_in_md: str = _MINERU_IMAGES_SUBDIR


# ---- Internal helpers --------------------------------------------------------

def _default_mineru_runner(
    input_file: Path,
    output_dir: Path,
    timeout: int,
    *,
    backend: str = DEFAULT_BACKEND,
) -> Path:
    """Real MinerU runner.  Lazy import so module-level import stays cheap.

    Identical behaviour to lab_corpus_mcp.ingest._default_mineru_runner
    (which now delegates here via alias).  Returns the path to the produced
    markdown file.

    The timeout parameter is accepted for API compatibility but not
    enforced -- do_parse is a sync call; callers wrap it with their own
    watchdog if needed.
    """
    # Lazy import: keeps `from corpus_core.pdf import ...` cheap on hosts
    # without MinerU installed (test laptops, CI without GPU).
    from mineru.cli.common import do_parse  # noqa: PLC0415

    del timeout  # not enforced; see docstring

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_file.stem
    pdf_bytes = input_file.read_bytes()
    parse_method = "auto"

    LOG.info("mineru (library): parse %s backend=%s", input_file.name, backend)
    try:
        do_parse(
            output_dir=str(output_dir),
            pdf_file_names=[stem],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=["ch"],
            backend=backend,
            parse_method=parse_method,
            formula_enable=True,
            table_enable=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise PdfParseError(
            f"mineru library call failed: {type(exc).__name__}: {exc}"
        ) from exc

    candidate = output_dir / stem / parse_method / f"{stem}.md"
    if candidate.exists():
        return candidate
    fallbacks = list(output_dir.rglob("*.md"))
    if not fallbacks:
        raise PdfParseError(f"mineru produced no markdown under {output_dir}")
    return fallbacks[0]


class PdfParseError(RuntimeError):
    """Raised when parsing fails (MinerU error, no output, etc.)."""


# ---- Public functions --------------------------------------------------------

def parse_pdf(
    pdf_path: Path,
    *,
    media_out_dir: Path,
    backend: str = DEFAULT_BACKEND,
    runner: MineruRunner | None = None,
) -> PdfParseResult:
    """Parse a PDF and return clean PdfParseResult.

    Puts image files into media_out_dir (creating it if needed).  The
    returned markdown uses refs of the form "<media_subdir_in_md>/<name>"
    (i.e. "images/<name>"), which is MinerU's native output format.  If
    you need a different public dir name in the markdown, call
    rewrite_image_refs() on the result afterwards.

    Parameters
    ----------
    pdf_path : Path
        Absolute path to the PDF to parse.
    media_out_dir : Path
        Directory where image files will be written.  Created automatically.
    backend : str
        MinerU backend ("pipeline" or "vlm-transformers").  Ignored when
        runner is supplied.
    runner : MineruRunner | None
        Test-injection seam.  When None (production), uses
        _default_mineru_runner with the specified backend.

    Returns
    -------
    PdfParseResult

    Raises
    ------
    PdfParseError
        When MinerU is not installed, or parsing fails, or produces no output.
    """
    if not pdf_path.exists():
        raise PdfParseError(f"PDF not found: {pdf_path}")

    with tempfile.TemporaryDirectory() as td:
        tmp_out = Path(td)
        if runner is not None:
            produced_md = runner(pdf_path, tmp_out, 600)
        else:
            produced_md = _default_mineru_runner(
                pdf_path, tmp_out, 600, backend=backend
            )

        markdown = produced_md.read_text(encoding="utf-8")
        images = _collect_images(produced_md, media_out_dir)

    return PdfParseResult(
        markdown=markdown,
        images=images,
        n_chars=len(markdown),
        backend=backend,
        media_subdir_in_md=_MINERU_IMAGES_SUBDIR,
    )


def _collect_images(produced_md: Path, media_out_dir: Path) -> list[dict]:
    """Copy images from MinerU's `images/` sibling to media_out_dir.

    Returns list of {name, abs_path} for each image that was copied.
    media_out_dir is created only when there are images to copy.
    """
    src = produced_md.parent / _MINERU_IMAGES_SUBDIR
    if not src.exists() or not any(src.iterdir()):
        return []
    media_out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, media_out_dir, dirs_exist_ok=True)
    result = []
    for img in sorted(media_out_dir.iterdir()):
        if img.is_file():
            result.append({"name": img.name, "abs_path": str(img.resolve())})
    return result


def unload_pdf_models() -> bool:
    """Release MinerU's cached pipeline / hybrid model singletons.

    Idempotent.  Returns True if anything was actually released, False
    when no MinerU model had been loaded (e.g. a server that only answered
    search queries this lifetime, or MinerU is not installed).

    Implementation mirrors lab_corpus_mcp.ingest.unload_mineru_models;
    lab-corpus now delegates to this function.
    """
    import gc  # noqa: PLC0415

    try:
        from mineru.backend.pipeline.model_init import (  # noqa: PLC0415
            AtomModelSingleton,
            HybridModelSingleton,
        )
    except ImportError:
        return False

    released_any = False
    for singleton_cls in (AtomModelSingleton, HybridModelSingleton):
        cached = getattr(singleton_cls, "_models", None)
        if cached:
            released_any = True
            cached.clear()

    if not released_any:
        return False

    gc.collect()
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass

    LOG.info("unloaded MinerU pipeline + hybrid model singletons (corpus_core.pdf)")
    return True


def is_pdf_parser_available() -> bool:
    """True iff MinerU (mineru) package is importable.

    Lazy check -- never loads the model weights, just probes import.
    Safe to call on any host; returns False on hosts without MinerU.
    """
    try:
        import importlib  # noqa: PLC0415
        importlib.import_module("mineru")
        return True
    except ImportError:
        return False


# Stub detection thresholds.  A PDF parse is considered a stub when:
#   - total chars < _STUB_MIN_CHARS, OR
#   - non-whitespace chars / total chars < _STUB_MIN_CONTENT_RATIO
# Calibrated on MinerU output for scan-only PDFs (image-only, no OCR layer):
# those produce near-empty markdown (< 50 chars, mostly just the title line).
# We keep the threshold at 100 so a single-sentence abstract passes.
_STUB_MIN_CHARS = 100
_STUB_MIN_CONTENT_RATIO = 0.10


def looks_like_pdf_stub(markdown: str) -> bool:
    """True when the markdown looks like a failed / scan-only PDF parse.

    Heuristic based on total character count and non-whitespace density.
    A scan-only PDF (no OCR layer) produces near-empty MinerU output;
    a table-of-contents-only parse produces a very low content ratio.

    Parameters
    ----------
    markdown : str
        The markdown to evaluate.

    Returns
    -------
    bool
        True when the markdown is likely a stub (not useful for search).
    """
    if not markdown:
        return True
    n_total = len(markdown)
    if n_total < _STUB_MIN_CHARS:
        return True
    n_nonws = sum(1 for ch in markdown if not ch.isspace())
    ratio = n_nonws / n_total
    return ratio < _STUB_MIN_CONTENT_RATIO


__all__ = [
    "DEFAULT_BACKEND",
    "MineruRunner",
    "PdfParseError",
    "PdfParseResult",
    "is_pdf_parser_available",
    "looks_like_pdf_stub",
    "parse_pdf",
    "unload_pdf_models",
]
