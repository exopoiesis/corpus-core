"""Tests for corpus_core.pdf -- PDF parsing primitives.

All tests run without MinerU installed:
  - parse_pdf() uses injected fake runner (test-seam).
  - is_pdf_parser_available() is tested via sys.modules injection.
  - unload_pdf_models() is tested via sys.modules injection.
  - looks_like_pdf_stub() is pure Python logic.
  - test_no_host_project_imports (from test_smoke.py) stays green because
    corpus_core.pdf uses only lazy imports.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from corpus_core.pdf import (
    DEFAULT_BACKEND,
    PdfParseError,
    PdfParseResult,
    is_pdf_parser_available,
    looks_like_pdf_stub,
    parse_pdf,
    unload_pdf_models,
)


# ---- helpers ----------------------------------------------------------------

def _make_fake_runner(markdown: str, *, images: dict[str, bytes] | None = None):
    """Build a MineruRunner that writes deterministic markdown + fake images.

    images: {filename: bytes}  -- if provided, creates an 'images/' subdir
    next to the markdown file (mimicking MinerU's actual output layout).
    """
    def _runner(input_file: Path, output_dir: Path, timeout: int) -> Path:
        stem = input_file.stem
        target = output_dir / stem / "auto"
        target.mkdir(parents=True, exist_ok=True)
        md = target / f"{stem}.md"
        md.write_text(markdown, encoding="utf-8")
        if images:
            img_dir = target / "images"
            img_dir.mkdir(exist_ok=True)
            for name, data in images.items():
                (img_dir / name).write_bytes(data)
        return md
    return _runner


_GOOD_MD = (
    "# Great Paper\n\n"
    "## Abstract\n\nWe present a study of something important.\n\n"
    "## Methods\n\nWe used these methods to obtain results.\n\n"
    "## Results\n\nThe results show significant improvement over baselines.\n"
)


# ---- parse_pdf: happy path --------------------------------------------------

def test_parse_pdf_returns_result(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    media_dir = tmp_path / "media"
    runner = _make_fake_runner(_GOOD_MD)

    result = parse_pdf(pdf, media_out_dir=media_dir, runner=runner)

    assert isinstance(result, PdfParseResult)
    assert result.markdown == _GOOD_MD
    assert result.n_chars == len(_GOOD_MD)
    assert result.backend == DEFAULT_BACKEND
    assert result.media_subdir_in_md == "images"
    assert result.images == []


def test_parse_pdf_copies_images_to_media_out_dir(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    media_dir = tmp_path / "2603.05238.media"
    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    runner = _make_fake_runner(
        _GOOD_MD,
        images={"fig1.png": fake_png, "fig2.png": fake_png},
    )

    result = parse_pdf(pdf, media_out_dir=media_dir, runner=runner)

    assert len(result.images) == 2
    names = {img["name"] for img in result.images}
    assert names == {"fig1.png", "fig2.png"}
    for img in result.images:
        p = Path(img["abs_path"])
        assert p.exists()
        assert p.parent == media_dir.resolve()


def test_parse_pdf_no_images_no_media_dir_created(tmp_path):
    """When the PDF has no images, media_out_dir must NOT be created."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    media_dir = tmp_path / "media"
    runner = _make_fake_runner(_GOOD_MD)

    result = parse_pdf(pdf, media_out_dir=media_dir, runner=runner)

    assert result.images == []
    assert not media_dir.exists()


def test_parse_pdf_uses_custom_backend(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    runner = _make_fake_runner(_GOOD_MD)
    result = parse_pdf(
        pdf, media_out_dir=tmp_path / "m",
        backend="vlm-transformers", runner=runner,
    )
    assert result.backend == "vlm-transformers"


def test_parse_pdf_missing_file_raises(tmp_path):
    with pytest.raises(PdfParseError, match="not found"):
        parse_pdf(
            tmp_path / "missing.pdf",
            media_out_dir=tmp_path / "media",
            runner=_make_fake_runner(_GOOD_MD),
        )


def test_parse_pdf_runner_failure_raises(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    def _boom(input_file, output_dir, timeout):
        raise PdfParseError("fake runner failure")

    with pytest.raises(PdfParseError, match="fake runner failure"):
        parse_pdf(pdf, media_out_dir=tmp_path / "media", runner=_boom)


# ---- is_pdf_parser_available ------------------------------------------------

def test_is_pdf_parser_available_false_when_mineru_absent(monkeypatch):
    """On hosts without MinerU, must return False without raising."""
    class _BlockMineru:
        def find_spec(self, name, path=None, target=None):
            if name.startswith("mineru"):
                raise ImportError(f"blocked for test: {name}")
            return None

    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("mineru"):
            monkeypatch.delitem(sys.modules, mod_name)

    monkeypatch.setattr("sys.meta_path", [_BlockMineru()] + sys.meta_path)
    assert is_pdf_parser_available() is False


def test_is_pdf_parser_available_true_when_mineru_present(monkeypatch):
    fake_mineru = types.ModuleType("mineru")
    monkeypatch.setitem(sys.modules, "mineru", fake_mineru)
    assert is_pdf_parser_available() is True


# ---- unload_pdf_models ------------------------------------------------------

def test_unload_pdf_models_false_without_mineru(monkeypatch):
    class _BlockMineru:
        def find_spec(self, name, path=None, target=None):
            if name.startswith("mineru"):
                raise ImportError(f"blocked: {name}")
            return None

    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("mineru"):
            monkeypatch.delitem(sys.modules, mod_name)
    monkeypatch.setattr("sys.meta_path", [_BlockMineru()] + sys.meta_path)
    assert unload_pdf_models() is False


def test_unload_pdf_models_clears_singleton_dicts(monkeypatch):
    class _FakeAtom:
        _models = {"layout-en": object(), "ocr-ch": object()}

    class _FakeHybrid:
        _models = {"hybrid-en": object()}

    pkg_mineru = types.ModuleType("mineru")
    pkg_backend = types.ModuleType("mineru.backend")
    pkg_pipeline = types.ModuleType("mineru.backend.pipeline")
    mod_init = types.ModuleType("mineru.backend.pipeline.model_init")
    mod_init.AtomModelSingleton = _FakeAtom
    mod_init.HybridModelSingleton = _FakeHybrid

    monkeypatch.setitem(sys.modules, "mineru", pkg_mineru)
    monkeypatch.setitem(sys.modules, "mineru.backend", pkg_backend)
    monkeypatch.setitem(sys.modules, "mineru.backend.pipeline", pkg_pipeline)
    monkeypatch.setitem(
        sys.modules, "mineru.backend.pipeline.model_init", mod_init
    )

    assert unload_pdf_models() is True
    assert _FakeAtom._models == {}
    assert _FakeHybrid._models == {}


def test_unload_pdf_models_idempotent_when_empty(monkeypatch):
    class _FakeAtom:
        _models: dict = {}

    class _FakeHybrid:
        _models: dict = {}

    pkg_mineru = types.ModuleType("mineru")
    pkg_backend = types.ModuleType("mineru.backend")
    pkg_pipeline = types.ModuleType("mineru.backend.pipeline")
    mod_init = types.ModuleType("mineru.backend.pipeline.model_init")
    mod_init.AtomModelSingleton = _FakeAtom
    mod_init.HybridModelSingleton = _FakeHybrid

    monkeypatch.setitem(sys.modules, "mineru", pkg_mineru)
    monkeypatch.setitem(sys.modules, "mineru.backend", pkg_backend)
    monkeypatch.setitem(sys.modules, "mineru.backend.pipeline", pkg_pipeline)
    monkeypatch.setitem(
        sys.modules, "mineru.backend.pipeline.model_init", mod_init
    )
    assert unload_pdf_models() is False


# ---- looks_like_pdf_stub ----------------------------------------------------

def test_stub_empty_string():
    assert looks_like_pdf_stub("") is True


def test_stub_very_short():
    assert looks_like_pdf_stub("# Title\n") is True


def test_stub_whitespace_heavy():
    # Almost entirely whitespace: ratio < 0.10
    markdown = "\n\n\n\n\n\n\n\n\n\n\n\n\n" + "x" * 5
    assert looks_like_pdf_stub(markdown) is True


def test_stub_normal_content():
    assert looks_like_pdf_stub(_GOOD_MD) is False


def test_stub_minimal_passing_content():
    # Exactly at/above threshold: 300 chars, all non-whitespace
    md = "x" * 300
    assert looks_like_pdf_stub(md) is False


# ---- no_host_project_imports ------------------------------------------------

def test_pdf_module_does_not_import_host_projects():
    """Importing corpus_core.pdf must not pull in arxiv_radar_mcp or
    lab_corpus_mcp (Phase 3 architectural invariant)."""
    import corpus_core.pdf  # noqa: F401

    leaked = [
        name for name in sys.modules
        if name.startswith("arxiv_radar_mcp")
        or name.startswith("lab_corpus_mcp")
    ]
    assert leaked == [], (
        f"corpus_core.pdf leaked host-project imports: {leaked}"
    )
