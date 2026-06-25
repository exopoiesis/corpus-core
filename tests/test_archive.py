"""Tests for corpus_core.archive — the shared paper-download zip builder."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

from corpus_core.archive import (PaperFiles, build_paper_archive,
                                 is_safe_paper_id)

_PNG = b"\x89PNG\r\n\x1a\n" + b"img" * 8


def _write(p: Path, data) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    return p


def test_is_safe_paper_id():
    assert is_safe_paper_id("2603.05238")
    assert is_safe_paper_id("cond-mat_0211034")
    assert is_safe_paper_id("sha256deadbeef")
    assert not is_safe_paper_id("")
    assert not is_safe_paper_id("../../etc/passwd")
    assert not is_safe_paper_id("..")
    assert not is_safe_paper_id("/abs/path")
    assert not is_safe_paper_id("back\\slash")
    assert not is_safe_paper_id("x" * 200)


def test_build_archive_with_media_and_meta(tmp_path: Path):
    md = _write(tmp_path / "sources" / "P1.md",
                "# P1\n\n![](media/x1.png)\n")
    _write(tmp_path / "sources" / "P1.meta.json", '{"source":"html"}')
    media = tmp_path / "media"
    _write(media / "x1.png", _PNG)

    data = build_paper_archive("P1", PaperFiles(
        markdown_path=md, media_dir=media, media_arcname="media",
        meta_path=tmp_path / "sources" / "P1.meta.json",
    ))
    assert data is not None
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        assert names == {"P1/P1.md", "P1/P1.meta.json", "P1/media/x1.png"}
        assert zf.read("P1/media/x1.png") == _PNG
        # ref resolves: md says media/x1.png, image is at P1/media/x1.png
        assert "![](media/x1.png)" in zf.read("P1/P1.md").decode()


def test_build_archive_media_arcname_renames_subdir(tmp_path: Path):
    """The in-archive media subdir follows media_arcname, NOT the disk dir
    name — this is how lab-corpus maps figures/<id>/ onto MinerU's
    `images/` refs."""
    md = _write(tmp_path / "sources" / "P2.md", "# P2\n\n![](images/f.png)\n")
    media = tmp_path / "figures" / "P2"   # on-disk name differs from ref
    _write(media / "f.png", _PNG)

    data = build_paper_archive("P2", PaperFiles(
        markdown_path=md, media_dir=media, media_arcname="images",
    ))
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert "P2/images/f.png" in zf.namelist()
        assert "P2/figures/f.png" not in zf.namelist()


def test_build_archive_md_only(tmp_path: Path):
    md = _write(tmp_path / "sources" / "P3.md", "# P3\n\nNo figures.")
    data = build_paper_archive("P3", PaperFiles(markdown_path=md))
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert zf.namelist() == ["P3/P3.md"]


def test_build_archive_none_when_md_missing(tmp_path: Path):
    assert build_paper_archive(
        "P4", PaperFiles(markdown_path=tmp_path / "nope.md")) is None


def test_build_archive_none_on_unsafe_id(tmp_path: Path):
    md = _write(tmp_path / "x.md", "# x")
    assert build_paper_archive("../escape", PaperFiles(markdown_path=md)) is None


def test_build_archive_skips_missing_media_dir(tmp_path: Path):
    md = _write(tmp_path / "sources" / "P5.md", "# P5")
    data = build_paper_archive("P5", PaperFiles(
        markdown_path=md, media_dir=tmp_path / "absent", media_arcname="images",
    ))
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert zf.namelist() == ["P5/P5.md"]


def test_download_handler_http(tmp_path: Path):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from corpus_core.archive import make_download_handler

    _write(tmp_path / "sources" / "P9.md", "# P9\n\n![](images/a.png)\n")
    _write(tmp_path / "figures" / "P9" / "a.png", _PNG)

    def resolve(pid):
        return PaperFiles(
            markdown_path=tmp_path / "sources" / f"{pid}.md",
            media_dir=tmp_path / "figures" / pid,
            media_arcname="images",
        )

    app = Starlette(routes=[
        Route("/download", endpoint=make_download_handler(resolve), methods=["GET"]),
    ])
    client = TestClient(app)

    # happy path → 200 zip
    r = client.get("/download?id=P9")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        assert "P9/images/a.png" in zf.namelist()

    # missing id → 400 ; unknown id → 404
    assert client.get("/download").status_code == 400
    assert client.get("/download?id=ZZZ").status_code == 404
