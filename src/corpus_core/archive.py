"""Paper download archive — zip markdown + figures + meta into one folder.

The binary side-channel both servers expose as `GET /download?id=<id>`:
MCP's JSON-RPC can't carry a multi-MB figure bundle, so the HTTP backend
ships a zip on the same port as `/mcp` (the mirror of lab-corpus's
`POST /upload`).

Both downstream servers store the parsed markdown under the same
`sources/<id>.md` (+ `<id>.meta.json`) convention but keep figures in
different places and reference them differently in the markdown:

  * arxiv-radar — images in `sources/<id>.media/`, md refs `![](<id>.media/x.png)`
  * lab-corpus  — images in `figures/<id>/`,     md refs `![](images/x.jpg)` (MinerU)

So the only per-server knobs are *where the images physically live*
(`media_dir`) and *what subdir name the md refs expect inside the
archive* (`media_arcname`). Placing the images under `media_arcname` next
to the markdown makes the relative refs resolve after unzip with no
rewriting. Everything else (the `<id>/` root folder, meta, the HTTP
handler, the traversal guard) is shared here.
"""
from __future__ import annotations

import asyncio
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Reject path-traversal / absolute / NUL / Windows-drive-relative ids.
# Permissive on the rest so both arxiv ids (`2603.05238`,
# old `cond-mat/0211034`) and lab-corpus hash/doi-derived ids pass.
# `:` is blocked because `C:` style paths bypass directory guards on Windows.
_UNSAFE_ID_RE = re.compile(r"\.\.|[\\\x00:]")


def is_safe_paper_id(paper_id: str) -> bool:
    """True iff `paper_id` is safe to interpolate into archive/file paths."""
    return bool(paper_id) and len(paper_id) <= 120 \
        and not paper_id.startswith("/") \
        and _UNSAFE_ID_RE.search(paper_id) is None


@dataclass
class PaperFiles:
    """Where a server keeps the pieces of one parsed paper.

    `media_arcname` is the subdir name the markdown's image refs expect
    (e.g. `images` or `<id>.media`); it defaults to the source dir's own
    name. `media_dir` may be None / absent (text-only papers).
    """
    markdown_path: Path
    media_dir: Path | None = None
    media_arcname: str | None = None
    meta_path: Path | None = None


def _is_within(path: Path, root: Path) -> bool:
    """Return True iff `path.resolve()` is under `root.resolve()`.

    Compatible with Python 3.9+ (is_relative_to was added in 3.9; for 3.8
    compatibility we use a try/except relative_to fallback).
    """
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def build_paper_archive(
    paper_id: str,
    files: PaperFiles,
    *,
    root_dir: Path | None = None,
) -> bytes | None:
    """Zip a parsed paper into a single `<paper_id>/` folder.

    Layout (so unzipped papers never collide, and md refs resolve as-is):

        <id>/<id>.md
        <id>/<id>.meta.json          (if present)
        <id>/<media_arcname>/<name>  (each figure, if any)

    Returns the zip bytes, or None when the id is unsafe or the markdown
    is missing (paper not fetched/ingested yet).

    `root_dir`: when provided, all paths in `files` are validated to be
    inside `root_dir` after resolution. Any path that escapes the root
    is silently rejected (returns None), preventing symlink/traversal
    attacks even when `is_safe_paper_id` passes.
    """
    if not is_safe_paper_id(paper_id):
        return None
    if root_dir is not None and not _is_within(files.markdown_path, root_dir):
        return None
    if not files.markdown_path.exists():
        return None

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{paper_id}/{paper_id}.md",
                    files.markdown_path.read_text(encoding="utf-8"))
        if files.meta_path and files.meta_path.exists():
            zf.writestr(f"{paper_id}/{paper_id}.meta.json",
                        files.meta_path.read_text(encoding="utf-8"))
        if files.media_dir and files.media_dir.is_dir():
            sub = files.media_arcname or files.media_dir.name
            for img in sorted(files.media_dir.iterdir()):
                if img.is_file():
                    zf.writestr(f"{paper_id}/{sub}/{img.name}", img.read_bytes())
    return buf.getvalue()


def make_download_handler(resolve: Callable[[str], "PaperFiles | None"]):
    """Build a Starlette `GET /download?id=<id>` endpoint.

    `resolve(paper_id)` maps an id to its `PaperFiles` (pure path
    construction — return None to reject an id outright; a None markdown
    on disk is handled here as 404). The zip is built off the event loop.

    Responses: 200 application/zip · 400 missing id · 404 unknown/not ready.
    """
    async def download(request):
        from starlette.responses import JSONResponse, Response  # noqa: PLC0415

        paper_id = (request.query_params.get("id") or "").strip()
        if not paper_id:
            return JSONResponse({"error": "missing query param ?id="},
                                status_code=400)
        files = resolve(paper_id)
        data = None
        if files is not None:
            data = await asyncio.to_thread(build_paper_archive, paper_id, files)
        if data is None:
            return JSONResponse(
                {"error": f"'{paper_id}' not available — fetch/ingest it first, "
                          "or check the id"},
                status_code=404,
            )
        safe = paper_id.replace("/", "_")
        return Response(
            content=data, media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{safe}.zip"'},
        )

    return download
