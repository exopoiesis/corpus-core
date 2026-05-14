"""Tests for ``corpus_core.http_fetch``.

Strategy: drive ``fetch_url`` / ``request_with_retry`` against an
``httpx.MockTransport`` so the suite stays hermetic (no live HTTP).
``Throttle`` is exercised with a fake monotonic clock + sleep stub.
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from corpus_core import http_fetch
from corpus_core.http_fetch import (
    ARXIV_RATE_LIMIT_S,
    FetchResult,
    Throttle,
    fetch_arxiv_pdf,
    fetch_url,
    get_arxiv_throttle,
    request_with_retry,
)


# ---------------------------------------------------------------------------
# Throttle
# ---------------------------------------------------------------------------


def test_throttle_first_call_no_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """First call to wait() must not sleep — the budget hasn't been spent."""
    sleeps: list[float] = []
    clock = [100.0]
    monkeypatch.setattr(http_fetch.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(http_fetch.time, "monotonic", lambda: clock[0])

    t = Throttle(min_interval_s=3.0)
    t.wait()
    assert sleeps == []


def test_throttle_back_to_back_call_sleeps_remainder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two calls within the window must sleep for the remaining gap."""
    sleeps: list[float] = []
    clock = [100.0]
    monkeypatch.setattr(http_fetch.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(http_fetch.time, "monotonic", lambda: clock[0])

    t = Throttle(min_interval_s=3.0)
    t.wait()                       # marks last_request_at = 100.0
    clock[0] = 101.0               # only 1 sec elapsed
    t.wait()                       # should sleep ~2 sec
    assert sleeps and abs(sleeps[0] - 2.0) < 1e-9


def test_throttle_after_full_window_no_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the full interval has elapsed, no extra sleep."""
    sleeps: list[float] = []
    clock = [100.0]
    monkeypatch.setattr(http_fetch.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(http_fetch.time, "monotonic", lambda: clock[0])

    t = Throttle(min_interval_s=3.0)
    t.wait()
    clock[0] = 200.0               # plenty later
    t.wait()
    assert sleeps == []


def test_arxiv_throttle_singleton_constants() -> None:
    a = get_arxiv_throttle()
    b = get_arxiv_throttle()
    assert a is b
    # Documented invariant — exposed for downstream consumers (lab-corpus
    # ingest) to pin-test their own behaviour against.
    assert ARXIV_RATE_LIMIT_S == 3.0


# ---------------------------------------------------------------------------
# request_with_retry
# ---------------------------------------------------------------------------


def _mock_client(handler) -> httpx.Client:
    """Build an httpx.Client backed by MockTransport."""
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_request_with_retry_success_first_try() -> None:
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        return httpx.Response(200, content=b"ok")

    client = _mock_client(handler)
    r = request_with_retry(client, "https://example.com/x", throttle=None)
    assert r.status_code == 200
    assert len(calls) == 1


def test_request_with_retry_404_no_retry() -> None:
    """Non-throttle codes (e.g., 404) must NOT retry."""
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        return httpx.Response(404)

    client = _mock_client(handler)
    r = request_with_retry(
        client, "https://example.com/x", throttle=None, max_attempts=3,
    )
    assert r.status_code == 404
    assert len(calls) == 1


def test_request_with_retry_429_retries_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(http_fetch.time, "sleep", lambda s: None)
    states = iter([429, 429, 200])

    def handler(req: httpx.Request) -> httpx.Response:
        code = next(states)
        return httpx.Response(code, content=b"ok" if code == 200 else b"")

    client = _mock_client(handler)
    r = request_with_retry(
        client, "https://example.com/x",
        throttle=None, max_attempts=3,
    )
    assert r.status_code == 200


def test_request_with_retry_503_max_attempts_returns_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(http_fetch.time, "sleep", lambda s: None)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = _mock_client(handler)
    r = request_with_retry(
        client, "https://example.com/x",
        throttle=None, max_attempts=2,
    )
    assert r.status_code == 503


def test_request_with_retry_honours_retry_after_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(http_fetch.time, "sleep", lambda s: sleeps.append(s))
    states = iter([429, 200])

    def handler(req: httpx.Request) -> httpx.Response:
        code = next(states)
        if code == 429:
            return httpx.Response(429, headers={"Retry-After": "7.5"})
        return httpx.Response(200, content=b"ok")

    client = _mock_client(handler)
    r = request_with_retry(
        client, "https://example.com/x",
        throttle=None, max_attempts=3,
    )
    assert r.status_code == 200
    assert sleeps and abs(sleeps[0] - 7.5) < 1e-9


def test_request_with_retry_invokes_throttle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(http_fetch.time, "sleep", lambda s: None)
    waits = [0]

    class CountingThrottle:
        def wait(self) -> None:
            waits[0] += 1

    states = iter([429, 200])

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(next(states), content=b"ok")

    client = _mock_client(handler)
    r = request_with_retry(
        client, "https://example.com/x",
        throttle=CountingThrottle(), max_attempts=3,
    )
    assert r.status_code == 200
    # Throttle invoked once per attempt (so 2 attempts → 2 waits).
    assert waits[0] == 2


# ---------------------------------------------------------------------------
# fetch_url
# ---------------------------------------------------------------------------


def test_fetch_url_writes_body_atomically(tmp_path: Path) -> None:
    body = b"%PDF-1.7 fake"

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = _mock_client(handler)
    dest = tmp_path / "subdir" / "out.pdf"
    res = fetch_url(
        "https://example.com/x.pdf", dest,
        throttle=None, client=client,
    )
    assert isinstance(res, FetchResult)
    assert res.ok
    assert res.status == 200
    assert res.dest_path == dest
    assert res.n_bytes == len(body)
    assert dest.read_bytes() == body
    # No leftover .tmp file from atomic rename.
    assert not (dest.parent / "out.pdf.tmp").exists()


def test_fetch_url_404_returns_failure_no_file(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = _mock_client(handler)
    dest = tmp_path / "x.pdf"
    res = fetch_url(
        "https://example.com/missing", dest,
        throttle=None, client=client,
    )
    assert not res.ok
    assert res.status == 404
    assert res.dest_path is None
    assert "404" in (res.error or "")
    assert not dest.exists()


def test_fetch_url_empty_body_is_failure(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    client = _mock_client(handler)
    dest = tmp_path / "x.pdf"
    res = fetch_url(
        "https://example.com/x.pdf", dest,
        throttle=None, client=client,
    )
    assert not res.ok
    assert res.status == 200
    assert res.error == "empty body"
    assert not dest.exists()


def test_fetch_url_transport_error_returns_failure(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    client = _mock_client(handler)
    dest = tmp_path / "x.pdf"
    res = fetch_url(
        "https://example.com/x.pdf", dest,
        throttle=None, client=client,
    )
    assert not res.ok
    assert res.status is None
    assert "ConnectError" in (res.error or "")


def test_fetch_url_overwrite_false_keeps_existing(tmp_path: Path) -> None:
    dest = tmp_path / "x.pdf"
    dest.write_bytes(b"existing")

    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, content=b"new")

    client = _mock_client(handler)
    res = fetch_url(
        "https://example.com/x.pdf", dest,
        throttle=None, client=client, overwrite=False,
    )
    assert res.ok
    assert dest.read_bytes() == b"existing"
    assert calls == []  # no HTTP call when cached file exists


def test_fetch_url_overwrite_true_replaces(tmp_path: Path) -> None:
    dest = tmp_path / "x.pdf"
    dest.write_bytes(b"existing")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"new")

    client = _mock_client(handler)
    res = fetch_url(
        "https://example.com/x.pdf", dest,
        throttle=None, client=client, overwrite=True,
    )
    assert res.ok
    assert dest.read_bytes() == b"new"


# ---------------------------------------------------------------------------
# fetch_arxiv_pdf
# ---------------------------------------------------------------------------


def test_fetch_arxiv_pdf_uses_correct_url_and_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_fetch_url(url: str, dest, **kw) -> FetchResult:
        captured["url"] = url
        captured["dest"] = Path(dest)
        captured["throttle"] = kw.get("throttle")
        return FetchResult(
            url=url, dest_path=Path(dest),
            ok=True, status=200, n_bytes=42, error=None,
        )

    monkeypatch.setattr(http_fetch, "fetch_url", fake_fetch_url)
    res = fetch_arxiv_pdf("2410.04059", tmp_path)
    assert res.ok
    assert captured["url"] == "https://arxiv.org/pdf/2410.04059"
    assert captured["dest"] == tmp_path / "2410.04059.pdf"
    # Must have wired in the singleton arxiv throttle.
    assert captured["throttle"] is get_arxiv_throttle()
