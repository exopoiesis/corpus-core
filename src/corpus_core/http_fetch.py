"""HTTP-fetch primitives: throttled GET with retry + atomic file write.

Used by both `arxiv-radar-mcp` (HTML/LaTeX cascade in `fulltext.py`) and
`lab-corpus-mcp` (`ingest_arxiv_pdf` / `ingest_url` tools). Lives in
`corpus-core` so the **same module-global arXiv throttle is shared
across both servers** when they run in one process (combined image) —
otherwise two unaware fetchers would double-spam arxiv.org and trip the
1 req / 3 sec ToS.

Public surface:
  * ``Throttle``                       — thread-safe min-interval gate
  * ``get_arxiv_throttle()``           — singleton for arxiv.org GETs
  * ``request_with_retry()``           — httpx.Client wrapper, 429/503
                                          backoff with Retry-After
  * ``fetch_url(url, dest_path, ...)`` — download to file, atomic rename
  * ``fetch_arxiv_pdf(arxiv_id, ...)`` — convenience wrapper for the
                                          arxiv.org/pdf/<id> endpoint
  * ``FetchResult``                    — dataclass returned by fetchers
  * ``DEFAULT_USER_AGENT``,
    ``ARXIV_RATE_LIMIT_S``             — module-level constants
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

LOG = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "corpus-core/0.1 (+https://github.com/exopoiesis/corpus-core)"
)

# arXiv ToS / robots.txt: no more than 1 outbound request / 3 seconds.
ARXIV_RATE_LIMIT_S = 3.0


class Throttle:
    """Thread-safe rate limiter — block until next request respects an interval.

    One instance per source domain. Within a process, every caller that
    shares the instance shares the budget; this is what lets the
    combined arxiv-radar+lab-corpus image enforce arXiv's 1 req / 3 sec
    across both servers' fetchers.
    """

    def __init__(self, min_interval_s: float):
        self._min_interval = float(min_interval_s)
        self._last_request_at = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until the next outbound request respects the rate limit."""
        with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = time.monotonic()


# Module-global arxiv.org throttle. Both arxiv-radar's HTML/LaTeX
# fetcher and lab-corpus's `ingest_arxiv_pdf` go through this; in the
# combined image they share one process and one Throttle instance, so
# arXiv sees one stream of requests at the contracted cadence.
_arxiv_throttle = Throttle(ARXIV_RATE_LIMIT_S)


def get_arxiv_throttle() -> Throttle:
    """Return the singleton arxiv.org throttle.

    Use this whenever you GET arxiv.org/{html,e-print,pdf,abs}/<id>.
    """
    return _arxiv_throttle


@dataclass
class FetchResult:
    """Outcome of a `fetch_url` call.

    ``ok=True`` iff a 2xx response was received with a non-empty body
    AND the body was atomically written to ``dest_path``. On failure,
    ``dest_path`` was not created/overwritten and ``error`` carries a
    human-readable reason.
    """
    url: str
    dest_path: Path | None
    ok: bool
    status: int | None
    n_bytes: int
    error: str | None


def request_with_retry(
    client: httpx.Client,
    url: str,
    *,
    throttle: Throttle | None,
    method: str = "GET",
    max_attempts: int = 3,
    backoff_seed_s: float = 3.0,
) -> httpx.Response:
    """HTTP request with optional throttle + exponential backoff on 429/503.

    Honours ``Retry-After`` if present (parsed as float seconds). Retries
    only on transient throttle codes; 404/410/5xx-other fall through
    after the first attempt.

    ``throttle=None`` skips rate-limiting (useful in tests and for hosts
    with no rate-limit policy).
    """
    backoff = backoff_seed_s
    r: httpx.Response | None = None
    for attempt in range(1, max_attempts + 1):
        if throttle is not None:
            throttle.wait()
        r = client.request(method, url)
        if r.status_code not in (429, 503):
            return r
        if attempt == max_attempts:
            return r
        retry_after = r.headers.get("Retry-After")
        try:
            wait = float(retry_after) if retry_after else backoff
        except (TypeError, ValueError):
            wait = backoff
        LOG.warning(
            f"http {r.status_code} on {url}; retry in {wait:.1f}s "
            f"(attempt {attempt}/{max_attempts})"
        )
        time.sleep(wait)
        backoff *= 2
    assert r is not None  # loop body always assigns r
    return r


def fetch_url(
    url: str,
    dest_path: Path | str,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    throttle: Throttle | None = None,
    timeout_s: float = 30.0,
    connect_timeout_s: float = 10.0,
    max_attempts: int = 3,
    overwrite: bool = True,
    client: httpx.Client | None = None,
) -> FetchResult:
    """Download ``url`` to ``dest_path`` with throttle + retry.

    Body is written atomically via ``<dest_path>.tmp`` rename. Parent
    dirs are auto-created. On any failure (transport error, non-2xx,
    empty body) ``dest_path`` is not touched.

    Args:
        url: Absolute URL to GET.
        dest_path: Where to save the body.
        user_agent: User-Agent header.
        throttle: Optional rate limiter. ``None`` skips rate-limiting.
        timeout_s, connect_timeout_s: per-request httpx timeouts.
        max_attempts: how many tries on 429/503 before giving up.
        overwrite: if False and ``dest_path`` already exists with size
                   > 0, return ``ok=True`` without re-downloading
                   (idempotent caching mode).
        client: optional shared ``httpx.Client`` for batch fetches with
                connection reuse. Caller manages lifecycle. When
                ``None`` (the default), a temporary client is opened
                with ``user_agent``, ``timeout_s``, ``connect_timeout_s``.
                A caller-supplied client is used as-is — its own
                ``headers`` and ``timeout`` win over the args here.

    Returns:
        ``FetchResult``. ``ok=True`` ↔ a 2xx with non-empty body was
        successfully persisted to ``dest_path``.
    """
    dest = Path(dest_path)

    if not overwrite and dest.exists() and dest.stat().st_size > 0:
        return FetchResult(
            url=url, dest_path=dest, ok=True, status=None,
            n_bytes=dest.stat().st_size, error=None,
        )

    own_client = client is None
    if own_client:
        client = httpx.Client(
            timeout=httpx.Timeout(timeout_s, connect=connect_timeout_s),
            headers={"User-Agent": user_agent},
            follow_redirects=True,
        )
    try:
        try:
            r = request_with_retry(
                client, url, throttle=throttle, max_attempts=max_attempts,
            )
        except httpx.HTTPError as e:
            return FetchResult(
                url=url, dest_path=None, ok=False, status=None,
                n_bytes=0,
                error=f"transport error: {type(e).__name__}: {e}",
            )

        if r.status_code != 200:
            return FetchResult(
                url=url, dest_path=None, ok=False, status=r.status_code,
                n_bytes=0, error=f"http {r.status_code}",
            )

        body = r.content
        if not body:
            return FetchResult(
                url=url, dest_path=None, ok=False, status=200,
                n_bytes=0, error="empty body",
            )

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(body)
        tmp.replace(dest)
        return FetchResult(
            url=url, dest_path=dest, ok=True, status=200,
            n_bytes=len(body), error=None,
        )
    finally:
        if own_client:
            client.close()


def fetch_arxiv_pdf(
    arxiv_id: str,
    dest_dir: Path | str,
    *,
    overwrite: bool = True,
    client: httpx.Client | None = None,
    timeout_s: float = 60.0,
    connect_timeout_s: float = 10.0,
    max_attempts: int = 3,
) -> FetchResult:
    """Fetch ``https://arxiv.org/pdf/<arxiv_id>`` into ``<dest_dir>/<arxiv_id>.pdf``.

    Convenience over ``fetch_url`` with the arxiv.org throttle wired in
    automatically and a longer default timeout (PDFs are larger than
    HTML responses).

    The destination filename is exactly ``<arxiv_id>.pdf`` so that
    downstream MinerU + lab-corpus paper-id derivation produce
    ``paper_id == arxiv_id`` without extra hints.
    """
    dest = Path(dest_dir) / f"{arxiv_id}.pdf"
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    return fetch_url(
        url, dest,
        throttle=get_arxiv_throttle(),
        timeout_s=timeout_s,
        connect_timeout_s=connect_timeout_s,
        max_attempts=max_attempts,
        overwrite=overwrite,
        client=client,
    )


__all__ = [
    "ARXIV_RATE_LIMIT_S",
    "DEFAULT_USER_AGENT",
    "FetchResult",
    "Throttle",
    "fetch_arxiv_pdf",
    "fetch_url",
    "get_arxiv_throttle",
    "request_with_retry",
]
