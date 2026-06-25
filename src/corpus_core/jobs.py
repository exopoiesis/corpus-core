"""Background job registry for fetch_papers and reindex.

Both operations take long enough (seconds to minutes) that blocking the
MCP tool call breaks the conversation. The flow is:

  client            registry          worker thread
   │                  │                    │
   │── submit ─────▶  │                    │
   │  ◀── job_id ────│                    │
   │                  │── start ──────────▶│
   │── status ─────▶ │                    │ (writes progress)
   │  ◀── progress ──│                    │
   │                  │   ◀── result ─────│
   │── status ─────▶ │                    │
   │  ◀── done ──────│                    │

Persistence:
  <cache_dir>/jobs/<job_id>.json  per job, atomic write per status update.
  Survives server restarts. Anything left in `running` after restart is
  marked `orphaned` (the server didn't survive its own job).

Lockfile:
  <cache_dir>/fulltext/.reindex.lock — only one reindex at a time. Second
  attempt is rejected with a clear error.

Cleanup:
  Jobs older than 7 days deleted on registry init.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

LOG = logging.getLogger(__name__)


JobState = str  # "pending" | "running" | "done" | "failed" | "orphaned"


@dataclass
class Job:
    job_id: str
    kind: str                      # "fetch_papers" | "reindex"
    state: JobState = "pending"
    progress: float = 0.0          # 0.0 .. 1.0
    n_total: int = 0
    n_done: int = 0
    started_at: str | None = None  # ISO 8601 UTC
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobError(Exception):
    """Raised by job functions to signal a clean failure (state=failed)."""


class JobRegistry:
    """Persistent, thread-safe job tracker.

    Uses a single ThreadPoolExecutor (max_workers=2 by default, enough for
    one fetch + one reindex in parallel). The reindex lockfile serializes
    multiple reindex attempts.
    """

    def __init__(self, cache_dir: Path, max_workers: int = 2,
                 retain_days: int = 7) -> None:
        self.cache_dir = cache_dir
        self.jobs_dir = cache_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._futures: dict[str, Future] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="arxiv-radar-job",
        )
        self._reindex_lock_path = cache_dir / "fulltext" / ".reindex.lock"
        self._retain_delta = timedelta(days=retain_days)

        self._reload_persisted()

    # ----- lifecycle -------------------------------------------------------

    def submit(
        self,
        kind: str,
        fn: Callable[["JobHandle"], dict[str, Any]],
        *,
        args: dict[str, Any] | None = None,
        n_total: int = 0,
    ) -> str:
        """Register a new job and start it in the worker pool.

        `fn` must accept a JobHandle and return a result dict. Use the handle
        to update progress/n_done; the registry persists each update.
        """
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id=job_id, kind=kind, args=args or {}, n_total=n_total)

        with self._lock:
            self._jobs[job_id] = job
            self._persist(job)

        future = self._executor.submit(self._run, job_id, fn)
        self._futures[job_id] = future
        return job_id

    def _run(self, job_id: str, fn: Callable[["JobHandle"], dict[str, Any]]) -> None:
        handle = JobHandle(self, job_id)
        with self._lock:
            job = self._jobs[job_id]
            job.state = "running"
            job.started_at = _utcnow_iso()
            self._persist(job)

        try:
            result = fn(handle)
            with self._lock:
                job = self._jobs[job_id]
                job.state = "done"
                job.progress = 1.0
                job.finished_at = _utcnow_iso()
                job.result = result
                self._persist(job)
        except JobError as e:
            with self._lock:
                job = self._jobs[job_id]
                job.state = "failed"
                job.finished_at = _utcnow_iso()
                job.error = str(e)
                self._persist(job)
            LOG.warning(f"job {job_id} ({job.kind}) failed cleanly: {e}")
        except Exception as e:  # noqa: BLE001 — capture any unexpected crash
            with self._lock:
                job = self._jobs[job_id]
                job.state = "failed"
                job.finished_at = _utcnow_iso()
                job.error = f"unexpected error: {type(e).__name__}: {e}"
                self._persist(job)
            LOG.exception(f"job {job_id} ({job.kind}) crashed")

    # ----- query API -------------------------------------------------------

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            # U1 fix: when in-memory still says pending/running, the persisted
            # jobs/<id>.json file is the source of truth — it gets written on
            # every state change inside _run() under the same lock that
            # mutates the dict, but the dict can lag behind disk in edge
            # cases observed during 2026-05-08 dogfood (job_status returned
            # 'running 0%' for 17+ min while job_list showed 'done'). For
            # terminal states (done/failed/orphaned) the dict was the last
            # writer so it stays authoritative — avoids the JSON-read cost
            # on the common-case status poll.
            if job.state in ("pending", "running"):
                disk = self._read_persisted(job_id)
                if disk is not None and disk.get("state") in (
                    "done", "failed", "orphaned"
                ):
                    return disk
            return job.to_dict()

    def _read_persisted(self, job_id: str) -> dict[str, Any] | None:
        """Read jobs/<job_id>.json off disk; None on missing/corrupt."""
        path = self.jobs_dir / f"{job_id}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None

    def list_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda j: j.started_at or "",
                reverse=True,
            )
            return [j.to_dict() for j in jobs[:limit]]

    # ----- reindex lockfile ------------------------------------------------

    def acquire_reindex_lock(self) -> bool:
        """Atomic create -- returns True if we got the lock, False if held.

        The lockfile carries real process identity (os.getpid() + hostname +
        start_time) so _reload_persisted can decide whether the previous owner
        is still alive before unconditionally deleting it.
        """
        self._reindex_lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = self._reindex_lock_path.open("x")
            fd.write(
                f"pid={os.getpid()}\n"
                f"hostname={socket.gethostname()}\n"
                f"start_time={_utcnow_iso()}\n"
            )
            fd.close()
            return True
        except FileExistsError:
            return False

    def release_reindex_lock(self) -> None:
        try:
            self._reindex_lock_path.unlink(missing_ok=True)
        except OSError as e:
            LOG.warning(f"could not release reindex lock: {e}")

    # ----- internals -------------------------------------------------------

    def _persist(self, job: Job) -> None:
        path = self.jobs_dir / f"{job.job_id}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(job.to_dict(), indent=1), encoding="utf-8")
        tmp.replace(path)

    def _reload_persisted(self) -> None:
        """Rehydrate jobs from disk. Mark orphaned anything stuck in running."""
        cutoff = datetime.now(timezone.utc) - self._retain_delta
        for path in self.jobs_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue

            try:
                started = datetime.fromisoformat(data.get("started_at") or "")
            except ValueError:
                started = None

            if started and started < cutoff and data.get("state") in ("done", "failed", "orphaned"):
                # Old terminal job — clean up.
                try:
                    path.unlink()
                except OSError:
                    pass
                continue

            if data.get("state") in ("pending", "running"):
                data["state"] = "orphaned"
                data["error"] = "server restarted while job was running"
                data["finished_at"] = _utcnow_iso()

            try:
                job = Job(**{k: v for k, v in data.items()
                            if k in Job.__dataclass_fields__})
                self._jobs[job.job_id] = job
                if data.get("state") == "orphaned":
                    self._persist(job)  # rewrite with the new state
            except TypeError:
                continue

        # On restart, try to clean up a stale reindex lockfile.
        # Only remove it when we can confirm the previous owner is dead.
        self._maybe_clear_stale_lock()

    def _maybe_clear_stale_lock(self) -> None:
        """Remove the reindex lockfile only when the previous owner is provably dead.

        Decision matrix:
        - hostname matches AND pid is dead  -> stale, remove (orphan recovery).
        - hostname matches AND pid is alive -> real owner, do NOT remove.
        - hostname does NOT match           -> foreign host, cannot check pid;
          conservatively leave the lock and log a WARNING.
        - lock file missing or unparseable  -> nothing to do.
        """
        if not self._reindex_lock_path.exists():
            return

        info = _parse_lock_file(self._reindex_lock_path)
        if info is None:
            # Unparseable old-format lock (e.g. "stale" or "pid=<thread_id>").
            # Treat as stale and remove so the server can acquire the lock.
            LOG.info("reindex lock: unparseable format, removing as stale")
            try:
                self._reindex_lock_path.unlink(missing_ok=True)
            except OSError as e:
                LOG.warning(f"reindex lock: could not remove unparseable lock: {e}")
            return

        lock_host = info.get("hostname", "")
        my_host = socket.gethostname()

        if lock_host != my_host:
            # Foreign-host lock -- we cannot check the pid.
            LOG.warning(
                f"reindex lock owned by foreign host {lock_host!r} "
                f"(this host is {my_host!r}); leaving lock in place. "
                "Remove manually if the other host is gone."
            )
            return

        # Same host -- check if the process is still alive.
        try:
            pid = int(info["pid"])
        except (KeyError, ValueError):
            LOG.info("reindex lock: could not parse pid, removing as stale")
            try:
                self._reindex_lock_path.unlink(missing_ok=True)
            except OSError as e:
                LOG.warning(f"reindex lock: could not remove: {e}")
            return

        if _pid_is_alive(pid):
            LOG.warning(
                f"reindex lock held by pid {pid} on this host -- "
                "process appears alive; leaving lock in place."
            )
        else:
            LOG.info(
                f"reindex lock: previous owner pid {pid} is dead, removing stale lock"
            )
            try:
                self._reindex_lock_path.unlink(missing_ok=True)
            except OSError as e:
                LOG.warning(f"reindex lock: could not remove stale lock: {e}")

    def shutdown(self) -> None:
        """Wait for in-flight jobs and stop the executor. Used in tests."""
        self._executor.shutdown(wait=True, cancel_futures=False)


class JobHandle:
    """Passed to the worker function so it can report progress."""

    def __init__(self, registry: JobRegistry, job_id: str) -> None:
        self._registry = registry
        self._job_id = job_id

    def update(self, *, n_done: int | None = None,
               progress: float | None = None,
               n_total: int | None = None) -> None:
        with self._registry._lock:  # noqa: SLF001 — internal API
            job = self._registry._jobs.get(self._job_id)
            if not job:
                return
            if n_done is not None:
                job.n_done = n_done
            if n_total is not None:
                job.n_total = n_total
            if progress is not None:
                job.progress = max(0.0, min(1.0, progress))
            elif n_done is not None and job.n_total > 0:
                job.progress = max(0.0, min(1.0, n_done / job.n_total))
            self._registry._persist(job)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_lock_file(path: Path) -> "dict[str, str] | None":
    """Parse the key=value lockfile written by acquire_reindex_lock.

    Returns a dict of parsed fields, or None if the file is missing,
    unreadable, or in an unrecognised format.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    result: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip()
    # Must have at least pid and hostname to be a valid new-format lock.
    if "pid" not in result or "hostname" not in result:
        return None
    return result


def _pid_is_alive(pid: int) -> bool:
    """Return True if a process with `pid` exists on this host.

    Uses os.kill(pid, 0): sends signal 0 (no actual signal), which succeeds
    if the process exists and we have permission, or raises ProcessLookupError
    (ESRCH) if it does not exist, or PermissionError (EPERM) if it exists but
    we lack permission.

    On Windows, os.kill(pid, 0) raises OSError with errno EINVAL for invalid
    PIDs and succeeds (or raises PermissionError) for valid ones -- same
    semantics as POSIX for our purposes.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        # ESRCH: no such process.
        return False
    except PermissionError:
        # EPERM: process exists but we lack permission to signal it.
        return True
    except OSError:
        # Catch-all for any other platform-specific error (e.g. EINVAL on Windows
        # for truly invalid pid values). Treat as unknown -> conservatively alive.
        return True
