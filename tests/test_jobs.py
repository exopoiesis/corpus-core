"""Tests for jobs.py — async job registry."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from corpus_core.jobs import Job, JobError, JobRegistry


def _await_done(registry: JobRegistry, job_id: str, timeout: float = 5.0) -> dict:
    """Poll until job hits terminal state or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = registry.get(job_id)
        if info and info["state"] in ("done", "failed", "orphaned"):
            return info
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def test_submit_runs_function_and_records_result(tmp_path: Path):
    reg = JobRegistry(cache_dir=tmp_path)

    def work(handle):
        handle.update(n_done=5, n_total=5)
        return {"answer": 42}

    job_id = reg.submit("demo", work, n_total=5)
    info = _await_done(reg, job_id)

    assert info["state"] == "done"
    assert info["progress"] == 1.0
    assert info["result"] == {"answer": 42}
    assert info["kind"] == "demo"
    reg.shutdown()


def test_submit_failure_marks_failed_with_error(tmp_path: Path):
    reg = JobRegistry(cache_dir=tmp_path)

    def work(handle):
        raise JobError("explicit failure")

    job_id = reg.submit("demo", work)
    info = _await_done(reg, job_id)

    assert info["state"] == "failed"
    assert "explicit failure" in info["error"]
    reg.shutdown()


def test_unexpected_exception_marked_failed(tmp_path: Path):
    reg = JobRegistry(cache_dir=tmp_path)

    def work(handle):
        raise ValueError("unexpected")

    job_id = reg.submit("demo", work)
    info = _await_done(reg, job_id)

    assert info["state"] == "failed"
    assert "unexpected" in info["error"]
    reg.shutdown()


def test_progress_updates_persist_to_disk(tmp_path: Path):
    reg = JobRegistry(cache_dir=tmp_path)

    barrier = {"go": False}

    def work(handle):
        handle.update(n_done=3, n_total=10)
        # Spin until test releases us.
        while not barrier["go"]:
            time.sleep(0.01)
        return {"ok": True}

    job_id = reg.submit("demo", work, n_total=10)

    # Wait for progress update.
    deadline = time.time() + 2.0
    while time.time() < deadline:
        info = reg.get(job_id)
        if info and info["n_done"] == 3:
            break
        time.sleep(0.02)
    else:
        barrier["go"] = True
        reg.shutdown()
        pytest.fail("progress update never landed")

    # Read directly from disk to verify persistence.
    persisted = json.loads((tmp_path / "jobs" / f"{job_id}.json").read_text())
    assert persisted["n_done"] == 3
    assert persisted["progress"] == pytest.approx(0.3)
    assert persisted["state"] == "running"

    barrier["go"] = True
    info = _await_done(reg, job_id)
    assert info["state"] == "done"
    reg.shutdown()


def test_get_unknown_job_returns_none(tmp_path: Path):
    reg = JobRegistry(cache_dir=tmp_path)
    assert reg.get("nonexistent") is None
    reg.shutdown()


def test_list_recent_returns_sorted_by_started_at(tmp_path: Path):
    reg = JobRegistry(cache_dir=tmp_path)

    ids = []
    for i in range(3):
        ids.append(reg.submit("demo", lambda h, x=i: {"i": x}))
        # Sleep ensures monotonic started_at timestamps.
        time.sleep(0.02)

    for j in ids:
        _await_done(reg, j)

    listed = reg.list_recent(limit=10)
    assert len(listed) == 3
    starts = [j["started_at"] for j in listed]
    assert starts == sorted(starts, reverse=True)
    reg.shutdown()


def test_reindex_lock_acquire_release(tmp_path: Path):
    reg = JobRegistry(cache_dir=tmp_path)
    assert reg.acquire_reindex_lock() is True
    assert reg.acquire_reindex_lock() is False  # already held
    reg.release_reindex_lock()
    assert reg.acquire_reindex_lock() is True
    reg.release_reindex_lock()
    reg.shutdown()


def test_persisted_running_job_marked_orphaned_on_reload(tmp_path: Path):
    """Simulate server crash mid-job: write a 'running' file, then reload."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    (jobs_dir / "abc123.json").write_text(json.dumps({
        "job_id": "abc123",
        "kind": "demo",
        "state": "running",
        "progress": 0.5,
        "n_total": 10,
        "n_done": 5,
        "started_at": "2026-05-01T12:00:00+00:00",
        "finished_at": None,
        "result": None,
        "error": None,
        "args": {},
    }), encoding="utf-8")

    reg = JobRegistry(cache_dir=tmp_path)
    info = reg.get("abc123")
    assert info is not None
    assert info["state"] == "orphaned"
    assert "restarted" in info["error"]
    reg.shutdown()


def test_old_terminal_jobs_cleaned_up_on_reload(tmp_path: Path):
    """Done/failed jobs older than retain_days are deleted on init."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    (jobs_dir / "old.json").write_text(json.dumps({
        "job_id": "old", "kind": "demo", "state": "done",
        "started_at": "2020-01-01T00:00:00+00:00",
        "finished_at": "2020-01-01T00:01:00+00:00",
        "progress": 1.0, "n_done": 0, "n_total": 0,
        "result": {}, "error": None, "args": {},
    }), encoding="utf-8")

    reg = JobRegistry(cache_dir=tmp_path, retain_days=7)
    assert reg.get("old") is None
    assert not (jobs_dir / "old.json").exists()
    reg.shutdown()


def test_stale_lockfile_cleared_on_init(tmp_path: Path):
    fulltext = tmp_path / "fulltext"
    fulltext.mkdir()
    (fulltext / ".reindex.lock").write_text("stale")

    reg = JobRegistry(cache_dir=tmp_path)
    assert not (fulltext / ".reindex.lock").exists()
    reg.shutdown()


def test_get_rereads_disk_when_inmemory_state_is_running(tmp_path: Path):
    """U1: If in-memory says 'running' but disk has progressed to 'done',
    get() must return the disk truth. Otherwise callers see stale 'running 0%'
    long after the job actually finished — observed 3× during 2026-05-08
    dogfood. Reading disk for terminal states is also cheap (small JSON).
    """
    reg = JobRegistry(cache_dir=tmp_path)

    # In-memory job stuck in 'running 0%' (simulates a missed in-memory update).
    reg._jobs["ghost"] = Job(  # noqa: SLF001
        job_id="ghost", kind="demo", state="running",
        progress=0.0, n_total=5, n_done=0,
    )

    # Disk reflects the actual completed state.
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    (jobs_dir / "ghost.json").write_text(json.dumps({
        "job_id": "ghost", "kind": "demo", "state": "done",
        "progress": 1.0, "n_total": 5, "n_done": 5,
        "started_at": "2026-05-08T12:00:00+00:00",
        "finished_at": "2026-05-08T12:01:00+00:00",
        "result": {"ok": True}, "error": None, "args": {},
    }), encoding="utf-8")

    info = reg.get("ghost")
    assert info is not None
    assert info["state"] == "done"
    assert info["progress"] == 1.0
    assert info["result"] == {"ok": True}
    reg.shutdown()


def test_get_does_not_reread_disk_for_terminal_inmemory_state(tmp_path: Path):
    """The disk-rehydrate path should only fire when in-memory state is
    pending/running. Terminal states (done/failed/orphaned) are authoritative
    in-memory because that's the last write _run made — no need to pay the
    JSON-read cost."""
    reg = JobRegistry(cache_dir=tmp_path)

    def work(handle):
        return {"answer": 42}

    job_id = reg.submit("demo", work)
    info = _await_done(reg, job_id)
    assert info["state"] == "done"

    # Tamper with the persisted file to a value that disagrees with memory.
    # If get() were re-reading disk unconditionally, we'd see the tamper.
    persisted = tmp_path / "jobs" / f"{job_id}.json"
    persisted.write_text(json.dumps({
        "job_id": job_id, "kind": "demo", "state": "failed",
        "progress": 1.0, "n_total": 0, "n_done": 0,
        "started_at": info["started_at"], "finished_at": info["finished_at"],
        "result": {"answer": 42}, "error": "tampered", "args": {},
    }), encoding="utf-8")

    again = reg.get(job_id)
    assert again["state"] == "done"  # memory wins for terminal states
    assert again["error"] is None
    reg.shutdown()


def test_get_disk_reread_handles_corrupted_json(tmp_path: Path):
    """Disk read is best-effort: corrupted JSON falls back to in-memory state
    rather than crashing the tool call."""
    reg = JobRegistry(cache_dir=tmp_path)
    reg._jobs["ghost"] = Job(  # noqa: SLF001
        job_id="ghost", kind="demo", state="running",
        progress=0.5, n_total=4, n_done=2,
    )

    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    (jobs_dir / "ghost.json").write_text("{not valid json", encoding="utf-8")

    info = reg.get("ghost")
    assert info is not None
    assert info["state"] == "running"
    assert info["n_done"] == 2
    reg.shutdown()
