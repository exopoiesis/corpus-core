"""Local stdio→remote-HTTP proxy.

Used when the user runs `arxiv-radar-mcp --remote user@host`. We:
  1. Open an SSH tunnel: localhost:<random>  →  user@host:127.0.0.1:8765
  2. Connect a streamable-HTTP MCP client to localhost:<random>/mcp
  3. Run an stdio MCP server bound to Claude Desktop on this side
  4. Bridge the two: every JSON-RPC message that comes in on stdio gets
     forwarded to the remote backend, and every response is piped back.

This way the heavy backend (Qwen3-4B + GPU + persistent reindex jobs)
lives on the GPU host, while the user's laptop runs only ~80 LOC of
proxy. SSH provides perimeter security — no Bearer tokens, no TLS certs,
just SSH keys the user already has.

Cross-platform: works on Windows because OpenSSH is built into modern
Windows (10+). On macOS/Linux it just uses /usr/bin/ssh.
"""
from __future__ import annotations

import logging
import shutil
import socket
import subprocess
import sys
import time

import anyio

LOG = logging.getLogger(__name__)


def run_proxy(target: str, remote_port: int, ssh_binary: str) -> int:
    """Public entry point — block until the proxy session ends.

    Returns 0 on clean exit, non-zero on tunnel-setup failure.
    """
    if shutil.which(ssh_binary) is None:
        LOG.error(f"ssh binary not found on PATH: {ssh_binary!r}. "
                  "On Windows install OpenSSH or set --ssh-binary explicitly.")
        return 2

    local_port = _pick_free_port()
    LOG.info(f"opening SSH tunnel  127.0.0.1:{local_port} → {target}:127.0.0.1:{remote_port}")

    tunnel = _start_tunnel(ssh_binary, target, local_port, remote_port)
    try:
        if not _wait_for_port(local_port, timeout=15.0):
            LOG.error("SSH tunnel didn't open within 15 s — check ssh keys / "
                      "host reachability / that the backend is running on the remote host.")
            tunnel.terminate()
            return 3
        LOG.info(f"tunnel up. backend at http://127.0.0.1:{local_port}/mcp")

        try:
            anyio.run(_run_bridge, f"http://127.0.0.1:{local_port}/mcp")
        except KeyboardInterrupt:
            LOG.info("interrupted, closing tunnel")
            return 0
        return 0
    finally:
        if tunnel.poll() is None:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel.kill()


def _pick_free_port() -> int:
    """Bind to port 0 → kernel assigns an unused port → close → return it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_tunnel(ssh: str, target: str, local: int, remote: int) -> subprocess.Popen:
    """ssh -N -o ExitOnForwardFailure=yes -L L:127.0.0.1:R user@host

    `-N` — no remote command, just keep the tunnel.
    `ExitOnForwardFailure=yes` — die if the port is already taken on the
                                  remote, instead of silently degrading.
    """
    cmd = [
        ssh, "-N",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-L", f"127.0.0.1:{local}:127.0.0.1:{remote}",
        target,
    ]
    LOG.debug(f"ssh cmd: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for_port(port: int, timeout: float) -> bool:
    """Poll TCP connect to 127.0.0.1:port until it succeeds or we time out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


async def _run_bridge(url: str, *,
                      max_consecutive_failures: int = 10,
                      max_backoff_s: float = 30.0) -> None:
    """Connect MCP streamable-HTTP client to `url` and pipe stdio↔remote.

    Reconnects automatically when the backend session ends (e.g. backend
    container restarted, SSH tunnel dropped, server-initiated close). The
    local stdio is opened once for the lifetime of the proxy so Claude
    Code's view of the proxy process never changes — events and tool
    calls flow over freshly-initialized backend sessions transparently.

    `max_consecutive_failures` caps the number of *back-to-back* failed
    connection attempts before we give up and let Claude Code's MCP
    supervisor respawn us. The counter resets every time a backend session
    successfully establishes (i.e. at least one initialize handshake
    completed). After exhaustion we return so the parent process can exit
    non-zero.

    Imported lazily so unit tests don't need the SDK loaded.
    """
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (local_read, local_write):
        await _bridge_loop(
            connect=lambda: streamablehttp_client(url),
            local_read=local_read,
            local_write=local_write,
            max_consecutive_failures=max_consecutive_failures,
            max_backoff_s=max_backoff_s,
        )


async def _bridge_loop(*, connect, local_read, local_write,
                       max_consecutive_failures: int = 10,
                       max_backoff_s: float = 30.0,
                       sleep=None) -> None:
    """Drive one stdio↔remote session at a time; reconnect on disconnect.

    `connect` is a zero-argument callable that returns an async context
    manager yielding `(remote_read, remote_write, _aux)` — same shape as
    the MCP SDK's `streamablehttp_client(url)`. The decoupling exists so
    the reconnect loop can be unit-tested with fake streams without
    spinning up an HTTP server.

    `sleep` is an injection point for tests; defaults to `anyio.sleep`.
    """
    if sleep is None:
        sleep = anyio.sleep

    consecutive_failures = 0
    backoff = 1.0

    while True:
        session_succeeded = False
        try:
            async with connect() as triple:
                remote_read, remote_write = triple[0], triple[1]
                session_succeeded = True
                LOG.info("backend session up; bridging stdio↔HTTP")
                consecutive_failures = 0
                backoff = 1.0
                async with anyio.create_task_group() as tg:
                    tg.start_soon(_pipe, local_read, remote_write)
                    tg.start_soon(_pipe, remote_read, local_write)
                LOG.info("backend session ended cleanly; will reconnect")
        except (anyio.BrokenResourceError, anyio.EndOfStream,
                ConnectionError, OSError) as e:
            LOG.warning(
                f"backend disconnected: {type(e).__name__}: {e}"
            )
        except Exception as e:  # noqa: BLE001
            LOG.exception(f"unexpected proxy error: {e}")

        if not session_succeeded:
            consecutive_failures += 1
        if consecutive_failures >= max_consecutive_failures:
            LOG.error(
                f"giving up after {consecutive_failures} consecutive backend "
                f"failures; exiting (Claude Code's MCP supervisor will respawn)"
            )
            return

        LOG.info(f"reconnecting to backend in {backoff:.1f}s "
                 f"(consecutive failures: {consecutive_failures}/"
                 f"{max_consecutive_failures})")
        await sleep(backoff)
        backoff = min(backoff * 2, max_backoff_s)


async def _pipe(read_stream, write_stream) -> None:
    """Forward every message from `read_stream` to `write_stream`. Exits when
    either end of the pair closes.

    NOTE: does **not** close `write_stream` on exit — the bridge loop
    re-uses the local stdio across reconnects, so closing it would
    terminate the whole proxy after the first backend disconnect. The
    remote streams are owned by `streamablehttp_client`'s context manager
    and get closed on its `__aexit__`."""
    try:
        async for msg in read_stream:
            await write_stream.send(msg)
    except anyio.BrokenResourceError:
        pass
