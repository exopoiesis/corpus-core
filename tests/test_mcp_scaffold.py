"""mcp_scaffold tests: dispatcher routing, build_mcp_app, transports.

The MCP SDK is imported lazily inside `build_mcp_app` /
`serve_stdio` / `serve_streamable_http`; this suite covers the pure
plumbing without actually starting a transport.
"""
from __future__ import annotations

import asyncio

import pytest

from corpus_core.mcp_scaffold import (
    BackgroundTaskFactory,
    Dispatcher,
    build_mcp_app,
    make_method_dispatcher,
    serve_stdio,
    serve_streamable_http,
)


class _StubHandler:
    """Tiny handler exposing a fake tool-method surface."""

    def search(self, query: str, k: int = 10) -> dict:
        return {"called": "search", "query": query, "k": k}

    def list_things(self) -> list[dict]:
        return [{"a": 1}]

    def boom(self) -> None:
        raise RuntimeError("intentional")


# ----- make_method_dispatcher ----------------------------------------------

def test_dispatcher_routes_to_method():
    d = make_method_dispatcher(_StubHandler(), ["search", "list_things"])
    assert d("search", {"query": "dft", "k": 3}) == {
        "called": "search", "query": "dft", "k": 3,
    }


def test_dispatcher_handles_none_arguments():
    d = make_method_dispatcher(_StubHandler(), ["list_things"])
    assert d("list_things", None) == [{"a": 1}]


def test_dispatcher_unknown_tool_returns_error():
    d = make_method_dispatcher(_StubHandler(), ["search"])
    out = d("drop_database", {})
    assert "error" in out and "drop_database" in out["error"]


def test_dispatcher_rejects_dunder_names():
    d = make_method_dispatcher(_StubHandler(), ["__init__"])
    out = d("__init__", {})
    assert "error" in out


def test_dispatcher_rejects_names_outside_allowlist():
    """Even a real method must be in the allowlist to be reachable."""
    d = make_method_dispatcher(_StubHandler(), ["search"])
    # `list_things` exists on the handler but isn't allowed.
    out = d("list_things", {})
    assert "error" in out


def test_dispatcher_typeerror_becomes_error_dict():
    d = make_method_dispatcher(_StubHandler(), ["search"])
    out = d("search", {"wrong_kw": "x"})
    assert "error" in out and "search" in out["error"]


def test_dispatcher_allows_other_exceptions_to_propagate():
    """RuntimeError from inside the tool body is NOT swallowed — only
    bad-args TypeError. The MCP SDK's call_tool wrapper converts it to
    an error response upstream; the dispatcher just lets it through."""
    d = make_method_dispatcher(_StubHandler(), ["boom"])
    with pytest.raises(RuntimeError, match="intentional"):
        d("boom", {})


# ----- build_mcp_app -------------------------------------------------------

def test_build_mcp_app_returns_mcp_server():
    """Construction is lazy — verify it succeeds with a synthetic
    catalogue but don't try to call a tool (would need an MCP session)."""
    handler = _StubHandler()
    specs = [
        {"name": "search",
         "description": "Search.",
         "inputSchema": {"type": "object",
                         "properties": {"query": {"type": "string"}},
                         "required": ["query"]}},
    ]
    dispatcher = make_method_dispatcher(handler, [s["name"] for s in specs])
    app = build_mcp_app(server_name="x-test", tool_specs=specs, dispatcher=dispatcher)
    assert app is not None
    # Server name appears on the constructed instance.
    assert hasattr(app, "list_tools")


def test_build_mcp_app_sets_instructions():
    """Server-level `instructions` metadata reaches the MCP Server (it's
    how we document the /upload + /download HTTP side-channels to clients)."""
    specs = [{"name": "search", "description": "Search.",
              "inputSchema": {"type": "object", "properties": {}}}]
    dispatcher = make_method_dispatcher(_StubHandler(), ["search"])
    note = "use GET /download?id=<id> for the full bundle"
    app = build_mcp_app(server_name="x-test", tool_specs=specs,
                        dispatcher=dispatcher, instructions=note)
    assert app.instructions == note
    # Default stays None when omitted.
    app2 = build_mcp_app(server_name="x-test", tool_specs=specs,
                         dispatcher=dispatcher)
    assert app2.instructions is None


# ----- serve_stdio / serve_streamable_http background-task contract --------

async def _quick_bg() -> None:
    """Background factory that finishes immediately — proves spawn works."""
    return None


def test_background_factory_signature_typecheck():
    """BackgroundTaskFactory is `Callable[[], Awaitable[None]]`. Verify
    the type-alias is defined and the factory pattern is invocable."""
    factories: list[BackgroundTaskFactory] = [lambda: _quick_bg()]
    coros = [f() for f in factories]
    assert all(asyncio.iscoroutine(c) for c in coros)
    # Drain so we don't leak un-awaited coroutines.
    for c in coros:
        c.close()


def test_dispatcher_type_alias_exists():
    """Dispatcher type-alias is exported."""
    handler = _StubHandler()
    d: Dispatcher = make_method_dispatcher(handler, ["search"])
    assert callable(d)


# Real serve_stdio / serve_streamable_http need stdio descriptors / a
# running event loop with the MCP SDK live; they're integration-tested
# via host projects (arxiv-radar-mcp + lab-corpus-mcp). Cover only the
# happy-path import here.
def test_transport_entrypoints_importable():
    assert callable(serve_stdio)
    assert callable(serve_streamable_http)
