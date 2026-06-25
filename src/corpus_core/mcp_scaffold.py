"""Generic MCP server scaffolding — transport, dispatch, app builder.

Both `arxiv-radar-mcp` and `lab-corpus-mcp` use this. Each downstream
project supplies:
  * a server name (advertised to MCP clients),
  * a `tool_specs` list — JSON-Schema tool catalogue,
  * a `dispatcher` — most often built with `make_method_dispatcher`,
  * optional `background_tasks` — zero-arg coroutine factories
    (encoder warm-up, periodic refresh, watchdogs, ...).

What is intentionally NOT here:
  * corpus / index loading (downstream choice; radar config differs
    from lab config),
  * tool-method implementations (those live on the handler object the
    dispatcher routes to),
  * domain-specific background loops — defined downstream and passed
    in as `background_tasks`.

Phase 1.5 of the corpus-core extraction (see
`arxiv-radar-mcp/docs/PLAN_CORE_EXTRACTION.md`).
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Iterable

LOG = logging.getLogger(__name__)


Dispatcher = Callable[[str, "dict[str, Any] | None"], Any]
"""Routes (tool_name, arguments) → result. Returns `{"error": ...}` for
unknown tools or bad arguments; everything else is forwarded as-is to
the JSON serializer."""

BackgroundTaskFactory = Callable[[], Awaitable[None]]
"""Zero-arg callable returning a fresh coroutine each time it is called.
The scaffold spawns these alongside the MCP transport and cancels them
when the transport exits. Factory (not a bare coroutine) so re-running
the server produces fresh tasks instead of reusing exhausted ones."""


def make_method_dispatcher(handler: object, tool_names: Iterable[str]) -> Dispatcher:
    """Build a dispatcher that maps `tool_name` → `handler.<tool_name>(**args)`.

    The returned callable validates that `name` is in the allowlist
    (defends against accidental dunder / private leakage even if a
    same-named method exists on `handler`) and converts `TypeError` from
    a bad-argument call into an `{"error": ...}` dict.

    Pure / synchronous — easy to unit-test without spinning up the SDK.
    """
    allowed = frozenset(tool_names)

    def dispatch(name: str, arguments: dict[str, Any] | None) -> Any:
        if name not in allowed or name.startswith("_"):
            return {"error": f"unknown tool: {name!r}"}
        method = getattr(handler, name, None)
        if method is None:
            return {"error": f"unknown tool: {name!r}"}
        try:
            return method(**(arguments or {}))
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}

    return dispatch


def build_mcp_app(
    *,
    server_name: str,
    tool_specs: list[dict[str, Any]],
    dispatcher: Dispatcher,
    instructions: str | None = None,
):
    """Construct an `mcp.server.Server` with `tool_specs` as the live
    catalogue and `dispatcher` as the call router.

    `instructions` is server-level metadata returned in the MCP
    `initialize` result — the place to document things clients can't learn
    from the tool catalogue, e.g. the binary HTTP side-channels
    (`GET /download`, `POST /upload`) that JSON-RPC can't carry. Compatible
    clients surface it to the model, so it travels with the server for
    every user, not just one local memory.

    Imported lazily so unit-tests don't pull in the MCP SDK just to
    inspect the catalogue.
    """
    from mcp.server import Server
    from mcp.types import TextContent, Tool

    app: Server = Server(server_name, instructions=instructions)

    @app.list_tools()
    async def _list_tools() -> list[Tool]:
        return [Tool(**spec) for spec in tool_specs]

    @app.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
        # Run the synchronous dispatcher in a thread so it does not block the
        # event loop.  Both the stdio and streamable-HTTP transports are async;
        # a long-running synchronous tool (embedding search, chunker rebuild)
        # would otherwise stall all other async tasks for its duration.
        #
        # Any uncaught exception from the dispatcher (or the tool method it
        # calls) is caught here and returned as a structured error response so
        # the MCP session stays alive.  make_method_dispatcher already handles
        # TypeError (bad args) and unknown-tool; this layer catches everything
        # else (RuntimeError, ValueError, IO errors, ...).
        try:
            result = await asyncio.to_thread(dispatcher, name, arguments)
        except Exception as e:  # noqa: BLE001
            LOG.exception(f"_call_tool: unhandled exception in tool {name!r}")
            result = {"error": f"{type(e).__name__}: {e}"}
        text = json.dumps(result, indent=1, ensure_ascii=False, default=str)
        return [TextContent(type="text", text=text)]

    return app


def _spawn_background(factories: Iterable[BackgroundTaskFactory]) -> list[asyncio.Task]:
    return [
        asyncio.create_task(make(), name=getattr(make, "__name__", "background"))
        for make in factories
    ]


async def serve_stdio(
    *,
    server_name: str,
    tool_specs: list[dict[str, Any]],
    dispatcher: Dispatcher,
    background_tasks: Iterable[BackgroundTaskFactory] = (),
    instructions: str | None = None,
) -> None:
    """Async stdio MCP loop. Spawns each `background_tasks` coroutine on
    entry and cancels them when the transport returns.
    """
    from mcp.server.stdio import stdio_server

    app = build_mcp_app(
        server_name=server_name, tool_specs=tool_specs, dispatcher=dispatcher,
        instructions=instructions,
    )
    bg = _spawn_background(background_tasks)
    try:
        async with stdio_server() as (read, write):
            await app.run(read, write, app.create_initialization_options())
    finally:
        for t in bg:
            t.cancel()


async def serve_streamable_http(
    *,
    server_name: str,
    tool_specs: list[dict[str, Any]],
    dispatcher: Dispatcher,
    host: str,
    port: int,
    background_tasks: Iterable[BackgroundTaskFactory] = (),
    extra_routes: Iterable[Any] = (),
    instructions: str | None = None,
) -> None:
    """Async streamable-HTTP MCP loop (protocol 2025-03-26+).

    One process serves many sessions; ideal for a GPU host where the
    encoder loads once. Bind `host="127.0.0.1"` in production — the
    perimeter is an SSH tunnel, NOT this server.

    `extra_routes` mounts downstream Starlette `Route`/`Mount` objects
    alongside `/mcp` on the same port — used for binary side-channels that
    JSON-RPC can't carry (lab-corpus `POST /upload`, arxiv-radar
    `GET /download`). They share the perimeter (SSH tunnel) with `/mcp`.
    """
    import uvicorn
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.routing import Mount

    app = build_mcp_app(
        server_name=server_name, tool_specs=tool_specs, dispatcher=dispatcher,
        instructions=instructions,
    )
    session_manager = StreamableHTTPSessionManager(
        app=app, json_response=True, stateless=False,
    )

    async def _handle(scope, receive, send):
        await session_manager.handle_request(scope, receive, send)

    starlette_app = Starlette(
        routes=[Mount("/mcp", app=_handle), *extra_routes],
    )

    bg = _spawn_background(background_tasks)
    try:
        async with session_manager.run():
            uv_config = uvicorn.Config(
                starlette_app, host=host, port=port,
                log_level="info", access_log=False,
            )
            await uvicorn.Server(uv_config).serve()
    finally:
        for t in bg:
            t.cancel()
