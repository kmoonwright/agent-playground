# Tracing MCP client/server calls

The Model Context Protocol (MCP) lets an agent call tools, fetch resources, and receive prompts from an independent server process, over a wire protocol (commonly stdio or HTTP). Because the client and server are separate processes, a naive setup produces two disconnected traces — one per process — instead of one trace showing the whole call.

## What MCPInstrumentor actually does

`openinference-instrumentation-mcp`'s `MCPInstrumentor` is unusual among OpenInference instrumentors: **it emits no spans of its own.** Its only job is context propagation — it patches the MCP SDK so the OpenTelemetry context active when a client makes a tool call is carried across the wire protocol to the server, and restored there before the server's own handler runs.

The practical effect: any spans you create independently on the client side and the server side get linked into a single trace, with the server's span parented correctly under the client's call, as long as both processes are instrumented and exporting to the same project.

## What you still have to do yourself

Since the instrumentor produces no spans, whatever your trace shows for "this was an MCP call" comes from spans you create manually on both sides — a span around the client's tool-call invocation, and a span around the server's tool handler. Nothing about their names or attributes is required by the protocol or the instrumentor; it's a convention you choose. A trace with generic span names for both sides looks identical to any other tool call — nothing marks it as having crossed a process boundary unless you name or tag it that way yourself.

## What crosses the wire, and what doesn't

Only standard OpenTelemetry trace context (trace id, span id, flags) crosses the MCP
wire — that's the propagator's whole job, and it's what makes server-side spans
parent correctly under the client's call. Session id, metadata, and tags set via
OpenInference's `using_attributes()` are stored as plain in-process OpenTelemetry
Context values, not as W3C Baggage, so they are not part of what any propagator
serializes — they never reach the server process. If server-side spans need to carry
the same session id as the client-side spans (for session-level evaluation, say),
pass it explicitly as a tool argument and re-apply it with `using_attributes()` on
the server. Don't assume anything beyond trace/span linkage crosses automatically.

## Setup shape

Both the client process and the server process call `register()` and `MCPInstrumentor().instrument(tracer_provider=...)` independently, pointed at the same Arize project. One operational detail specific to a stdio-transport server: stdout is the wire protocol itself, so the server process must not print anything to stdout — anything printed there corrupts the JSON-RPC framing the client is trying to read. `register()`'s `verbose` argument defaults to `True` and prints a startup banner straight to stdout regardless of any other logging option — pass `verbose=False` explicitly, on both sides, or a stdio server breaks the moment real credentials are configured (it can look fine with no credentials, since the no-credentials fallback path never calls `register()` at all).
