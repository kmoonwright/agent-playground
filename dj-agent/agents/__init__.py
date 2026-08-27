"""Everything that talks to a model: the host agent, its sub-agents, their tools.

Deliberately re-exports nothing. Importing this package must not pull in the LLM
SDKs, because `instrumentation.ensure_tracing()` has to patch them first, and it
runs after argv is parsed.
"""
