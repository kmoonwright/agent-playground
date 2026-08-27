"""Which sources are enabled for this run, and in what order.

Adding a source means writing one module and adding one line here. Nothing
else in the project names a concrete source.
"""

from __future__ import annotations

from pathlib import Path

from music.source import Source, SourceError

KNOWN = ("ccmixter", "local")


def parse_names(spec: str) -> tuple[str, ...]:
    names = tuple(n.strip().lower() for n in spec.split(",") if n.strip())
    unknown = [n for n in names if n not in KNOWN]
    if unknown:
        raise SourceError(f"unknown source(s): {', '.join(unknown)}. known: {', '.join(KNOWN)}")
    if not names:
        raise SourceError("at least one source is required")
    return names


def build_sources(
    spec: str,
    *,
    crate_dir: Path,
    offline: bool = False,
) -> dict[str, Source]:
    """Instantiate the requested sources, honouring --offline."""
    names = parse_names(spec)
    built: dict[str, Source] = {}
    for name in names:
        if name == "local":
            from music.local import LocalSource

            built[name] = LocalSource(crate_dir)
        elif name == "ccmixter":
            from music.ccmixter import CCMixterSource

            # Offline does not mean "disabled": the source still serves from the
            # on-disk API cache and from already-downloaded files, which is what
            # makes `--seed-cache` once, then `--offline` forever, work. Any
            # request that would actually hit the network raises instead.
            built[name] = CCMixterSource(offline=offline)
    if not built:
        raise SourceError("no usable sources — pass at least one of: " + ", ".join(KNOWN))
    return built


def default_spec(offline: bool) -> str:
    return "local" if offline else "ccmixter"
