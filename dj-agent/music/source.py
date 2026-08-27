"""The Source protocol and the Track record every source must produce.

No HTTP, no audio decoding, no tracing — this module exists so `library.py`,
`session.py`, and the agents can talk about tracks without knowing whether they
came from a web API or a folder on disk.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import config


def track_id_for(stable_uri: str) -> str:
    """Stable id from whatever uniquely names a track within its source.

    ccMixter uses the file page URL; local files use the absolute path. Both are
    stable across runs, which is what makes the on-disk cache reusable.
    """
    return hashlib.sha1(stable_uri.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Track:
    """A candidate track. `bpm=None` means analysis has to compute it."""

    source: str
    track_id: str
    title: str
    artist: str
    fetch_uri: str
    bpm: float | None = None
    bpm_source: str = "unknown"
    duration_s: float | None = None
    license_name: str | None = None
    license_url: str | None = None
    license_verified: bool = False
    page_url: str | None = None
    tags: tuple[str, ...] = ()
    size_bytes: int | None = None
    # The source's payload, stored verbatim. Attribution is derived from this,
    # never from our parsing, so a parser change can't silently mis-credit.
    raw: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def label(self) -> str:
        return f"{self.title} — {self.artist}"

    def attribution(self) -> str:
        parts = [f'"{self.title}" by {self.artist}']
        if self.license_name:
            parts.append(self.license_name)
        if self.page_url:
            parts.append(self.page_url)
        return " — ".join(parts)


@dataclass(frozen=True)
class SearchQuery:
    bpm_min: float
    bpm_max: float
    tags: tuple[str, ...] = ()
    limit: int = 8


def rank_by_tempo(tracks: list[Track], query: SearchQuery) -> list[Track]:
    """Closest to the middle of the requested range first, unknown tempo last."""
    mid = (query.bpm_min + query.bpm_max) / 2.0
    return sorted(tracks, key=lambda t: abs(t.bpm - mid) if t.bpm else float("inf"))


def within_size_limits(track: Track) -> bool:
    """Reject a track a source has already told us is too big to decode.

    Both limits are advisory here because they come from source metadata; the
    authoritative checks are the byte counter in `CCMixterSource.fetch` and the
    sample count in `audio.analysis.analyze`.
    """
    if track.size_bytes and track.size_bytes > config.MAX_TRACK_BYTES:
        return False
    if track.duration_s and track.duration_s > config.MAX_TRACK_SECONDS:
        return False
    return True


class SourceError(RuntimeError):
    """Raised for anything a source can't do — never leaks past a tool call."""


@runtime_checkable
class Source(Protocol):
    name: str

    def search(self, query: SearchQuery) -> list[Track]: ...

    def fetch(self, track: Track, dest_dir: Path) -> Path: ...
