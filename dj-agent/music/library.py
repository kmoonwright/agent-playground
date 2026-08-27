"""The on-disk cache: track metadata, analysis sidecars, decoded audio, credits.

One directory per track under cache/tracks/<track_id>/, keyed by a hash of the
source's stable URI so the cache survives restarts and is shared across sources.
The source's payload is stored verbatim in meta.json — attribution is always
derived from that, never re-parsed from a search result that may have changed
shape.

`credits.jsonl` is append-only and is written when a deck *actually starts
sounding*, not when a track is queued, so the setlist reflects what was played.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import config
from schema import ANALYSIS_SCHEMA_VERSION, Analysis, CreditEntry, TrackRef
from music.source import Source, Track


LICENSE_NOTE = "license unverified"


def license_flag(verified: bool) -> str:
    """One wording for "we could not confirm this license", used everywhere."""
    return "" if verified else f"  ({LICENSE_NOTE})"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def track_dir(track_id: str) -> Path:
    """The cache directory for a track id, refusing to escape the cache.

    `schema.TrackId` already constrains every id the agents can supply, but this
    is the check that survives someone adding a call site that skips the models.
    """
    directory = (config.TRACKS_DIR / track_id).resolve()
    if not directory.is_relative_to(config.TRACKS_DIR.resolve()):
        raise ValueError(f"track id escapes the cache directory: {track_id!r}")
    return directory


def meta_path(track_id: str) -> Path:
    return track_dir(track_id) / "meta.json"


def analysis_path(track_id: str) -> Path:
    return track_dir(track_id) / "analysis.json"


def audio_path(track_id: str) -> Path:
    return track_dir(track_id) / "audio.f32.npy"


def source_file(track_id: str) -> Path | None:
    return next(iter(sorted(track_dir(track_id).glob("source.*"))), None)


# ---------------------------------------------------------------- meta


def save_track(track: Track) -> Path:
    """Persist a Track (including its verbatim source payload)."""
    directory = track_dir(track.track_id)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "track_id": track.track_id,
        "source": track.source,
        "title": track.title,
        "artist": track.artist,
        "fetch_uri": track.fetch_uri,
        "bpm": track.bpm,
        "bpm_source": track.bpm_source,
        "duration_s": track.duration_s,
        "license_name": track.license_name,
        "license_url": track.license_url,
        "license_verified": track.license_verified,
        "page_url": track.page_url,
        "tags": list(track.tags),
        "size_bytes": track.size_bytes,
        "attribution": track.attribution(),
        "cached_at": now_iso(),
        "raw": track.raw,
    }
    path = meta_path(track.track_id)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def load_track(track_id: str) -> Track | None:
    path = meta_path(track_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return _track_from_meta(data)
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        # A missing key must fail the same way a missing file does; otherwise the
        # difference is an existence oracle for whatever the path pointed at.
        return None


def _track_from_meta(data: dict[str, Any]) -> Track:
    return Track(
        source=data["source"],
        track_id=data["track_id"],
        title=data["title"],
        artist=data["artist"],
        fetch_uri=data["fetch_uri"],
        bpm=data.get("bpm"),
        bpm_source=data.get("bpm_source", "unknown"),
        duration_s=data.get("duration_s"),
        license_name=data.get("license_name"),
        license_url=data.get("license_url"),
        license_verified=bool(data.get("license_verified")),
        page_url=data.get("page_url"),
        tags=tuple(data.get("tags") or ()),
        size_bytes=data.get("size_bytes"),
        raw=data.get("raw") or {},
    )


# ---------------------------------------------------------------- analysis


def save_analysis(analysis: Analysis) -> Path:
    directory = track_dir(analysis.track_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = analysis_path(analysis.track_id)
    path.write_text(analysis.model_dump_json(indent=2))
    return path


def load_analysis(track_id: str, *, file_sha1: str | None = None) -> Analysis | None:
    """Load a cached analysis, rejecting stale schema versions or changed files."""
    path = analysis_path(track_id)
    if not path.exists():
        return None
    try:
        analysis = Analysis.model_validate_json(path.read_text())
    except Exception:
        return None
    if analysis.schema_version != ANALYSIS_SCHEMA_VERSION:
        return None
    if file_sha1 is not None and analysis.file_sha1 != file_sha1:
        return None
    return analysis


def is_ready(track_id: str) -> bool:
    """True when a track can go straight to a deck with no network and no librosa."""
    return (
        load_analysis(track_id) is not None
        and audio_path(track_id).exists()
        and meta_path(track_id).exists()
    )


# ---------------------------------------------------------------- offline search


def cached_tracks() -> list[Track]:
    if not config.TRACKS_DIR.exists():
        return []
    ids = sorted(p.name for p in config.TRACKS_DIR.iterdir() if p.is_dir())
    return [t for t in (load_track(i) for i in ids) if t is not None]


def search_cache(
    bpm_min: float,
    bpm_max: float,
    tags: Iterable[str] = (),
    limit: int = 8,
    sources: Iterable[str] | None = None,
) -> list[Track]:
    """Search what is already on disk. Used by --offline and to prefer cache hits.

    `sources` must be passed whenever the result may be played: the cache
    outlives any single run, so it can hold tracks from a source that is not
    enabled right now, and offering one of those produces a track nothing can
    fetch.
    """
    wanted = {t.lower() for t in tags}
    enabled = set(sources) if sources is not None else None
    mid = (bpm_min + bpm_max) / 2.0
    scored: list[tuple[float, Track]] = []
    for track in cached_tracks():
        if enabled is not None and track.source not in enabled:
            continue
        bpm = effective_bpm(track)
        if bpm is None or not (bpm_min <= bpm <= bpm_max):
            continue
        if wanted:
            haystack = " ".join([track.title.lower(), track.artist.lower(), *track.tags])
            if not any(tag in haystack for tag in wanted):
                continue
        scored.append((abs(bpm - mid), track))
    scored.sort(key=lambda pair: pair[0])
    return [track for _, track in scored[:limit]]


def duration_of(track_id: str) -> float | None:
    """Analysed duration, which is authoritative over any metadata guess."""
    analysis = load_analysis(track_id)
    return analysis.duration_s if analysis else None


def bpm_of(track_id: str) -> float | None:
    """Effective BPM for a track we only have an id for."""
    track = load_track(track_id)
    return effective_bpm(track) if track else None


def effective_bpm(track: Track) -> float | None:
    """Analysis BPM if we have one, else whatever the source's metadata said."""
    analysis = load_analysis(track.track_id)
    return analysis.bpm_used if analysis else track.bpm


# ---------------------------------------------------------------- agent views


def to_ref(track: Track, *, target_bpm: float | None = None) -> TrackRef:
    """The compact shape an agent sees. Never includes raw payloads or URLs to fetch."""
    analysis = load_analysis(track.track_id)
    bpm = analysis.bpm_used if analysis else track.bpm
    delta = None
    if target_bpm and bpm:
        delta = round((bpm - target_bpm) / target_bpm * 100.0, 2)
    return TrackRef(
        track_id=track.track_id,
        source=track.source,
        title=track.title,
        artist=track.artist,
        bpm=bpm,
        bpm_source=(analysis.bpm_source if analysis else track.bpm_source),  # type: ignore[arg-type]
        duration_s=(analysis.duration_s if analysis else track.duration_s),
        license_name=track.license_name,
        license_verified=track.license_verified,
        cached=is_ready(track.track_id),
        bpm_delta_pct=delta,
        grid_confidence=(analysis.grid_confidence if analysis else None),
    )


# ---------------------------------------------------------------- credits


def fetch_and_analyze(track: Track, source: Source) -> tuple[Analysis, Any, bool]:
    """Get a track onto disk and analysed. Returns (analysis, audio, was_cached).

    The single path from "a Track someone chose" to "a buffer a deck can play".
    Used by the opening track, the loader thread, and cache seeding, so the
    cache-hit behaviour is identical in all three.
    """
    # Imported here to break the library <-> audio.analysis cycle; analysis
    # imports this module at the top, so one of the two directions has to be lazy.
    import audio.analysis as analysis_mod

    existing = source_file(track.track_id)
    path = existing or source.fetch(track, track_dir(track.track_id))
    save_track(track)
    record, audio = analysis_mod.ensure_analyzed(track, path)
    return record, audio, existing is not None


def append_credit(entry: CreditEntry) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with config.CREDITS_PATH.open("a") as fh:
        fh.write(entry.model_dump_json() + "\n")


def read_credits() -> list[CreditEntry]:
    if not config.CREDITS_PATH.exists():
        return []
    out: list[CreditEntry] = []
    for line in config.CREDITS_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(CreditEntry.model_validate_json(line))
        except Exception:
            continue
    return out


def credit_for(track: Track, bpm_used: float, seconds_played: float | None = None) -> CreditEntry:
    return CreditEntry(
        played_at=now_iso(),
        track_id=track.track_id,
        source=track.source,
        title=track.title,
        artist=track.artist,
        license_name=track.license_name,
        license_url=track.license_url,
        page_url=track.page_url,
        fetch_uri=track.fetch_uri,
        license_verified=track.license_verified,
        bpm_used=bpm_used,
        seconds_played=seconds_played,
    )


def write_setlist() -> Path:
    """Render credits.jsonl as a human-readable setlist with license flags."""
    entries = read_credits()
    target = config.OUT_DIR / "SETLIST.md"
    target.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Setlist", ""]
    if not entries:
        lines.append("_Nothing played yet._")
    unverified = 0
    for i, e in enumerate(entries, start=1):
        bits = [
            f"{i}. **{config.sanitize(e.title)}** — {config.sanitize(e.artist)}",
            f"{e.bpm_used:.1f} BPM",
        ]
        if e.license_name:
            bits.append(config.sanitize(e.license_name))
        if e.page_url:
            bits.append(f"<{e.page_url}>")
        line = "  \n   ".join([bits[0], " · ".join(bits[1:])])
        if not e.license_verified:
            unverified += 1
            line += f"  \n   ⚠️ {LICENSE_NOTE} — do not redistribute this mix"
        lines.append(line)
    if unverified:
        lines += [
            "",
            f"> {unverified} of {len(entries)} tracks have an unverified license. "
            "Local-crate files are unverified unless the crate ships a LICENSE.json.",
        ]
    target.write_text("\n".join(lines) + "\n")
    return target
