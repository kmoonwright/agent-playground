"""A static local library: whatever audio files are sitting in a crate directory.

This is the source that makes `--offline` real. It reads BPM, title, and artist
from tags where they exist and leaves `bpm=None` where they don't, so analysis
computes the tempo instead of guessing.

Licensing is handled honestly: a local file has no license unless the crate ships
a LICENSE.json mapping filenames to one, in which case the track is marked
`license_verified`. Otherwise it is labelled "local (unverified)" and every
credits line records that we could not confirm the license.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import config
from music.source import (
    SearchQuery,
    SourceError,
    Track,
    rank_by_tempo,
    track_id_for,
    within_size_limits,
)

AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aiff", ".aif"}
LICENSE_SIDECAR = "LICENSE.json"


def _read_tags(path: Path) -> dict[str, object]:
    """Best-effort tag read. mutagen returns wildly different shapes per format."""
    try:
        import mutagen
    except ImportError:  # pragma: no cover
        return {}
    try:
        audio = mutagen.File(path, easy=False)
    except Exception:
        return {}
    if audio is None:
        return {}

    out: dict[str, object] = {}
    if getattr(audio, "info", None) is not None:
        length = getattr(audio.info, "length", None)
        if length:
            out["duration_s"] = float(length)
        channels = getattr(audio.info, "channels", None)
        if channels:
            out["channels"] = int(channels)

    def first(*keys: str) -> str | None:
        tags = getattr(audio, "tags", None)
        if not tags:
            return None
        for key in keys:
            try:
                value = tags[key]
            except (KeyError, TypeError):
                continue
            if value is None:
                continue
            if isinstance(value, list) and value:
                value = value[0]
            text = getattr(value, "text", value)
            if isinstance(text, list) and text:
                text = text[0]
            text = str(text).strip()
            if text:
                return text
        return None

    title = first("TIT2", "title", "\xa9nam")
    artist = first("TPE1", "artist", "\xa9ART")
    bpm_raw = first("TBPM", "bpm", "tmpo")
    if title:
        out["title"] = title
    if artist:
        out["artist"] = artist
    if bpm_raw:
        try:
            bpm = float(str(bpm_raw).split()[0])
            if 40.0 <= bpm <= 250.0:
                out["bpm"] = bpm
        except ValueError:
            pass
    return out


class LocalSource:
    """Scans a directory tree once per search. Cheap enough for a crate."""

    name = "local"

    def __init__(self, crate_dir: Path) -> None:
        self.crate_dir = Path(crate_dir)
        self._licenses = self._load_licenses()

    def _load_licenses(self) -> dict[str, dict[str, str]]:
        path = self.crate_dir / LICENSE_SIDECAR
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SourceError(f"{path} is not readable JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise SourceError(f"{path} must be an object mapping filename -> license info")
        return {str(k): dict(v) for k, v in data.items() if isinstance(v, dict)}

    def scan(self) -> list[Track]:
        if not self.crate_dir.exists():
            return []
        tracks: list[Track] = []
        for path in sorted(self.crate_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
                continue
            tracks.append(self._to_track(path))
        return tracks

    def _to_track(self, path: Path) -> Track:
        tags = _read_tags(path)
        lic = self._licenses.get(path.name) or self._licenses.get(
            str(path.relative_to(self.crate_dir))
        )
        bpm = tags.get("bpm")
        stat = path.stat()
        return Track(
            source=self.name,
            track_id=track_id_for(str(path.resolve())),
            title=str(tags.get("title") or path.stem),
            artist=str(tags.get("artist") or "unknown"),
            fetch_uri=str(path.resolve()),
            bpm=float(bpm) if bpm else None,
            bpm_source="metadata" if bpm else "unknown",
            duration_s=float(tags["duration_s"]) if tags.get("duration_s") else None,
            license_name=(lic or {}).get("license_name") or "local (unverified)",
            license_url=(lic or {}).get("license_url"),
            license_verified=bool(lic),
            page_url=(lic or {}).get("page_url"),
            tags=tuple(sorted({path.parent.name.lower(), path.suffix.lower().lstrip(".")})),
            size_bytes=stat.st_size,
            raw={
                "path": str(path.resolve()),
                "filename": path.name,
                "mtime": stat.st_mtime,
                "tags": tags,
                "license_sidecar": lic,
            },
        )

    def search(self, query: SearchQuery) -> list[Track]:
        """Filter the crate by BPM and tags, closest tempo first.

        Untagged tracks are *kept* rather than filtered out — analysis will
        compute their tempo, and dropping them would make an untagged crate look
        empty. They sort last, since a known tempo match is always the safer bet.
        """
        wanted = {t.lower() for t in query.tags}
        hits = []
        for track in self.scan():
            if wanted:
                haystack = " ".join([track.title.lower(), track.artist.lower(), *track.tags])
                if not any(tag in haystack for tag in wanted):
                    continue
            if not within_size_limits(track):
                continue
            if track.bpm is None or query.bpm_min <= track.bpm <= query.bpm_max:
                hits.append(track)
        return rank_by_tempo(hits, query)[: query.limit]

    def fetch(self, track: Track, dest_dir: Path) -> Path:
        """Copy into the cache so the rest of the pipeline is source-agnostic."""
        src = Path(track.fetch_uri)
        if not src.exists():
            raise SourceError(f"local file has gone missing: {src}")
        size = src.stat().st_size
        if size > config.MAX_TRACK_BYTES:
            raise SourceError(
                f"{src.name} is {size} bytes, over the {config.MAX_TRACK_BYTES} byte cap"
            )
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"source{src.suffix.lower()}"
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            return dest
        shutil.copy2(src, dest)
        return dest

