"""ccMixter: Creative Commons remixes with producer-authored BPM in the payload.

Two things about this API shape the whole module:

  * BPM is a real integer in `upload_extra.bpm`, and it is also indexed as
    5-BPM-wide `bpm_XXX_YYY` tags, so a tempo range becomes a set of band
    queries. That metadata BPM is what the mixer uses to set playback rate --
    it's producer-authored and usually exact, where a beat tracker's estimate
    is not.

  * Downloading a track requires BOTH a browser-like User-Agent AND
    `Referer: https://ccmixter.org/`. With only one of them the CDN returns a
    bare 403 with no explanation. This cost real debugging time; hence the
    comment rather than a silent constant.

robots.txt disallows /api/ as a courtesy signal, so every request goes through a
module-global rate limiter and every response is cached to disk.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

import requests

import config
from music.source import (
    SearchQuery,
    SourceError,
    Track,
    rank_by_tempo,
    track_id_for,
    within_size_limits,
)

API_URL = "https://ccmixter.org/api/query"
SITE_URL = "https://ccmixter.org/"

# `download_url` comes out of the API response verbatim, so it is untrusted input.
# Downloads are pinned to these hosts over https and redirects are refused, which
# is what stops a crafted (or hijacked) URL turning `fetch` into a request the
# user's machine makes to an address of someone else's choosing.
DOWNLOAD_HOST = "ccmixter.org"

# Both of these are required to download audio. See the module docstring.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
DOWNLOAD_HEADERS = {"User-Agent": USER_AGENT, "Referer": SITE_URL}
API_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}

BAND_WIDTH = 5
API_CACHE_TTL_S = 6 * 3600
MIN_REQUEST_INTERVAL_S = 1.5

# ccmixter.org's TLS handshake is genuinely slow -- measured at 21-27 seconds
# to complete, with TCP connect finishing in 0.16s. `requests` counts the
# handshake against the *connect* timeout, so a normal 10s connect timeout
# fails every single call. The Session is reused so that cost is paid once per
# process and subsequent band queries ride the same keep-alive connection.
CONNECT_TIMEOUT_S = 60.0
READ_TIMEOUT_S = 45.0
HANDSHAKE_RETRIES = 1

_rate_lock = threading.Lock()
_last_request_at = 0.0


def _throttle() -> None:
    """At least MIN_REQUEST_INTERVAL_S between *any* two ccmixter.org requests."""
    global _last_request_at
    with _rate_lock:
        wait = MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


def _check_download_host(uri: str) -> None:
    """Refuse to fetch anything the API points at that is not ccMixter over https."""
    parts = urlsplit(uri)
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or not (host == DOWNLOAD_HOST or host.endswith("." + DOWNLOAD_HOST)):
        raise SourceError(f"refusing to download from a non-ccMixter URL: {uri!r}")


def bpm_bands(bpm_min: float, bpm_max: float) -> list[str]:
    """The `bpm_XXX_YYY` tags covering a tempo range.

    Bands are 5 BPM wide and aligned to multiples of 5 (e.g. bpm_120_125), so a
    120-126 request needs two of them.
    """
    lo = int(bpm_min // BAND_WIDTH) * BAND_WIDTH
    hi = int(bpm_max // BAND_WIDTH) * BAND_WIDTH
    return [f"bpm_{b}_{b + BAND_WIDTH}" for b in range(lo, hi + 1, BAND_WIDTH)]


def _cache_path(url: str) -> Path:
    return config.API_CACHE_DIR / f"{hashlib.sha1(url.encode()).hexdigest()}.json"


def _cached(url: str) -> Any | None:
    path = _cache_path(url)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if time.time() - float(payload.get("fetched_at", 0)) > API_CACHE_TTL_S:
        return None
    return payload.get("body")


def _store(url: str, body: Any) -> None:
    config.API_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(url).write_text(
        json.dumps({"fetched_at": time.time(), "url": url, "body": body})
    )


def _parse_duration(ps: str | None) -> float | None:
    """ccMixter reports duration as "3:58" in file_format_info.ps."""
    if not ps:
        return None
    parts = str(ps).split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    seconds = 0.0
    for n in nums:
        seconds = seconds * 60 + n
    return seconds


class CCMixterSource:
    name = "ccmixter"

    def __init__(self, *, offline: bool = False) -> None:
        self.offline = offline
        self._session = requests.Session()
        self._session.headers.update(API_HEADERS)

    # ------------------------------------------------------------ http

    def _get_json(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        url = f"{API_URL}?{urlencode(params)}"
        cached = _cached(url)
        if cached is not None:
            return cached
        if self.offline:
            raise SourceError(f"offline: no cached response for {url}")

        _throttle()
        last_exc: Exception | None = None
        body = None
        for attempt in range(HANDSHAKE_RETRIES + 1):
            try:
                resp = self._session.get(url, timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S))
                resp.raise_for_status()
                body = resp.json()
                break
            except requests.Timeout as exc:
                last_exc = exc
                if attempt < HANDSHAKE_RETRIES:
                    _throttle()
                    continue
                raise SourceError(
                    f"ccMixter timed out after {attempt + 1} attempts ({exc}). "
                    "Their TLS handshake can take 20-30s; try again or use "
                    "--seed-cache once and then --offline."
                ) from exc
            except requests.RequestException as exc:
                raise SourceError(f"ccMixter API request failed: {exc}") from exc
            except ValueError as exc:
                raise SourceError(f"ccMixter returned non-JSON: {exc}") from exc
        if body is None:
            raise SourceError(f"ccMixter request produced no body: {last_exc}")

        if isinstance(body, dict) and body.get("code"):
            raise SourceError(f"ccMixter API error {body.get('code')}: {body.get('message')}")
        if not isinstance(body, list):
            raise SourceError(f"expected a JSON array from ccMixter, got {type(body).__name__}")

        _store(url, body)
        return body

    # ------------------------------------------------------------ parsing

    def _to_track(self, item: dict[str, Any]) -> Track | None:
        files = item.get("files") or []
        audio = next(
            (
                f
                for f in files
                if str((f.get("file_format_info") or {}).get("media-type", "")).lower() == "audio"
                and f.get("download_url")
            ),
            None,
        )
        if audio is None:
            return None

        page_url = item.get("file_page_url")
        if not page_url:
            return None

        extra = item.get("upload_extra") or {}
        info = audio.get("file_format_info") or {}
        bpm = extra.get("bpm")
        try:
            bpm_val = float(bpm) if bpm else None
        except (TypeError, ValueError):
            bpm_val = None

        size = audio.get("file_rawsize")
        try:
            size_val = int(size) if size is not None else None
        except (TypeError, ValueError):
            size_val = None

        tags = tuple(
            sorted({t for t in str(item.get("upload_tags") or "").split(",") if t.strip()})
        )

        return Track(
            source=self.name,
            track_id=track_id_for(str(page_url)),
            title=str(item.get("upload_name") or "untitled").strip(),
            artist=str(item.get("user_real_name") or item.get("user_name") or "unknown").strip(),
            fetch_uri=str(audio["download_url"]),
            bpm=bpm_val,
            bpm_source="metadata" if bpm_val else "unknown",
            duration_s=_parse_duration(info.get("ps")),
            license_name=item.get("license_name"),
            license_url=item.get("license_url"),
            # ccMixter states the license per upload, so it *is* verified.
            license_verified=bool(item.get("license_name")),
            page_url=str(page_url),
            tags=tags,
            size_bytes=size_val,
            raw=item,
        )

    # ------------------------------------------------------------ Source

    def search(self, query: SearchQuery) -> list[Track]:
        """One request per covering BPM band, merged and ranked by tempo distance.

        Extra tags are ANDed server-side first; if that comes back empty we retry
        the band alone and filter client-side, because ccMixter's user tags are
        free-form and an exact AND match misses a lot.
        """
        bands = bpm_bands(query.bpm_min, query.bpm_max)
        extra_tags = [t.strip().lower().replace(" ", "_") for t in query.tags if t.strip()]
        per_band = max(4, query.limit * 2)

        seen: dict[str, Track] = {}
        for band in bands:
            tag_spec = ",".join([band, *extra_tags]) if extra_tags else band
            # lic=open restricts to licenses that permit commercial use (CC BY / CC0).
            params = {"f": "json", "limit": per_band, "tags": tag_spec, "lic": "open"}

            items = self._get_json(params)
            if not items and extra_tags:
                items = self._get_json({**params, "tags": band})
                items = [
                    it
                    for it in items
                    if any(t in str(it.get("upload_tags", "")).lower() for t in extra_tags)
                ]

            for item in items:
                track = self._to_track(item)
                if track is None or not within_size_limits(track):
                    continue
                if track.bpm is not None and not (query.bpm_min <= track.bpm <= query.bpm_max):
                    continue
                seen.setdefault(track.track_id, track)

        return rank_by_tempo(list(seen.values()), query)[: query.limit]

    def fetch(self, track: Track, dest_dir: Path) -> Path:
        """Stream the mp3 into the cache. Needs UA *and* Referer or it 403s."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(track.fetch_uri).suffix.lower() or ".mp3"
        dest = dest_dir / f"source{suffix}"
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        if self.offline:
            raise SourceError(f"offline: {track.label} is not in the cache")
        _check_download_host(track.fetch_uri)

        _throttle()
        tmp = dest.with_suffix(dest.suffix + ".part")
        written = 0
        try:
            with self._session.get(
                track.fetch_uri,
                headers=DOWNLOAD_HEADERS,
                stream=True,
                allow_redirects=False,
                timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
            ) as resp:
                if resp.is_redirect:
                    raise SourceError(
                        f"ccMixter redirected the download for {track.label} to "
                        f"{resp.headers.get('Location', '?')!r}; refusing to follow it"
                    )
                if resp.status_code == 403:
                    raise SourceError(
                        "ccMixter returned 403. Both a browser User-Agent and "
                        "Referer: https://ccmixter.org/ are required to download audio."
                    )
                resp.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        written += len(chunk)
                        if written > config.MAX_TRACK_BYTES:
                            raise SourceError(
                                f"{track.label} exceeds the {config.MAX_TRACK_BYTES} byte cap"
                            )
                        fh.write(chunk)
        except requests.RequestException as exc:
            tmp.unlink(missing_ok=True)
            raise SourceError(f"download failed for {track.label}: {exc}") from exc
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

        tmp.replace(dest)
        return dest

