"""Sources: id stability, BPM band mapping, local tag parsing, HTTP caching."""

from __future__ import annotations

import json

import pytest

import config
from music.source import SearchQuery, Source, SourceError, track_id_for, within_size_limits
from music.ccmixter import CCMixterSource, bpm_bands
from music.local import LocalSource
from music.registry import build_sources, parse_names


def test_track_id_is_stable_and_source_independent():
    """The id is a hash of the stable URI, so the cache survives restarts."""
    assert track_id_for("https://ccmixter.org/files/x/1") == track_id_for(
        "https://ccmixter.org/files/x/1"
    )
    assert track_id_for("/a/b.mp3") != track_id_for("/a/c.mp3")


@pytest.mark.parametrize(
    ("lo", "hi", "expected"),
    [
        (120, 124, ["bpm_120_125"]),
        (120, 126, ["bpm_120_125", "bpm_125_130"]),
        (118, 126, ["bpm_115_120", "bpm_120_125", "bpm_125_130"]),
        (122, 122, ["bpm_120_125"]),
        (100, 105, ["bpm_100_105", "bpm_105_110"]),
    ],
)
def test_bpm_bands_cover_the_range(lo, hi, expected):
    assert bpm_bands(lo, hi) == expected



def test_local_source_reads_bpm_tag_and_keeps_untagged_files(local_source):
    tracks = {t.title: t for t in local_source.scan()}
    assert tracks["tagged_120"].bpm == 120.0
    assert tracks["tagged_120"].bpm_source == "metadata"
    # An untagged file must NOT be dropped -- analysis computes its tempo later,
    # and dropping it would make an untagged crate look empty.
    assert tracks["untagged"].bpm is None
    assert tracks["untagged"].bpm_source == "unknown"


def test_local_search_keeps_untagged_candidates_last(local_source):
    found = local_source.search(SearchQuery(bpm_min=118, bpm_max=122, limit=8))
    assert [t.bpm for t in found] == [120.0, None]


def test_local_licenses_are_unverified_without_a_sidecar(local_source):
    for track in local_source.scan():
        assert track.license_verified is False
        assert "unverified" in (track.license_name or "")


def test_local_license_sidecar_marks_tracks_verified(crate):
    (crate / "LICENSE.json").write_text(
        json.dumps(
            {
                "tagged_120.flac": {
                    "license_name": "CC0 1.0",
                    "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                }
            }
        )
    )
    tracks = {t.title: t for t in LocalSource(crate).scan()}
    assert tracks["tagged_120"].license_verified is True
    assert tracks["tagged_120"].license_name == "CC0 1.0"
    # A file absent from the sidecar stays unverified.
    assert tracks["untagged"].license_verified is False


def test_local_source_rejects_a_broken_sidecar(crate):
    (crate / "LICENSE.json").write_text("{not json")
    with pytest.raises(SourceError):
        LocalSource(crate)


def test_ccmixter_search_serves_from_the_disk_cache_without_http(monkeypatch):
    """A cached response must satisfy a search with zero network calls."""
    source = CCMixterSource()
    payload = [
        {
            "upload_name": "Cached Track",
            "user_real_name": "Someone",
            "file_page_url": "https://ccmixter.org/files/someone/1",
            "license_name": "Attribution (3.0)",
            "license_url": "http://creativecommons.org/licenses/by/3.0/",
            "upload_extra": {"bpm": 122},
            "upload_tags": ",house,vocals,",
            "files": [
                {
                    "download_url": "https://ccmixter.org/content/someone/x.mp3",
                    "file_rawsize": 4_000_000,
                    "file_format_info": {"media-type": "audio", "ps": "3:20"},
                }
            ],
        }
    ]

    from urllib.parse import urlencode

    import music.ccmixter as mod

    url = f"{mod.API_URL}?{urlencode({'f': 'json', 'limit': 8, 'tags': 'bpm_120_125', 'lic': 'open'})}"
    mod._store(url, payload)

    def explode(*args, **kwargs):
        raise AssertionError("the cache should have satisfied this request")

    monkeypatch.setattr(source._session, "get", explode)

    found = source.search(SearchQuery(bpm_min=120, bpm_max=124, limit=4))
    assert len(found) == 1
    track = found[0]
    assert track.bpm == 122.0
    assert track.bpm_source == "metadata"
    assert track.duration_s == 200.0
    assert track.license_verified is True
    assert track.raw["upload_name"] == "Cached Track"


def test_ccmixter_offline_refuses_rather_than_dialling_out():
    source = CCMixterSource(offline=True)
    with pytest.raises(SourceError, match="offline"):
        source.search(SearchQuery(bpm_min=120, bpm_max=124))


def test_ccmixter_skips_oversized_files(monkeypatch):
    source = CCMixterSource()
    item = {
        "upload_name": "Huge",
        "user_name": "x",
        "file_page_url": "https://ccmixter.org/files/x/9",
        "license_name": "Attribution",
        "upload_extra": {"bpm": 122},
        "files": [
            {
                "download_url": "https://ccmixter.org/content/x/huge.mp3",
                "file_rawsize": config.MAX_TRACK_BYTES + 1,
                "file_format_info": {"media-type": "audio", "ps": "3:20"},
            }
        ],
    }
    track = source._to_track(item)
    assert track is not None
    assert within_size_limits(track) is False


def test_registry_validates_names_and_honours_offline(crate):
    assert parse_names("local, ccmixter") == ("local", "ccmixter")
    with pytest.raises(SourceError, match="unknown source"):
        parse_names("spotify")

    built = build_sources("local", crate_dir=crate, offline=True)
    assert set(built) == {"local"}
    assert isinstance(built["local"], Source)


def test_offline_keeps_ccmixter_enabled_in_cache_only_mode(crate):
    """--seed-cache once, then --offline forever: the source serves from disk."""
    built = build_sources("ccmixter,local", crate_dir=crate, offline=True)
    assert set(built) == {"ccmixter", "local"}
    assert built["ccmixter"].offline is True
    # But it refuses to reach the network.
    from music.source import SearchQuery

    with pytest.raises(SourceError, match="offline"):
        built["ccmixter"].search(SearchQuery(bpm_min=120, bpm_max=124))


def test_cache_search_only_offers_tracks_from_enabled_sources(fixture_wav):
    """The cache outlives a run; offering a disabled source's track breaks prime."""
    import music.library as library
    from music.source import Track, track_id_for

    for source in ("local", "ccmixter"):
        track = Track(
            source=source,
            track_id=track_id_for(f"{source}://x"),
            title=f"{source} track",
            artist="a",
            fetch_uri=str(fixture_wav),
            bpm=120.0,
            bpm_source="metadata",
        )
        library.save_track(track)

    both = library.search_cache(115, 125, limit=5)
    assert {t.source for t in both} == {"local", "ccmixter"}

    only_ccmixter = library.search_cache(115, 125, limit=5, sources=["ccmixter"])
    assert {t.source for t in only_ccmixter} == {"ccmixter"}

    assert library.search_cache(115, 125, limit=5, sources=["local"])[0].source == "local"


def test_find_candidates_respects_enabled_sources(session, fixture_wav):
    import music.library as library
    from music.source import Track, track_id_for

    stale = Track(
        source="ccmixter",
        track_id=track_id_for("ccmixter://stale"),
        title="stale",
        artist="a",
        fetch_uri="https://example.invalid/x.mp3",
        bpm=122.0,
        bpm_source="metadata",
    )
    library.save_track(stale)
    # The session only has `local` enabled, so the cached ccmixter track must not
    # be offered as an opening track.
    candidates = session.find_candidates(115, 130)
    assert all(t.source == "local" for t in candidates)
