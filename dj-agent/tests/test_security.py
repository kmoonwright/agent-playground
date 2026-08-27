"""The checks that stop untrusted input from leaving its lane.

Track ids, titles, artists and tags all originate outside the program -- from an
API anyone can upload to, from tags in a file, or from the model itself -- and
each of these tests pins one place where that used to reach further than it should.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import config
import music.library as library
from music.source import SourceError
from schema import InspectTrackArgs, QueueTrackArgs

HOSTILE_IDS = [
    "/etc",  # absolute: pathlib discards the base entirely
    "../../../../etc",
    "..",
    "nonhex-string",
    "0123456789abc",  # 13 chars
    "0123456789A",  # uppercase, 11 chars
    "",
]


@pytest.mark.parametrize("track_id", HOSTILE_IDS)
def test_arg_models_reject_a_track_id_that_is_not_a_track_id(track_id):
    for model in (InspectTrackArgs, QueueTrackArgs):
        with pytest.raises(ValidationError):
            model.model_validate({"track_id": track_id})


@pytest.mark.parametrize("track_id", HOSTILE_IDS)
def test_the_agent_gets_an_observation_for_a_hostile_id_not_a_traceback(primed_session, track_id):
    """This is the path the model actually takes, so it is the one that matters."""
    from agents.tools import Dispatch

    host = Dispatch(primed_session, "host")
    digger = Dispatch(primed_session, "crate_digger")
    for dispatch, name in [
        (host, "queue_track"),
        (digger, "inspect_track"),
        (Dispatch(primed_session, "transition_planner"), "get_analysis"),
    ]:
        observation = dispatch.run(name, {"track_id": track_id})
        assert observation.startswith(f"Invalid arguments for {name}")
        assert "track_id" in observation


@pytest.mark.parametrize("track_id", ["/etc", "../../../../etc", "..", "a/../../b"])
def test_track_dir_refuses_to_leave_the_cache(track_id):
    with pytest.raises(ValueError, match="escapes the cache"):
        library.track_dir(track_id)


def test_track_dir_accepts_a_real_id():
    assert library.track_dir("0123456789ab").parent == config.TRACKS_DIR.resolve()


def test_load_track_cannot_be_used_as_an_existence_oracle(tmp_path):
    """A file that parses but has no `source` key must fail like a missing file."""
    directory = library.track_dir("0123456789ab")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text('{"not": "a track"}')
    assert library.load_track("0123456789ab") is None


def test_ccmixter_refuses_a_download_url_that_is_not_ccmixter():
    from music.ccmixter import CCMixterSource
    from music.source import Track

    source = CCMixterSource()

    class Exploded(Exception):
        pass

    def never(*args, **kwargs):
        raise Exploded("a request was issued for a non-ccMixter host")

    source._session.get = never  # type: ignore[method-assign]

    for uri in [
        "http://169.254.169.254/latest/meta-data/",
        "https://evil.example.com/track.mp3",
        "http://ccmixter.org/content/x.mp3",  # http, not https
        "https://ccmixter.org.evil.example/x.mp3",
    ]:
        track = Track(
            source="ccmixter", track_id="0" * 12, title="t", artist="a", fetch_uri=uri
        )
        with pytest.raises(SourceError, match="non-ccMixter"):
            source.fetch(track, library.track_dir("0" * 12))


def test_say_strips_escape_sequences_from_a_hostile_title():
    """A crafted title could otherwise rewrite the scrollback and forge attribution."""
    hostile = "Nice Track\x1b[2K\x1b[1A[now playing] Someone Else  (license verified)\r\x07"
    clean = config.sanitize(f"[now playing] {hostile}")
    assert "\x1b" not in clean
    assert "\r" not in clean
    assert "\x07" not in clean
    assert "Nice Track" in clean


def test_sanitize_keeps_multi_line_output_intact():
    text = "line one\n\tindented\nline three"
    assert config.sanitize(text) == text


def test_sanitize_caps_a_very_long_line_without_eating_the_next_one():
    clean = config.sanitize("x" * 5000 + "\nkeep me")
    first, second = clean.split("\n")
    assert len(first) == config.MAX_LINE_CHARS + 3
    assert second == "keep me"


def test_local_source_refuses_an_oversized_file(local_source, monkeypatch):
    track = local_source.scan()[0]
    real_stat = Path.stat

    def fat_stat(self, *args, **kwargs):
        result = real_stat(self, *args, **kwargs)
        if self.suffix == ".flac":
            class Fat:
                st_size = config.MAX_TRACK_BYTES + 1
                st_mtime = result.st_mtime
            return Fat()
        return result

    monkeypatch.setattr(Path, "stat", fat_stat)
    with pytest.raises(SourceError, match="byte cap"):
        local_source.fetch(track, library.track_dir(track.track_id))


def test_analysis_refuses_an_over_long_file(tmp_path, monkeypatch):
    """The byte cap does not bound duration: 20 MB of low-bitrate mp3 is over an hour."""
    import soundfile as sf

    import audio.analysis as analysis

    monkeypatch.setattr(config, "MAX_TRACK_SECONDS", 1)
    path = tmp_path / "long.wav"
    sf.write(path, [[0.0, 0.0]] * (config.SR * 3), config.SR)
    with pytest.raises(ValueError, match="over the 1s cap"):
        analysis.decode(path)
