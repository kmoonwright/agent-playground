"""LangGraph pipeline: load_audio -> transcribe -> analyze -> respond.

Each node is a LangGraph node, so LangChainInstrumentor gives every node
(and every langchain chat-model call inside analyze/respond) its own span
for free. Only the raw Whisper call in `transcribe` needs a manual span --
LangChain has no wrapper for OpenAI's audio transcription endpoint.
"""

from __future__ import annotations

import io
import json
import mimetypes
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

import config
from instrumentation import LLM, TOOL, set_audio_attributes, set_output, traced_span, using_run_context


class AudioState(TypedDict, total=False):
    audio_path: str | None
    selftest: bool
    model_name: str
    duration_s: float
    sample_rate: int
    channels: int
    transcript: str
    stubbed: bool
    analysis: dict
    response: str


def _tone_array(duration_s: float = 2.0, sample_rate: int = 16000, freq: float = 440.0):
    t = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return tone, sample_rate


def _synthetic_tone() -> io.BytesIO:
    """A 440Hz sine wave, written to an in-memory WAV -- so --selftest has a
    real (if silly) audio object to report duration/sample-rate for, and to
    attach to the transcribe span, without needing a microphone."""
    tone, sample_rate = _tone_array()
    buffer = io.BytesIO()
    sf.write(buffer, tone, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer


def make_sample_file(path: str | Path = "samples/some.wav") -> Path:
    """Write the same synthetic tone to disk, so there's a real file on
    disk to point --file at -- see cli.py's --make-sample."""
    tone, sample_rate = _tone_array()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, tone, sample_rate)
    return path


def load_audio(state: AudioState) -> dict:
    selftest = state.get("selftest", False)
    audio_path = state.get("audio_path")
    with traced_span(
        "load_audio", TOOL, input_value=audio_path or "synthetic-tone", attributes={"selftest": selftest}
    ) as span:
        if selftest or not audio_path:
            info = sf.info(_synthetic_tone())
        else:
            info = sf.info(audio_path)
        result = {
            "duration_s": info.frames / info.samplerate,
            "sample_rate": info.samplerate,
            "channels": info.channels,
        }
        set_output(span, result)
    return result


def transcribe(state: AudioState) -> dict:
    selftest = state.get("selftest", False)
    audio_path = state.get("audio_path")
    stubbed = selftest or not config.openai_api_key()

    # What we attach depends on whether a real file was given, not on
    # whether the API call itself is stubbed -- a real --file with no
    # OPENAI_API_KEY should still attach the file the user actually passed,
    # not an unrelated synthetic tone.
    if audio_path:
        audio_bytes = Path(audio_path).read_bytes()
        mime_type = mimetypes.guess_type(audio_path)[0] or "audio/wav"
        upload_name = Path(audio_path).name
    else:
        audio_bytes = _synthetic_tone().getvalue()
        mime_type = "audio/wav"
        upload_name = "selftest-tone.wav"

    with traced_span(
        "transcribe",
        LLM,
        input_value=audio_path or "synthetic-tone",
        attributes={"llm.model_name": config.whisper_model(), "stubbed": stubbed},
    ) as span:
        if stubbed:
            transcript = "This is a stubbed transcript, generated for --selftest or a missing OPENAI_API_KEY."
        else:
            from openai import OpenAI

            client = OpenAI(api_key=config.openai_api_key())
            upload = io.BytesIO(audio_bytes)
            upload.name = upload_name
            result = client.audio.transcriptions.create(model=config.whisper_model(), file=upload)
            transcript = result.text

        set_audio_attributes(span, data=audio_bytes, mime_type=mime_type, transcript=transcript)
        set_output(span, transcript)
    return {"transcript": transcript, "stubbed": stubbed}


def analyze(state: AudioState) -> dict:
    transcript = state["transcript"]
    if state.get("stubbed"):
        with traced_span("analyze_stub", LLM, input_value=transcript, attributes={"llm.model_name": "mock"}) as span:
            analysis = {"sentiment": "neutral", "topic": "unknown"}
            set_output(span, analysis)
        return {"analysis": analysis}

    model = config.build_chat_model(state["model_name"])
    prompt = (
        "Read this transcript and reply with strict JSON only, no prose: "
        '{"sentiment": "<positive|neutral|negative>", "topic": "<a few words>"}.\n\n'
        f"Transcript:\n{transcript}"
    )
    reply = model.invoke(prompt)
    try:
        analysis = json.loads(reply.content)
    except (json.JSONDecodeError, TypeError):
        analysis = {"sentiment": "unknown", "topic": "unknown", "raw": reply.content}
    return {"analysis": analysis}


def respond(state: AudioState) -> dict:
    transcript = state["transcript"]
    analysis = state["analysis"]
    if state.get("stubbed"):
        with traced_span("respond_stub", LLM, input_value=transcript, attributes={"llm.model_name": "mock"}) as span:
            response = f"Stubbed response: {transcript}"
            set_output(span, response)
        return {"response": response}

    model = config.build_chat_model(state["model_name"])
    prompt = (
        f"A user said this (sentiment={analysis.get('sentiment')}, topic={analysis.get('topic')}):\n"
        f"{transcript}\n\nReply to them directly, in one short paragraph."
    )
    reply = model.invoke(prompt)
    return {"response": reply.content}


def build_graph():
    graph = StateGraph(AudioState)
    graph.add_node("load_audio", load_audio)
    graph.add_node("transcribe", transcribe)
    graph.add_node("analyze", analyze)
    graph.add_node("respond", respond)
    graph.set_entry_point("load_audio")
    graph.add_edge("load_audio", "transcribe")
    graph.add_edge("transcribe", "analyze")
    graph.add_edge("analyze", "respond")
    graph.add_edge("respond", END)
    return graph.compile()


def run(audio_path: str | None, *, selftest: bool = False, model_name: str | None = None) -> AudioState:
    model_name = config.resolve_chat_model_name(model_name)
    run_id = str(uuid.uuid4())
    graph = build_graph()
    initial: AudioState = {"audio_path": audio_path, "selftest": selftest, "model_name": model_name}
    with using_run_context(run_id, {"model": model_name, "selftest": selftest}):
        return graph.invoke(initial)
