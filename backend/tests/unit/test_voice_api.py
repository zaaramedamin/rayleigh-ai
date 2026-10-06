import io
import logging
import math
import struct
import wave
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.ai.speech.base import (
    MAX_AUDIO_SECONDS,
    SAMPLE_RATE,
    SPEECH_MODEL_MARKER,
    AudioError,
    SpeechModelNotAvailableError,
    SpeechRuntimeError,
    decode_wav,
    is_silent,
    is_speech_model_downloaded,
)
from app.ai.speech.whisper import WhisperRecognizer
from app.api.deps import get_recognizer
from app.api.v1 import voice
from app.api.v1.voice import MAX_AUDIO_BYTES
from app.core.config import Settings, get_settings
from app.main import app
from tests.fakes import FakeRecognizer

SPEECH_FOLDER = "openai__whisper-base"


def make_wav(
    seconds: float = 1.0,
    *,
    rate: int = SAMPLE_RATE,
    channels: int = 1,
    width: int = 2,
    amplitude: float = 0.5,
) -> bytes:
    """A WAV recording of a 440 Hz tone."""
    frames = int(seconds * rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        for index in range(frames):
            value = amplitude * math.sin(2 * math.pi * 440 * index / rate)
            sample = (
                struct.pack("<h", int(value * 32767))
                if width == 2
                else bytes([int((value + 1) * 127.5)])
            )
            writer.writeframesraw(sample * channels)
    return buffer.getvalue()


# --- reading a recording ----------------------------------------------------------------------


def test_a_recording_becomes_mono_samples_between_minus_one_and_one() -> None:
    samples = decode_wav(make_wav(0.5))

    assert samples.dtype == np.float32
    assert samples.size == SAMPLE_RATE // 2
    assert 0.45 < float(np.max(np.abs(samples))) <= 0.5


def test_a_stereo_recording_is_mixed_down() -> None:
    assert decode_wav(make_wav(0.5, channels=2)).size == SAMPLE_RATE // 2


@pytest.mark.parametrize(
    ("data", "problem"),
    [
        (b"", "not a WAV file"),
        (b"this is not audio at all", "not a WAV file"),
        (b"RIFF\x00\x00\x00\x00WAVE", "not a WAV file"),
        (make_wav(0.5, rate=44100), "16000 Hz"),
        (make_wav(0.5, width=1), "16-bit"),
        (make_wav(0.05), "too short"),
        (make_wav(MAX_AUDIO_SECONDS + 1), "longer than"),
    ],
    ids=["empty", "text", "header-only", "44100-hz", "8-bit", "too-short", "too-long"],
)
def test_unusable_recordings_are_refused_with_a_reason(data: bytes, problem: str) -> None:
    with pytest.raises(AudioError, match=problem):
        decode_wav(data)


def test_silence_is_recognised_as_silence() -> None:
    assert is_silent(decode_wav(make_wav(0.5, amplitude=0.0))) is True
    assert is_silent(decode_wav(make_wav(0.5, amplitude=0.001))) is True
    assert is_silent(decode_wav(make_wav(0.5, amplitude=0.2))) is False


def test_the_speech_model_is_found_by_its_marker_file(models_dir: Path) -> None:
    assert is_speech_model_downloaded(models_dir, "openai/whisper-base") is False

    (models_dir / SPEECH_FOLDER).mkdir(parents=True)
    (models_dir / SPEECH_FOLDER / SPEECH_MODEL_MARKER).write_text("{}")

    assert is_speech_model_downloaded(models_dir, "openai/whisper-base") is True


def test_a_model_that_is_not_downloaded_is_never_fetched(models_dir: Path) -> None:
    with pytest.raises(SpeechModelNotAvailableError, match="download-voice-model"):
        WhisperRecognizer("openai/whisper-base", models_dir)


# --- the API ----------------------------------------------------------------------------------


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app.dependency_overrides[get_settings] = lambda: settings
    voice._loaded.clear()
    yield TestClient(app)
    app.dependency_overrides.clear()
    voice._loaded.clear()


@pytest.fixture
def recognizer(client: TestClient) -> FakeRecognizer:
    fake = FakeRecognizer("open the settings page")
    app.dependency_overrides[get_recognizer] = lambda: fake
    return fake


def test_a_recording_is_turned_into_text(client: TestClient, recognizer: FakeRecognizer) -> None:
    response = client.post(
        "/api/v1/voice/transcribe", content=make_wav(1.5), headers={"Content-Type": "audio/wav"}
    )

    assert response.status_code == 200
    assert response.json() == {"text": "open the settings page", "seconds": 1.5}
    assert recognizer.heard == [(int(SAMPLE_RATE * 1.5), None)]


def test_the_expected_language_is_passed_to_the_model(
    client: TestClient, recognizer: FakeRecognizer, make_settings: Callable[..., Settings]
) -> None:
    app.dependency_overrides[get_settings] = lambda: make_settings(speech_language="fr")

    client.post(
        "/api/v1/voice/transcribe", content=make_wav(), headers={"Content-Type": "audio/wav"}
    )

    assert recognizer.heard[0][1] == "fr"


@pytest.mark.parametrize(
    "data", [b"", b"not audio", make_wav(0.5, rate=8000)], ids=["empty", "text", "8000-hz"]
)
def test_an_unusable_recording_is_rejected_before_the_model_runs(
    client: TestClient, recognizer: FakeRecognizer, data: bytes
) -> None:
    response = client.post(
        "/api/v1/voice/transcribe", content=data, headers={"Content-Type": "audio/wav"}
    )

    assert response.status_code == 422
    assert recognizer.heard == []


def test_an_oversized_upload_is_refused(client: TestClient, recognizer: FakeRecognizer) -> None:
    response = client.post(
        "/api/v1/voice/transcribe",
        content=b"x" * (MAX_AUDIO_BYTES + 1),
        headers={"Content-Type": "audio/wav"},
    )

    assert response.status_code == 413
    assert recognizer.heard == []


def test_without_the_speech_model_the_answer_says_how_to_get_it(client: TestClient) -> None:
    response = client.post(
        "/api/v1/voice/transcribe", content=make_wav(), headers={"Content-Type": "audio/wav"}
    )

    assert response.status_code == 503
    assert "download-voice-model" in response.json()["detail"]


def test_a_failing_speech_model_is_a_clear_error(
    client: TestClient, recognizer: FakeRecognizer
) -> None:
    recognizer.error = SpeechRuntimeError("the speech model failed on this recording")

    response = client.post(
        "/api/v1/voice/transcribe", content=make_wav(), headers={"Content-Type": "audio/wav"}
    )

    assert response.status_code == 503
    assert "speech model failed" in response.json()["detail"]


def test_what_was_said_is_never_logged(
    client: TestClient, recognizer: FakeRecognizer, caplog: pytest.LogCaptureFixture
) -> None:
    recognizer.text = "my password is marmalade"

    with caplog.at_level(logging.DEBUG):
        client.post(
            "/api/v1/voice/transcribe", content=make_wav(), headers={"Content-Type": "audio/wav"}
        )

    assert "transcribed audio_seconds=1.0" in caplog.text
    assert "marmalade" not in caplog.text


def test_the_status_says_when_the_model_is_missing(client: TestClient) -> None:
    body = client.get("/api/v1/voice/status").json()

    assert body["model"] == "openai/whisper-base"
    assert body["state"] == "not_downloaded"
    assert body["language"] is None
    assert "download-voice-model" in body["hint"]


def test_the_status_follows_the_model_from_disk_to_memory(
    client: TestClient, recognizer: FakeRecognizer, settings: Settings
) -> None:
    (settings.models_dir / SPEECH_FOLDER).mkdir(parents=True)
    (settings.models_dir / SPEECH_FOLDER / SPEECH_MODEL_MARKER).write_text("{}")

    assert client.get("/api/v1/voice/status").json()["state"] == "not_loaded"

    client.post(
        "/api/v1/voice/transcribe", content=make_wav(), headers={"Content-Type": "audio/wav"}
    )

    status = client.get("/api/v1/voice/status").json()
    assert (status["state"], status["hint"]) == ("ready", None)


def test_preparing_loads_the_model_in_the_background(
    client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    (settings.models_dir / SPEECH_FOLDER).mkdir(parents=True)
    (settings.models_dir / SPEECH_FOLDER / SPEECH_MODEL_MARKER).write_text("{}")
    loaded: list[str] = []
    monkeypatch.setattr(voice, "load_recognizer", lambda name, _dir: loaded.append(name))

    assert client.post("/api/v1/voice/prepare").json()["state"] == "loading"

    for thread in voice.threading.enumerate():
        if thread is not voice.threading.current_thread() and thread.daemon:
            thread.join(timeout=5)
    assert loaded == ["openai/whisper-base"]
    assert client.post("/api/v1/voice/prepare").json()["state"] == "ready"
    assert loaded == ["openai/whisper-base"]  # not loaded twice


def test_preparing_does_nothing_without_the_model(client: TestClient) -> None:
    assert client.post("/api/v1/voice/prepare").json()["state"] == "not_downloaded"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/v1/voice/status"),
        ("POST", "/api/v1/voice/prepare"),
        ("POST", "/api/v1/voice/transcribe"),
    ],
)
def test_the_voice_routes_need_the_access_password(
    client: TestClient, make_settings: Callable[..., Settings], method: str, path: str
) -> None:
    from app.api.access import require_access

    app.dependency_overrides.pop(require_access, None)  # the real check, as in production
    app.dependency_overrides[get_settings] = lambda: make_settings(access_required=True)

    assert client.request(method, path).status_code == 401
