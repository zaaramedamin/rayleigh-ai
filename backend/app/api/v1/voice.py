import logging
import threading
import time
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.ai.speech.base import (
    MAX_AUDIO_SECONDS,
    SAMPLE_RATE,
    AudioError,
    SpeechModelNotAvailableError,
    SpeechRuntimeError,
    decode_wav,
    is_speech_model_downloaded,
)
from app.api.deps import RecognizerDep, SettingsDep
from app.core.config import Settings
from app.knowledge.components import load_recognizer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])

# The longest recording, as 16-bit stereo, plus room for the WAV header.
MAX_AUDIO_BYTES = SAMPLE_RATE * 2 * 2 * MAX_AUDIO_SECONDS + 4096

_loading = threading.Lock()
_loaded: set[tuple[str, str]] = set()  # (model, folder) pairs whose model is in memory


class VoiceStatus(BaseModel):
    model: str
    state: Literal["ready", "loading", "not_loaded", "not_downloaded"]
    language: str | None = Field(description="The language expected, or null to detect it.")
    hint: str | None = Field(default=None, description="What to do about it, if anything.")


class Transcription(BaseModel):
    text: str = Field(description="What was said. Empty if nothing was heard.")
    seconds: float = Field(description="Length of the recording.")


def _key(settings: Settings) -> tuple[str, str]:
    return settings.speech_model, str(settings.models_dir)


def _status(settings: Settings) -> VoiceStatus:
    language = settings.speech_language or None
    if not is_speech_model_downloaded(settings.models_dir, settings.speech_model):
        return VoiceStatus(
            model=settings.speech_model,
            state="not_downloaded",
            language=language,
            hint="Run `python -m app download-voice-model` once (needs internet).",
        )
    if _key(settings) in _loaded:
        state: Literal["ready", "loading", "not_loaded"] = "ready"
    else:
        state = "loading" if _loading.locked() else "not_loaded"
    return VoiceStatus(model=settings.speech_model, state=state, language=language)


def _load(settings: Settings) -> None:
    with _loading:
        try:
            load_recognizer(settings.speech_model, settings.models_dir)
            _loaded.add(_key(settings))
        except (SpeechModelNotAvailableError, SpeechRuntimeError):
            logger.warning("the speech model could not be loaded")


@router.get("/status")
def voice_status(settings: SettingsDep) -> VoiceStatus:
    """Whether voice orders can be understood on this machine. Model names only."""
    return _status(settings)


@router.post("/prepare")
def prepare_voice(settings: SettingsDep) -> VoiceStatus:
    """Start loading the speech model in the background, so the first order is not slow."""
    current = _status(settings)
    if current.state == "not_loaded":
        threading.Thread(target=_load, args=(settings,), daemon=True).start()
        current.state = "loading"
    return current


async def _read_body(request: Request) -> bytes:
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_AUDIO_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "The recording is too large.")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_AUDIO_BYTES:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "The recording is too large.")
    return bytes(body)


@router.post("/transcribe")
async def transcribe(
    request: Request, recognizer: RecognizerDep, settings: SettingsDep
) -> Transcription:
    """Turn a short recording (the request body: 16-bit PCM WAV, 16 kHz) into text, locally.

    The recording is used for this request only: it is not stored and not logged.
    """
    try:
        samples = decode_wav(await _read_body(request))
    except AudioError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    started = time.monotonic()
    try:
        text = await run_in_threadpool(
            recognizer.transcribe, samples, settings.speech_language or None
        )
    except SpeechRuntimeError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    _loaded.add(_key(settings))
    seconds = samples.size / SAMPLE_RATE
    # Timing and sizes only: what was said is never logged.
    logger.info(
        "transcribed audio_seconds=%.1f seconds=%.1f chars=%d",
        seconds,
        time.monotonic() - started,
        len(text),
    )
    return Transcription(text=text, seconds=round(seconds, 2))
