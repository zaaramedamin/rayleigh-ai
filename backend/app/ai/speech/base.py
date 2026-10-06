"""Speech recognition: turning a short recording of the owner's voice into text.

Like the other models, the recogniser runs entirely on this machine. A recording is held in
memory for the length of one request, is never written to disk and is never logged.
"""

import io
import wave
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from app.ai.embeddings.base import model_dir_for

Samples = NDArray[np.float32]

SAMPLE_RATE = 16000  # what the recogniser expects; the interface records at this rate
MIN_AUDIO_SECONDS = 0.2
MAX_AUDIO_SECONDS = 30  # one spoken order or question, not a dictation
# Quieter than this at its loudest point is silence, and is not sent to the model at all:
# speech models tend to "hear" words in silence.
SILENCE_PEAK = 0.01

# Present in every model folder saved by `download_speech_model`.
SPEECH_MODEL_MARKER = "config.json"


class SpeechModelNotAvailableError(RuntimeError):
    """The speech model has not been downloaded into MODELS_DIR."""


class SpeechRuntimeError(RuntimeError):
    """The speech library is installed but cannot be loaded or run on this machine."""


class AudioError(ValueError):
    """The recording cannot be used: wrong format, too short or too long."""


class SpeechRecognizer(Protocol):
    """Turns speech into text. Implementations must run entirely on this machine."""

    model_name: str

    def transcribe(self, samples: Samples, language: str | None = None) -> str:
        """The words spoken in `samples` (mono, 16 kHz, -1..1). Empty if nothing was said.

        `language` is a code such as "en"; None lets the model detect it.
        """
        ...


def is_speech_model_downloaded(models_dir: Path, model_name: str) -> bool:
    return (model_dir_for(models_dir, model_name) / SPEECH_MODEL_MARKER).is_file()


def is_silent(samples: Samples) -> bool:
    return samples.size == 0 or float(np.max(np.abs(samples))) < SILENCE_PEAK


def decode_wav(data: bytes) -> Samples:
    """Read a 16-bit PCM WAV recording at 16 kHz into mono samples. Raises AudioError."""
    try:
        with wave.open(io.BytesIO(data), "rb") as reader:
            channels = reader.getnchannels()
            width = reader.getsampwidth()
            rate = reader.getframerate()
            raw = reader.readframes(reader.getnframes())
    except (wave.Error, EOFError) as exc:
        raise AudioError("the recording is not a WAV file") from exc
    if width != 2 or channels not in (1, 2):
        raise AudioError("the recording must be 16-bit PCM, mono or stereo")
    if rate != SAMPLE_RATE:
        raise AudioError(f"the recording must be sampled at {SAMPLE_RATE} Hz")

    usable = len(raw) - len(raw) % (2 * channels)
    samples = np.frombuffer(raw[:usable], dtype="<i2").astype(np.float32) / 32768.0
    if channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1).astype(np.float32)
    seconds = samples.size / SAMPLE_RATE
    if seconds < MIN_AUDIO_SECONDS:
        raise AudioError("the recording is too short")
    if seconds > MAX_AUDIO_SECONDS:
        raise AudioError(f"the recording is longer than {MAX_AUDIO_SECONDS} seconds")
    return samples
