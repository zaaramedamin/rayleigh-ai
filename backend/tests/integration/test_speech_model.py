"""Tests against the real speech model. Skipped until `python -m app download-voice-model` ran."""

import numpy as np
import pytest

from app.ai.speech.base import SAMPLE_RATE, SpeechRuntimeError, is_speech_model_downloaded
from app.ai.speech.whisper import WhisperRecognizer
from app.core.config import get_settings
from app.evaluation.network_guard import NetworkGuard

SETTINGS = get_settings()

pytestmark = pytest.mark.skipif(
    not is_speech_model_downloaded(SETTINGS.models_dir, SETTINGS.speech_model),
    reason="speech model not downloaded (run `python -m app download-voice-model`)",
)


@pytest.fixture(scope="module")
def recognizer() -> WhisperRecognizer:
    try:
        return WhisperRecognizer(SETTINGS.speech_model, SETTINGS.models_dir)
    except SpeechRuntimeError as exc:
        # A library file blocked by Windows Smart App Control is this machine's problem, not a bug.
        if "Smart App Control" in str(exc):
            pytest.skip(str(exc))
        raise


def test_silence_is_nothing(recognizer: WhisperRecognizer) -> None:
    assert recognizer.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32)) == ""


@pytest.mark.parametrize("loudness", [0.02, 0.1])
def test_noise_is_not_turned_into_a_made_up_sentence(
    recognizer: WhisperRecognizer, loudness: float
) -> None:
    noise = np.random.default_rng(0).standard_normal(2 * SAMPLE_RATE) * loudness

    assert recognizer.transcribe(noise.astype(np.float32), "en") == ""


def test_listening_never_uses_the_network(recognizer: WhisperRecognizer) -> None:
    tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(SAMPLE_RATE) / SAMPLE_RATE)

    with NetworkGuard() as guard:
        recognizer.transcribe(tone.astype(np.float32))

    assert guard.blocked == []
