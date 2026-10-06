import logging
import os
import threading
from pathlib import Path
from typing import Any

from app.ai.embeddings.base import model_dir_for
from app.ai.speech.base import (
    SAMPLE_RATE,
    Samples,
    SpeechModelNotAvailableError,
    SpeechRuntimeError,
    is_silent,
    is_speech_model_downloaded,
)

# Given noise instead of speech, the model still writes a fluent sentence, but with little
# confidence in each word. Below this average the text is dropped. Measured on this model:
# clear orders score about -0.1, noise about -1.6. It is also Whisper's own default cut-off.
MIN_AVERAGE_LOG_PROBABILITY = -1.0
# These repeat the same remark about the model's saved settings on every recording.
_NOISY_LOGGERS = (
    "transformers.generation.utils",
    "transformers.generation.configuration_utils",
    "transformers.tokenization_utils_tokenizers",
)


def _force_offline() -> None:
    """Make sure the Hugging Face libraries never reach the network from this process."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


class WhisperRecognizer:
    """Local speech recognition with a Whisper model stored under MODELS_DIR.

    Uses the `transformers` library that the embedding model already depends on, so it adds no
    package of its own.
    """

    def __init__(self, model_name: str, models_dir: Path) -> None:
        if not is_speech_model_downloaded(models_dir, model_name):
            raise SpeechModelNotAvailableError(
                f"speech model {model_name!r} is not downloaded; "
                "run `python -m app download-voice-model`"
            )
        _force_offline()
        # Deferred: importing torch takes seconds and is only needed once a model is used.
        try:
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except ImportError as exc:
            hint = ""
            if "Application Control" in str(exc):
                hint = " Windows (Smart App Control) blocked a library file."
            raise SpeechRuntimeError(
                f"the speech library could not be loaded: {str(exc).rstrip('.')}.{hint}"
            ) from exc

        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.ERROR)
        folder = str(model_dir_for(models_dir, model_name))
        try:
            # trust_remote_code off: code shipped with a model is never run.
            self._processor: Any = WhisperProcessor.from_pretrained(
                folder, local_files_only=True, trust_remote_code=False
            )
            self._model: Any = WhisperForConditionalGeneration.from_pretrained(
                folder, local_files_only=True, trust_remote_code=False
            )
        except (OSError, ValueError) as exc:
            raise SpeechRuntimeError(
                f"the speech model in {folder} could not be loaded ({type(exc).__name__}); "
                "delete that folder and run `python -m app download-voice-model` again"
            ) from exc
        self._torch: Any = torch
        self._lock = threading.Lock()  # one recording at a time
        self.model_name = model_name

    def transcribe(self, samples: Samples, language: str | None = None) -> str:
        if is_silent(samples):
            return ""
        inputs = self._processor(
            samples, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True
        )
        try:
            with self._lock, self._torch.inference_mode():
                output = self._model.generate(
                    inputs.input_features,
                    attention_mask=inputs.attention_mask,
                    task="transcribe",
                    language=language,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                scores = self._model.compute_transition_scores(
                    output.sequences, output.scores, normalize_logits=True
                )
        except (RuntimeError, ValueError) as exc:
            raise SpeechRuntimeError(
                f"the speech model failed on this recording ({type(exc).__name__})"
            ) from exc
        if scores.numel() == 0 or float(scores.mean()) < MIN_AVERAGE_LOG_PROBABILITY:
            return ""  # the model was guessing: noise, not words
        text = self._processor.batch_decode(output.sequences, skip_special_tokens=True)[0]
        return " ".join(str(text).split())
