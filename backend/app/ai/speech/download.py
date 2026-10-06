import gc
import os
import tempfile
from pathlib import Path

from app.ai.embeddings.base import model_dir_for
from app.ai.speech.base import is_speech_model_downloaded


def download_speech_model(model_name: str, models_dir: Path) -> Path:
    """Download a Whisper speech model into MODELS_DIR and return its folder.

    Like the embedding model's download, this uses the network and runs only when the owner asks
    for it (`python -m app download-voice-model`). Everything else loads the saved copy from
    disk. Does nothing if the model is already there.
    """
    target = model_dir_for(models_dir, model_name)
    if is_speech_model_downloaded(models_dir, model_name):
        return target

    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    models_dir.mkdir(parents=True, exist_ok=True)
    # Download into a temporary cache inside MODELS_DIR, save a clean copy, then move it into
    # place, so an interrupted download never leaves a half-written model folder behind.
    with tempfile.TemporaryDirectory(
        dir=models_dir, prefix=".download-", ignore_cleanup_errors=True
    ) as cache:
        processor = WhisperProcessor.from_pretrained(
            model_name, cache_dir=cache, trust_remote_code=False
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name, cache_dir=cache, trust_remote_code=False
        )
        staged = Path(cache) / "saved"
        processor.save_pretrained(str(staged))
        model.save_pretrained(str(staged))
        # Release memory-mapped weight files so Windows can delete the temporary cache.
        del model, processor
        gc.collect()
        os.replace(staged, target)
    return target
