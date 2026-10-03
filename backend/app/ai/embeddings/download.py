import gc
import os
import tempfile
from pathlib import Path

from app.ai.embeddings.base import is_model_downloaded, model_dir_for
from app.ai.embeddings.compat import import_sentence_transformer


def download_model(model_name: str, models_dir: Path) -> Path:
    """Download a sentence-transformers model into MODELS_DIR and return its folder.

    This is the only function in the application that uses the network. It runs only when
    the user asks for it (`python -m app download-model`); everything else loads the saved
    copy from disk. Does nothing if the model is already there.
    """
    target = model_dir_for(models_dir, model_name)
    if is_model_downloaded(models_dir, model_name):
        return target

    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    SentenceTransformer = import_sentence_transformer()

    models_dir.mkdir(parents=True, exist_ok=True)
    # Download into a temporary cache inside MODELS_DIR, save a clean copy, then move it into
    # place, so an interrupted download never leaves a half-written model folder behind.
    with tempfile.TemporaryDirectory(
        dir=models_dir, prefix=".download-", ignore_cleanup_errors=True
    ) as cache:
        model = SentenceTransformer(
            model_name, cache_folder=cache, device="cpu", trust_remote_code=False
        )
        staged = Path(cache) / "saved"
        model.save(str(staged), create_model_card=False)
        # Release memory-mapped weight files so Windows can delete the temporary cache.
        del model
        gc.collect()
        os.replace(staged, target)
    return target
