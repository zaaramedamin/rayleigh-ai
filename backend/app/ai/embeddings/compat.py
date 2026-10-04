"""Import sentence-transformers even when scikit-learn cannot be loaded.

sentence-transformers imports scikit-learn at start-up only for similarity and evaluation helpers
(`pairwise_distances`, `roc_curve`, ...). Reyleight never calls them: it needs only the model's
`encode`, and does its own cosine search in Qdrant.

On some Windows machines Smart App Control blocks scikit-learn's compiled files ("An Application
Control policy has blocked this file"), which would make the whole embedding library unusable.
When, and only when, scikit-learn really cannot be imported, we register a placeholder for the
three scikit-learn modules involved. We never try to load the blocked file or change any security
setting. Calling any scikit-learn function through the placeholder raises a clear error.
"""

import importlib
import importlib.machinery
import logging
import sys
import types
from typing import Any

logger = logging.getLogger(__name__)

_STUBBED_MODULES = ("sklearn", "sklearn.metrics", "sklearn.metrics.pairwise")


class _MissingScikitLearn(types.ModuleType):
    """Placeholder module: any attribute is a function that fails when called."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        module_name = self.__name__

        def unavailable(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError(
                f"{module_name}.{name} is not available: scikit-learn could not be loaded "
                "on this machine"
            )

        return unavailable


def _purge(prefixes: tuple[str, ...]) -> None:
    for name in [m for m in sys.modules if m.split(".")[0] in prefixes]:
        del sys.modules[name]


def install_scikit_learn_placeholder_if_blocked() -> bool:
    """Register the placeholder if scikit-learn cannot be imported. Returns True if it was used."""
    try:
        importlib.import_module("sklearn.metrics")
    except ImportError as exc:
        # A failed import can leave half-loaded modules behind; clear them before retrying.
        _purge(("sklearn", "sentence_transformers"))
        for name in _STUBBED_MODULES:
            module = _MissingScikitLearn(name)
            module.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
            module.__path__ = []
            sys.modules[name] = module
        logger.warning(
            "scikit-learn could not be loaded (%s); continuing without it. "
            "Embeddings are unaffected.",
            str(exc).splitlines()[0][:160],
        )
        return True
    return False


def _silence_progress_bars() -> None:
    """Command output should be the answer, not a "Loading weights" bar on every run."""
    try:
        from transformers.utils import logging as transformers_logging

        # The transformers helper has no type annotations; the call is a plain no-argument function.
        transformers_logging.disable_progress_bar()  # type: ignore[no-untyped-call]
    except ImportError:
        pass


def import_sentence_transformer() -> Any:
    """Return the `SentenceTransformer` class. Raises ImportError if it cannot be imported."""
    install_scikit_learn_placeholder_if_blocked()
    from sentence_transformers import SentenceTransformer

    _silence_progress_bars()
    return SentenceTransformer
