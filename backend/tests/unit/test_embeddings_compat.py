import sys
from collections.abc import Iterator

import pytest

from app.ai.embeddings.compat import (
    _STUBBED_MODULES,
    install_scikit_learn_placeholder_if_blocked,
)

PREFIXES = ("sklearn", "sentence_transformers")


@pytest.fixture
def restore_modules() -> Iterator[None]:
    """Put back whatever scikit-learn / sentence-transformers modules were loaded."""
    saved = {n: m for n, m in sys.modules.items() if n.split(".")[0] in PREFIXES}
    yield
    for name in [n for n in sys.modules if n.split(".")[0] in PREFIXES]:
        del sys.modules[name]
    sys.modules.update(saved)


def _block_scikit_learn(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry makes `import sklearn.metrics` raise ImportError, like a blocked file does.
    for name in _STUBBED_MODULES:
        monkeypatch.setitem(sys.modules, name, None)


def test_placeholder_is_installed_only_when_scikit_learn_cannot_be_imported(
    monkeypatch: pytest.MonkeyPatch, restore_modules: None
) -> None:
    _block_scikit_learn(monkeypatch)

    assert install_scikit_learn_placeholder_if_blocked() is True

    for name in _STUBBED_MODULES:
        assert sys.modules[name].__name__ == name
        assert sys.modules[name].__spec__ is not None


def test_nothing_is_changed_when_scikit_learn_imports_fine(
    monkeypatch: pytest.MonkeyPatch, restore_modules: None
) -> None:
    class Fine:
        pass

    for name in _STUBBED_MODULES:
        monkeypatch.setitem(sys.modules, name, Fine())

    assert install_scikit_learn_placeholder_if_blocked() is False
    assert all(isinstance(sys.modules[name], Fine) for name in _STUBBED_MODULES)


def test_importing_names_works_but_calling_them_gives_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, restore_modules: None
) -> None:
    _block_scikit_learn(monkeypatch)
    install_scikit_learn_placeholder_if_blocked()

    from sklearn.metrics import pairwise_distances  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="scikit-learn could not be loaded"):
        pairwise_distances([[1.0]], [[1.0]])


def test_half_loaded_modules_from_a_failed_import_are_cleared(
    monkeypatch: pytest.MonkeyPatch, restore_modules: None
) -> None:
    _block_scikit_learn(monkeypatch)
    monkeypatch.setitem(sys.modules, "sklearn.metrics.cluster", object())  # a leftover

    install_scikit_learn_placeholder_if_blocked()

    assert "sklearn.metrics.cluster" not in sys.modules
