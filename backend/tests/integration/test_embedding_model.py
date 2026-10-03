"""Tests against the real embedding model. Skipped until `python -m app download-model` ran."""

import math
import os

import pytest

from app.ai.embeddings.sentence_transformer import SentenceTransformerProvider
from app.core.config import get_settings
from tests.helpers import real_embedding_model_problem

SETTINGS = get_settings()

_PROBLEM = real_embedding_model_problem()
pytestmark = pytest.mark.skipif(_PROBLEM is not None, reason=_PROBLEM or "")


@pytest.fixture(scope="module")
def provider() -> SentenceTransformerProvider:
    return SentenceTransformerProvider(SETTINGS.embedding_model, SETTINGS.models_dir)


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))  # vectors are normalised


def test_vectors_have_the_model_dimension_and_unit_length(
    provider: SentenceTransformerProvider,
) -> None:
    vector = provider.embed_query("hello world")

    assert len(vector) == provider.dimension > 0
    assert math.isclose(sum(x * x for x in vector), 1.0, rel_tol=1e-4)


def test_same_text_gives_the_same_vector(provider: SentenceTransformerProvider) -> None:
    first = provider.embed_query("Oats are a good source of fibre.")
    second = provider.embed_query("Oats are a good source of fibre.")

    assert all(math.isclose(x, y, abs_tol=1e-6) for x, y in zip(first, second, strict=True))


def test_similar_sentences_score_higher_than_unrelated_ones(
    provider: SentenceTransformerProvider,
) -> None:
    anchor, similar, unrelated = provider.embed_documents(
        [
            "How many calories are in a bowl of oatmeal?",
            "Calorie content of one serving of porridge oats",
            "The train to Paris leaves at noon from platform four.",
        ]
    )

    assert _cosine(anchor, similar) > _cosine(anchor, unrelated) + 0.2


def test_batch_and_single_embeddings_agree(provider: SentenceTransformerProvider) -> None:
    texts = ["a short note", "a much longer note about cooking rice with vegetables"]

    batch = provider.embed_documents(texts)
    single = [provider.embed_query(text) for text in texts]

    for a, b in zip(batch, single, strict=True):
        assert all(math.isclose(x, y, abs_tol=1e-4) for x, y in zip(a, b, strict=True))


def test_empty_input_gives_no_vectors(provider: SentenceTransformerProvider) -> None:
    assert provider.embed_documents([]) == []


def test_loading_switches_hugging_face_libraries_to_offline_mode(
    provider: SentenceTransformerProvider,
) -> None:
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
