"""Step 8 acceptance: with the real model, paraphrased questions find the right note.

Skipped until `python -m app download-model` has been run.
"""

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.ai.embeddings.base import is_model_downloaded
from app.ai.embeddings.sentence_transformer import SentenceTransformerProvider
from app.core.config import get_settings
from app.knowledge.retrieval.service import retrieve
from app.storage.vector_store import QdrantVectorStore
from tests.helpers import add_and_index

SETTINGS = get_settings()

pytestmark = pytest.mark.skipif(
    not is_model_downloaded(SETTINGS.models_dir, SETTINGS.embedding_model),
    reason="embedding model not downloaded (run `python -m app download-model`)",
)

NOTES = {
    "oats.md": (
        "# Oats\n\n"
        "Oats are a whole grain rich in soluble fibre (beta-glucan), which supports healthy "
        "digestion and helps lower cholesterol.\n\n"
        "## Cooking\n\n"
        "Simmer rolled oats in milk for five minutes and stir often.\n"
    ),
    "rice.txt": (
        "Cook white rice with twice its volume of water. Bring it to the boil, then cover "
        "and simmer for 18 minutes."
    ),
    "travel.md": (
        "# Lisbon trip\n\n"
        "The train to Lisbon departs at 9:40 from platform 4 at Porto Campanha.\n\n"
        "## Hotel\n\n"
        "Check-in at the hotel is from 3 pm; the room is booked for three nights.\n"
    ),
    "office.md": (
        "# Office\n\nThe guest wifi network is called ReyNet; the password is on the fridge.\n"
    ),
}

CASES = [
    ("How many minutes should rice cook?", "rice.txt", ""),
    ("Which platform is the Lisbon train on?", "travel.md", "Lisbon trip"),
    ("When can I check into my hotel?", "travel.md", "Lisbon trip > Hotel"),
    ("Is porridge good for digestion?", "oats.md", "Oats"),
    ("How do I make oatmeal?", "oats.md", "Oats > Cooking"),
    ("How do I get on the internet at work?", "office.md", "Office"),
]


@pytest.fixture(scope="module")
def provider() -> SentenceTransformerProvider:
    return SentenceTransformerProvider(SETTINGS.embedding_model, SETTINGS.models_dir)


def test_paraphrased_questions_find_the_right_chunk(
    session: Session, data_dir: Path, provider: SentenceTransformerProvider
) -> None:
    store = QdrantVectorStore.in_memory("real", provider.dimension)
    add_and_index(session, data_dir, provider, store, NOTES)

    misses = []
    for question, source, heading_path in CASES:
        results = retrieve(session, provider, store, question, top_k=3)
        top = results[0]
        if (top.source, top.heading_path) != (source, heading_path):
            ranked = [(r.source, r.heading_path, round(r.score, 3)) for r in results]
            misses.append((question, ranked))
    store.close()

    assert misses == []
