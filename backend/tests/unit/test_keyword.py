"""Slice 4.1 and 4.2: keyword search in memory, and hybrid ranking that keeps the gate honest."""

import math
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.knowledge.ingestion.service import ingest_folders
from app.knowledge.retrieval import keyword
from app.knowledge.retrieval.keyword import (
    MAX_TERMS,
    KeywordIndex,
    KeywordSearchUnavailable,
    forget_indexes,
    keyword_search,
    match_expression,
    query_terms,
)
from app.knowledge.retrieval.service import reciprocal_rank_fusion, retrieve
from app.storage.models import DOC_SUPERSEDED, Chunk, Document
from app.storage.vector_store import QdrantVectorStore, VectorHit
from tests.fakes import HashingEmbedder
from tests.helpers import add_and_index

NOTES = {
    "invoices.md": (
        "# Invoices\n\nInvoice INV-2026-0418 totals 1284 euros for the Lisbon workshop.\n\n"
        "## Contacts\n\nThe accountant is Ms Okonkwo-Silva, phone extension 5503.\n"
    ),
    "oats.md": "# Oats\n\nOats are high in fibre. Simmer the oats in milk for five minutes.\n",
    "rice.txt": "Rice needs twice its volume of water and eighteen minutes.",
    "cafe.md": "# Café notes\n\nThe café on the corner serves the best pastéis de nata.\n",
}


@pytest.fixture(autouse=True)
def fresh_indexes() -> Iterator[None]:
    forget_indexes()
    yield
    forget_indexes()


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def store(embedder: HashingEmbedder) -> Iterator[QdrantVectorStore]:
    vector_store = QdrantVectorStore.in_memory("keyword", embedder.dimension)
    yield vector_store
    vector_store.close()


@pytest.fixture
def documents(
    session: Session, data_dir: Path, embedder: HashingEmbedder, store: QdrantVectorStore
) -> dict[str, Document]:
    return add_and_index(session, data_dir, embedder, store, NOTES)


# --- turning a question into words --------------------------------------------------------------


def test_a_question_is_cut_into_its_useful_words() -> None:
    assert query_terms("What is the total of invoice INV-2026-0418?") == [
        "total",
        "invoice",
        "inv",
        "2026",
        "0418",
    ]


def test_words_are_lower_cased_accents_kept_and_repeats_dropped() -> None:
    assert query_terms("Café CAFÉ café notes Notes") == ["café", "notes"]


def test_single_letters_are_dropped_but_single_digits_are_kept() -> None:
    assert query_terms("a b 7 x 12") == ["7", "12"]


def test_only_stop_words_gives_no_terms() -> None:
    assert query_terms("what is the of and") == []
    assert query_terms("") == [] and query_terms("   ?!  ") == []


def test_the_number_of_terms_is_capped() -> None:
    query = " ".join(f"word{number}" for number in range(100))

    assert len(query_terms(query)) == MAX_TERMS


def test_every_term_is_quoted_so_it_can_only_be_a_word() -> None:
    assert match_expression(["total", "and", "near"]) == '"total" OR "and" OR "near"'
    assert match_expression(['a"b']) == '"a""b"'


# --- the index ------------------------------------------------------------------------------------


def test_an_exact_code_is_found(session: Session, documents: dict[str, Document]) -> None:
    hits = keyword_search(session, "INV-2026-0418", top_k=3)

    assert hits[0].document_id == documents["invoices.md"].id
    assert hits[0].score > 0


def test_a_name_with_a_hyphen_and_a_number_are_found(
    session: Session, documents: dict[str, Document]
) -> None:
    by_name = keyword_search(session, "Okonkwo-Silva", top_k=3)
    by_number = keyword_search(session, "extension 5503", top_k=3)

    assert by_name[0].document_id == by_number[0].document_id == documents["invoices.md"].id


def test_accents_do_not_get_in_the_way(session: Session, documents: dict[str, Document]) -> None:
    assert keyword_search(session, "cafe", top_k=3)[0].document_id == documents["cafe.md"].id
    assert (
        keyword_search(session, "café pasteis", top_k=3)[0].document_id == documents["cafe.md"].id
    )


def test_a_match_in_a_heading_counts_for_more_than_one_in_the_body(
    session: Session, data_dir: Path, embedder: HashingEmbedder, store: QdrantVectorStore
) -> None:
    add_and_index(
        session,
        data_dir,
        embedder,
        store,
        {
            "body.md": "# Notes\n\nThe zeppelin hangar is by the river.\n",
            "heading.md": "# Zeppelin\n\nA long story about a harbour.\n",
        },
    )

    hits = keyword_search(session, "zeppelin", top_k=2)

    names = {d.id: d.original_filename for d in session.scalars(select(Document))}
    assert [names[hit.document_id] for hit in hits] == ["heading.md", "body.md"]


def test_results_can_be_limited_to_documents_and_file_types(
    session: Session, documents: dict[str, Document]
) -> None:
    both = keyword_search(session, "minutes", top_k=5)
    only_rice = keyword_search(session, "minutes", top_k=5, document_ids=[documents["rice.txt"].id])
    only_md = keyword_search(session, "minutes", top_k=5, file_types=[".md"])
    only_txt = keyword_search(session, "minutes", top_k=5, file_types=["TXT"])

    assert len(both) == 2
    assert [h.document_id for h in only_rice] == [documents["rice.txt"].id]
    assert [h.document_id for h in only_md] == [documents["oats.md"].id]
    assert [h.document_id for h in only_txt] == [documents["rice.txt"].id]


@pytest.mark.parametrize(
    "query",
    [
        'foo" OR "bar',
        "oats AND NOT rice",
        "NEAR(oats milk, 3)",
        "heading:oats",
        "oats*",
        "^oats",
        '"unclosed quote',
        "-oats",
        "(((",
        "oats OR",
        "body: ' DROP TABLE chunks_fts; --",
        "\\",
    ],
)
def test_query_syntax_is_just_words_and_never_an_error(
    session: Session, documents: dict[str, Document], query: str
) -> None:
    hits = keyword_search(session, query, top_k=5)

    assert isinstance(hits, list)


def test_an_operator_word_does_not_widen_or_narrow_the_search(
    session: Session, documents: dict[str, Document]
) -> None:
    plain = {h.document_id for h in keyword_search(session, "oats rice", top_k=5)}
    with_not = {h.document_id for h in keyword_search(session, "oats NOT rice", top_k=5)}

    assert plain == with_not == {documents["oats.md"].id, documents["rice.txt"].id}


def test_a_query_with_nothing_to_look_for_returns_nothing(
    session: Session, documents: dict[str, Document]
) -> None:
    assert keyword_search(session, "what is the", top_k=5) == []
    assert keyword_search(session, "zzzznothingmatches", top_k=5) == []


# --- keeping the index right ---------------------------------------------------------------------


def test_the_index_is_built_once_and_reused(
    session: Session, documents: dict[str, Document], monkeypatch: pytest.MonkeyPatch
) -> None:
    builds: list[int] = []
    original = keyword._build

    def counting(*args, **kwargs):  # type: ignore[no-untyped-def]
        builds.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(keyword, "_build", counting)

    for _ in range(3):
        keyword_search(session, "oats", top_k=1)

    assert builds == [1]


def test_the_index_follows_the_library(
    session: Session,
    data_dir: Path,
    documents: dict[str, Document],
    embedder: HashingEmbedder,
    store: QdrantVectorStore,
) -> None:
    assert keyword_search(session, "marmalade", top_k=3) == []

    add_and_index(session, data_dir, embedder, store, {"jam.md": "Orange marmalade recipe."})

    assert len(keyword_search(session, "marmalade", top_k=3)) == 1


def test_a_replaced_version_is_no_longer_found(
    session: Session, data_dir: Path, tmp_path: Path
) -> None:
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "plan.txt").write_text("The launch codename is BANANA.")
    ingest_folders(session, data_dir, [folder], 100_000)
    assert len(keyword_search(session, "banana", top_k=3)) == 1

    (folder / "plan.txt").write_text("The launch codename is CHERRY.")
    ingest_folders(session, data_dir, [folder], 100_000)

    assert keyword_search(session, "banana", top_k=3) == []
    assert len(keyword_search(session, "cherry", top_k=3)) == 1
    assert session.scalars(select(Document).where(Document.status == DOC_SUPERSEDED)).one()


def test_the_index_is_never_written_to_disk(
    session: Session, data_dir: Path, documents: dict[str, Document]
) -> None:
    before = sorted(p.name for p in data_dir.rglob("*") if p.is_file())

    keyword_search(session, "oats", top_k=3)

    assert sorted(p.name for p in data_dir.rglob("*") if p.is_file()) == before
    assert "fts" not in "".join(before).lower()


def test_without_fts5_building_the_index_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    import sqlite3

    class NoFts(sqlite3.Connection):
        def execute(self, sql, *args):  # type: ignore[no-untyped-def]
            if "fts5" in str(sql):
                raise sqlite3.OperationalError("no such module: fts5")
            return super().execute(sql, *args)

    real_connect = sqlite3.connect
    monkeypatch.setattr(
        keyword.sqlite3, "connect", lambda *a, **k: real_connect(*a, factory=NoFts, **k)
    )

    with pytest.raises(KeywordSearchUnavailable):
        KeywordIndex((0, 0, 0))


# --- merging the two rankings --------------------------------------------------------------------


def test_fusion_prefers_what_both_rankings_like() -> None:
    vector = [(1, 0), (2, 0), (3, 0), (4, 0)]
    words = [(9, 0), (3, 0), (1, 0)]

    fused = reciprocal_rank_fusion([vector, words])

    assert fused[:2] == [(1, 0), (3, 0)]  # both rankings have them, near the top
    assert set(fused) == {(1, 0), (2, 0), (3, 0), (4, 0), (9, 0)}


def test_fusion_math_is_the_reciprocal_rank_sum() -> None:
    fused = reciprocal_rank_fusion([[(1, 0), (2, 0)], [(2, 0)]])

    # (2,0): 1/62 + 1/61 beats (1,0): 1/61
    assert fused == [(2, 0), (1, 0)]


def test_fusion_of_nothing_and_of_one_ranking() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[(5, 1), (4, 2)]]) == [(5, 1), (4, 2)]
    assert reciprocal_rank_fusion([[(5, 1)], []]) == [(5, 1)]


def test_equal_scores_keep_the_order_of_the_first_ranking() -> None:
    assert reciprocal_rank_fusion([[(1, 0)], [(2, 0)]]) == [(1, 0), (2, 0)]


# --- retrieve in the three modes -----------------------------------------------------------------


def test_keyword_mode_finds_the_exact_chunk_and_reports_its_match_strength(
    session: Session,
    embedder: HashingEmbedder,
    store: QdrantVectorStore,
    documents: dict[str, Document],
) -> None:
    results = retrieve(session, embedder, store, "INV-2026-0418", top_k=3, mode="keyword")

    assert results[0].source == "invoices.md"
    assert results[0].keyword_score is not None and results[0].keyword_score > 0


def test_vector_mode_is_unchanged_and_has_no_keyword_score(
    session: Session,
    embedder: HashingEmbedder,
    store: QdrantVectorStore,
    documents: dict[str, Document],
) -> None:
    results = retrieve(session, embedder, store, "oats in milk", top_k=3)

    assert results[0].source == "oats.md"
    assert all(result.keyword_score is None for result in results)


def test_hybrid_finds_an_exact_match_that_meaning_search_missed(
    session: Session,
    embedder: HashingEmbedder,
    store: QdrantVectorStore,
    documents: dict[str, Document],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oats, rice = documents["oats.md"], documents["rice.txt"]

    def meaning_misses(*_args: object, **_kwargs: object) -> list[VectorHit]:
        return [VectorHit(oats.id, 0, "x", 0.5), VectorHit(rice.id, 0, "x", 0.4)]

    monkeypatch.setattr(store, "search", meaning_misses)

    vector = retrieve(session, embedder, store, "INV-2026-0418", top_k=3, mode="vector")
    hybrid = retrieve(session, embedder, store, "INV-2026-0418", top_k=3, mode="hybrid")

    assert "invoices.md" not in {r.source for r in vector}
    assert hybrid[0].source == "invoices.md"


def test_the_score_of_every_result_is_the_cosine_similarity_whatever_the_mode(
    session: Session,
    embedder: HashingEmbedder,
    store: QdrantVectorStore,
    documents: dict[str, Document],
) -> None:
    question = "INV-2026-0418 workshop"
    query_vector = embedder.embed_query(question)
    chunk = session.scalars(
        select(Chunk)
        .where(Chunk.document_id == documents["invoices.md"].id)
        .order_by(Chunk.chunk_index)
    ).first()
    assert chunk is not None
    text = f"{chunk.heading_path}\n\n{chunk.text}" if chunk.heading_path else chunk.text
    chunk_vector = embedder.embed_documents([text])[0]
    expected = sum(a * b for a, b in zip(query_vector, chunk_vector, strict=True)) / (
        math.sqrt(sum(a * a for a in query_vector)) * math.sqrt(sum(b * b for b in chunk_vector))
    )

    for mode in ("vector", "keyword", "hybrid"):
        results = retrieve(session, embedder, store, question, top_k=5, mode=mode)  # type: ignore[arg-type]
        found = next(
            r for r in results if r.chunk_index == chunk.chunk_index and r.source == "invoices.md"
        )
        assert found.score == pytest.approx(expected, abs=1e-6), mode


def test_the_gate_still_sees_meaning_not_keyword_strength(
    session: Session,
    embedder: HashingEmbedder,
    store: QdrantVectorStore,
    documents: dict[str, Document],
) -> None:
    # A rare word matches strongly by keyword, but the chunk is about something else entirely.
    results = retrieve(
        session, embedder, store, "Okonkwo-Silva", top_k=3, mode="keyword", min_score=0.9
    )

    assert results == []  # keyword strength alone never makes a note "relevant enough"


def test_notes_that_are_not_searchable_yet_are_not_returned_by_keyword(
    session: Session,
    data_dir: Path,
    embedder: HashingEmbedder,
    store: QdrantVectorStore,
    documents: dict[str, Document],
) -> None:
    session.execute(
        update(Chunk)
        .where(Chunk.document_id == documents["rice.txt"].id)
        .values(indexed_model=None)
    )
    session.commit()
    forget_indexes()

    results = retrieve(session, embedder, store, "eighteen minutes", top_k=5, mode="keyword")

    assert "rice.txt" not in {r.source for r in results}


def test_hybrid_falls_back_to_meaning_when_fts5_is_missing_but_keyword_mode_fails(
    session: Session,
    embedder: HashingEmbedder,
    store: QdrantVectorStore,
    documents: dict[str, Document],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise KeywordSearchUnavailable("this SQLite was built without FTS5")

    monkeypatch.setattr("app.knowledge.retrieval.service.keyword_search", unavailable)

    hybrid = retrieve(session, embedder, store, "oats in milk", top_k=3, mode="hybrid")

    assert hybrid[0].source == "oats.md"
    with pytest.raises(KeywordSearchUnavailable):
        retrieve(session, embedder, store, "oats in milk", top_k=3, mode="keyword")


def test_an_unknown_mode_is_rejected(
    session: Session, embedder: HashingEmbedder, store: QdrantVectorStore
) -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        retrieve(session, embedder, store, "oats", top_k=3, mode="fuzzy")  # type: ignore[arg-type]


def test_filters_apply_in_every_mode(
    session: Session,
    embedder: HashingEmbedder,
    store: QdrantVectorStore,
    documents: dict[str, Document],
) -> None:
    for mode in ("keyword", "hybrid"):
        results = retrieve(
            session,
            embedder,
            store,
            "minutes",
            top_k=5,
            file_types=[".txt"],
            mode=mode,  # type: ignore[arg-type]
        )
        assert {r.source for r in results} <= {"rice.txt"}, mode


def real_hit(session: Session, document: Document, score: float) -> VectorHit:
    """A meaning-search hit for a document's first chunk, with the text hash the index keeps."""
    from app.knowledge.indexing.service import text_hash

    chunk = session.scalars(
        select(Chunk).where(Chunk.document_id == document.id).order_by(Chunk.chunk_index)
    ).first()
    assert chunk is not None
    return VectorHit(document.id, chunk.chunk_index, text_hash(chunk.text), score)


# --- how much of the question a chunk holds ------------------------------------------------------


def test_a_hit_says_how_many_of_the_questions_words_it_holds(
    session: Session, documents: dict[str, Document]
) -> None:
    hits = keyword_search(session, "invoice INV-2026-0418 totals", top_k=5)
    by_document = {h.document_id: h for h in hits}

    assert by_document[documents["invoices.md"].id].coverage == 1.0  # all five words are there
    assert query_terms("invoice INV-2026-0418 totals") == [
        "invoice",
        "inv",
        "2026",
        "0418",
        "totals",
    ]


def test_coverage_counts_the_words_present_not_how_often(
    session: Session, documents: dict[str, Document]
) -> None:
    # oats.md has "oats" and "milk" but not "banana" or "cheese": two of four words.
    hits = keyword_search(session, "oats milk banana cheese", top_k=5)

    oats = next(h for h in hits if h.document_id == documents["oats.md"].id)
    assert oats.coverage == pytest.approx(0.5)


def test_accents_count_as_present_when_measuring_coverage(
    session: Session, documents: dict[str, Document]
) -> None:
    hit = keyword_search(session, "cafe pasteis", top_k=3)[0]

    assert hit.document_id == documents["cafe.md"].id and hit.coverage == 1.0


def test_underscores_split_words_the_way_the_index_does() -> None:
    assert query_terms("snake_case and room_5503") == ["snake", "case", "room", "5503"]


def test_a_weak_keyword_match_does_not_change_the_order(
    session: Session,
    embedder: HashingEmbedder,
    store: QdrantVectorStore,
    documents: dict[str, Document],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oats, rice = documents["oats.md"], documents["rice.txt"]
    # Meaning search is sure about the oats. A generic word of the question happens to appear in the
    # rice note: one word of five, which is not enough to matter.
    hits = [real_hit(session, oats, 0.6), real_hit(session, rice, 0.4)]
    monkeypatch.setattr(store, "search", lambda *_a, **_k: hits)

    results = retrieve(
        session, embedder, store, "fibre cooking oats minutes eighteen", top_k=3, mode="hybrid"
    )

    assert results[0].source == "oats.md"


def test_a_chunk_holding_all_the_words_beats_one_meaning_search_liked_slightly_more(
    session: Session,
    embedder: HashingEmbedder,
    store: QdrantVectorStore,
    documents: dict[str, Document],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oats, invoices = documents["oats.md"], documents["invoices.md"]
    hits = [real_hit(session, oats, 0.5), real_hit(session, invoices, 0.49)]
    monkeypatch.setattr(store, "search", lambda *_a, **_k: hits)

    results = retrieve(session, embedder, store, "INV-2026-0418", top_k=2, mode="hybrid")

    assert (
        results[0].source == "invoices.md"
    )  # the exact code is in this chunk and not in the other


def test_weighted_fusion_lets_a_strong_keyword_hit_win_a_tie() -> None:
    from app.knowledge.retrieval.service import weighted_rank_fusion

    vector = [((1, 0), 1.0), ((2, 0), 1.0)]
    keyword = [((2, 0), 1.0), ((1, 0), 0.6)]

    assert weighted_rank_fusion([vector, keyword])[0] == (2, 0)
    # equal weights would have tied them, and the first ranking would have won
    assert reciprocal_rank_fusion([[(1, 0), (2, 0)], [(2, 0), (1, 0)]])[0] == (1, 0)
