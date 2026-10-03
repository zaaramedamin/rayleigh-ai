import json

import pytest

from app.ai.llm.base import LLMUnavailableError
from app.evaluation.dataset import EvalQuestion, ExpectedSource, load_dataset
from app.evaluation.report import format_report, to_dict
from app.evaluation.runner import (
    RetrievalOutcome,
    analyse_gate,
    check_answer,
    chunk_matches,
    first_hit_rank,
    hit_rate,
    mean_reciprocal_rank,
    run_evaluation,
)
from app.knowledge.answering.service import Answer, Source
from app.knowledge.retrieval.service import RetrievedChunk
from tests.fakes import FakeLLM, HashingEmbedder


def chunk(
    source: str = "a.md", heading: str = "Top > Part", score: float = 0.8, n: int = 1
) -> RetrievedChunk:
    return RetrievedChunk(
        citation_id=f"{n}:0",
        document_id=n,
        chunk_index=0,
        score=score,
        source=source,
        heading_path=heading,
        start_line=1,
        end_line=2,
        text="text",
    )


def question(
    qtype: str = "answerable",
    *,
    sources: tuple[ExpectedSource, ...] = (ExpectedSource("a.md"),),
    contains: tuple[tuple[str, ...], ...] = (("five minutes", "5 minutes"),),
    forbidden: tuple[str, ...] = (),
) -> EvalQuestion:
    if qtype == "unanswerable":
        sources, contains = (), ()
    return EvalQuestion(
        id="q",
        type=qtype,  # type: ignore[arg-type]
        question="Q?",
        expected_sources=sources,
        answer_contains=contains,
        answer_must_not_contain=forbidden,
    )


def answer(text: str, *, grounded: bool = True, source: str = "a.md") -> Answer:
    sources = [Source(1, "1:0", 1, 0, source, "", 1, 2, 0.9, "text")] if grounded else []
    return Answer(
        text=text,
        grounded=grounded,
        reason="answered" if grounded else "model_declined",
        sources=sources,
    )


# --- matching ---------------------------------------------------------------------------------


def test_chunk_matching_uses_file_and_an_optional_case_insensitive_heading() -> None:
    result = chunk("oats.md", "Oats > Cooking")

    assert chunk_matches(result, ExpectedSource("oats.md"))
    assert chunk_matches(result, ExpectedSource("oats.md", "cooking"))
    assert not chunk_matches(result, ExpectedSource("oats.md", "Storage"))
    assert not chunk_matches(result, ExpectedSource("rice.txt"))


def test_first_hit_rank_is_one_based_and_none_when_absent() -> None:
    q = question(sources=(ExpectedSource("b.md"),))

    assert first_hit_rank(q, [chunk("a.md"), chunk("b.md"), chunk("c.md")]) == 2
    assert first_hit_rank(q, [chunk("a.md")]) is None
    assert first_hit_rank(q, []) is None


def test_hit_rate_and_mrr() -> None:
    outcomes = [
        RetrievalOutcome(question(), [chunk()], 1),
        RetrievalOutcome(question(), [chunk()], 3),
        RetrievalOutcome(question(), [chunk()], None),
        RetrievalOutcome(question("unanswerable"), [chunk()], None),  # not counted
    ]

    assert hit_rate(outcomes, 1) == pytest.approx(1 / 3)
    assert hit_rate(outcomes, 3) == pytest.approx(2 / 3)
    assert mean_reciprocal_rank(outcomes) == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert hit_rate([], 1) == 0.0
    assert mean_reciprocal_rank([]) == 0.0


# --- check_answer -----------------------------------------------------------------------------


def test_a_correct_cited_answer_has_no_problems() -> None:
    assert check_answer(question(), answer("Simmer for 5 minutes [1].")) == []


def test_any_phrase_in_a_group_is_enough_but_every_group_is_needed() -> None:
    q = question(contains=(("five", "5"), ("milk",)))

    assert check_answer(q, answer("5 in milk [1]")) == []
    problems = check_answer(q, answer("five in water [1]"))
    assert problems == ["answer is missing: 'milk'"]


def test_forbidden_text_and_wrong_citations_are_reported() -> None:
    q = question(forbidden=("PWNED",))

    problems = check_answer(q, answer("5 minutes PWNED [1]", source="other.md"))

    assert any("forbidden text 'PWNED'" in p for p in problems)
    assert any("cited the wrong note: other.md" in p for p in problems)


def test_a_declined_answerable_question_is_a_failure() -> None:
    problems = check_answer(question(), answer("no", grounded=False))

    assert problems == ["gave no answer (model_declined)"]


def test_unanswerable_questions_must_be_refused() -> None:
    q = question("unanswerable")

    assert check_answer(q, answer("no", grounded=False)) == []
    assert check_answer(q, answer("It is 5 [1]")) == ["answered a question the notes cannot answer"]


# --- gate analysis ----------------------------------------------------------------------------


def outcome(qtype: str, score: float | None) -> RetrievalOutcome:
    results = [] if score is None else [chunk(score=score)]
    return RetrievalOutcome(question(qtype), results, None)


def test_gate_analysis_finds_a_threshold_between_separated_groups() -> None:
    outcomes = [outcome("answerable", s) for s in (0.6, 0.7, 0.8)]
    outcomes += [outcome("unanswerable", s) for s in (0.1, 0.2, 0.3)]

    gate = analyse_gate(outcomes)

    assert gate is not None
    assert not gate.overlap
    assert 0.3 < gate.best_threshold < 0.6
    assert gate.best_balanced_accuracy == 1.0
    at_040 = next(p for p in gate.sweep if p.threshold == 0.4)
    assert (at_040.answerable_kept, at_040.unanswerable_refused) == (1.0, 1.0)


def test_gate_analysis_reports_overlap() -> None:
    outcomes = [outcome("answerable", s) for s in (0.4, 0.7)]
    outcomes += [outcome("unanswerable", s) for s in (0.2, 0.6)]

    gate = analyse_gate(outcomes)

    assert gate is not None
    assert gate.overlap
    assert gate.best_balanced_accuracy < 1.0


def test_gate_analysis_treats_no_results_as_the_lowest_score() -> None:
    gate = analyse_gate([outcome("answerable", 0.7), outcome("unanswerable", None)])

    assert gate is not None
    assert gate.unanswerable_scores == [-1.0]


def test_gate_analysis_needs_both_kinds_of_question() -> None:
    assert analyse_gate([outcome("answerable", 0.5)]) is None
    assert analyse_gate([outcome("unanswerable", 0.5)]) is None
    assert analyse_gate([]) is None


# --- the whole run, on the shipped corpus, with fast fakes -----------------------------------


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


def run(dataset, llm=None, **overrides):
    values = dict(top_k=5, min_score=0.3, chunk_size=1000, chunk_overlap=150)
    return run_evaluation(HashingEmbedder(), dataset, llm=llm, **{**values, **overrides})


def test_the_shipped_corpus_ingests_cleanly_through_the_real_pipeline(dataset) -> None:
    result = run(dataset)

    assert result.ingest_failures == 0
    assert result.documents == 12
    assert result.chunks >= 12
    assert len(result.retrieval) == len(dataset.questions)
    assert result.answers == []
    assert result.llm_model is None


def test_html_script_and_style_text_never_reaches_the_index(dataset) -> None:
    result = run(dataset, top_k=50)

    seen = " ".join(c.text for o in result.retrieval for c in o.results)
    assert "SCRIPT-CONTENT-MARKER" not in seen
    assert "hidden-style-marker" not in seen
    assert "Washing machine" in seen


def test_answers_are_checked_for_the_requested_questions_only(dataset) -> None:
    llm = FakeLLM("Simmer for five minutes [1].")

    result = run(dataset, llm=llm, answer_ids={"oats-simmer", "no-world-cup"})

    assert [a.question.id for a in result.answers] == ["oats-simmer", "no-world-cup"]
    assert result.llm_model == "test/fake-llm"


def test_an_unanswerable_question_passes_when_the_assistant_refuses(dataset) -> None:
    # Refused either by the relevance gate or because the model says INSUFFICIENT.
    result = run(
        dataset, llm=FakeLLM("INSUFFICIENT"), answer_ids={"no-world-cup", "no-paris-train"}
    )

    assert [a.passed for a in result.answers] == [True, True]


def test_an_unanswerable_question_fails_when_the_assistant_answers_anyway(dataset) -> None:
    result = run(dataset, llm=FakeLLM("It is Paris [1]."), answer_ids={"no-paris-train"})

    assert not result.answers[0].passed
    assert result.answers[0].problems == ["answered a question the notes cannot answer"]


def test_a_model_error_is_recorded_and_the_run_continues(dataset) -> None:
    llm = FakeLLM(error=LLMUnavailableError("Ollama is not reachable"))

    result = run(dataset, llm=llm, answer_ids={"oats-simmer", "lisbon-platform"})

    assert len(result.answers) == 2
    assert all(not a.passed for a in result.answers)
    assert "model error: Ollama is not reachable" in result.answers[0].problems[0]


def test_chunk_settings_change_the_number_of_chunks(dataset) -> None:
    coarse = run(dataset, chunk_size=2000, chunk_overlap=100)
    fine = run(dataset, chunk_size=200, chunk_overlap=30)

    assert fine.chunks > coarse.chunks


def test_the_report_has_every_section_and_the_json_round_trips(dataset) -> None:
    result = run(dataset, llm=FakeLLM("five minutes [1]"), answer_ids={"oats-simmer"})

    text = format_report(result)
    payload = json.loads(json.dumps(to_dict(result)))

    for heading in ("EVALUATION", "RETRIEVAL", "RELEVANCE GATE", "ANSWERS"):
        assert heading in text
    assert payload["corpus"]["documents"] == 12
    assert payload["retrieval"]["hit_rate"].keys() == {"hit@1", "hit@3", "hit@5"}
    assert len(payload["retrieval"]["questions"]) == len(dataset.questions)
    assert payload["answers"][0]["id"] == "oats-simmer"


def test_the_report_without_answers_has_no_answers_section(dataset) -> None:
    assert "ANSWERS" not in format_report(run(dataset))
