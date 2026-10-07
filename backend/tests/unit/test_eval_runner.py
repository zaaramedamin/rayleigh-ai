import json
import re
from pathlib import Path

import pytest

from app.ai.llm.base import LLMUnavailableError
from app.evaluation.dataset import (
    DEFAULT_EVAL_DIR,
    EvalQuestion,
    EvalTask,
    ExpectedSource,
    load_dataset,
)
from app.evaluation.report import format_report, to_dict
from app.evaluation.runner import (
    RetrievalOutcome,
    analyse_gate,
    check_answer,
    check_task,
    chunk_matches,
    first_hit_rank,
    hit_rate,
    mean_reciprocal_rank,
    run_evaluation,
)
from app.knowledge.answering.prompts import EXTRACT, versions
from app.knowledge.answering.service import Answer, Source
from app.knowledge.retrieval.service import RetrievedChunk
from tests.fakes import FakeLLM, HashingEmbedder

# How many notes the shipped evaluation corpus holds, so tests do not break when it grows.
CORPUS_DOCUMENTS = sum(1 for p in (DEFAULT_EVAL_DIR / "corpus").rglob("*") if p.is_file())


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
    assert result.documents == CORPUS_DOCUMENTS
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
    assert payload["corpus"]["documents"] == CORPUS_DOCUMENTS
    assert payload["retrieval"]["hit_rate"].keys() == {"hit@1", "hit@3", "hit@5"}
    assert len(payload["retrieval"]["questions"]) == len(dataset.questions)
    assert payload["answers"][0]["id"] == "oats-simmer"


def test_the_report_without_answers_has_no_answers_section(dataset) -> None:
    assert "ANSWERS" not in format_report(run(dataset))


def test_every_search_mode_is_measured_on_the_same_questions(dataset) -> None:
    result = run(dataset)

    assert set(result.by_mode) == {"vector", "keyword", "hybrid"}
    assert all(len(outcomes) == len(dataset.questions) for outcomes in result.by_mode.values())
    assert result.retrieval is result.by_mode[result.mode]


def test_the_report_compares_the_modes_and_the_question_groups(dataset) -> None:
    result = run(dataset)

    text = format_report(result)
    payload = json.loads(json.dumps(to_dict(result)))

    assert "SEARCH MODES" in text
    for name in ("vector", "keyword", "hybrid"):
        assert name in text and "hit@1" in payload["modes"][name]["hit_rate"]
    assert "exact-match" in text  # the shipped set has a group of exact-code questions
    assert "exact-match" in payload["modes"]["hybrid"]["groups"]


def test_the_mode_the_gate_and_answers_use_can_be_chosen(dataset) -> None:
    result = run(dataset, mode="keyword")

    assert result.mode == "keyword" and result.retrieval is result.by_mode["keyword"]
    assert "search mode keyword" in format_report(result)


# --- follow-up questions ---------------------------------------------------------------------


def test_follow_ups_need_the_model_and_are_not_part_of_the_retrieval_numbers(dataset) -> None:
    result = run(dataset)

    assert result.follow_ups == []
    assert len(result.retrieval) == len(dataset.questions)


def test_a_follow_up_is_rewritten_searched_and_answered(dataset) -> None:
    llm = FakeLLM()
    llm.script = [
        "How much was invoice INV-2026-0418 for the Lisbon workshop?",
        "Invoice INV-2026-0418 totals 1,284.50 euros [1].",
    ]

    result = run(dataset, llm=llm, answer_ids={"fu-invoice-amount"})

    (outcome,) = result.follow_ups
    assert outcome.searched_for == "How much was invoice INV-2026-0418 for the Lisbon workshop?"
    assert outcome.first_hit_rank == 1
    assert outcome.passed, outcome.problems
    rewrite_prompt = llm.calls[0][1]
    assert "When was invoice INV-2026-0418 paid?" in rewrite_prompt  # the earlier exchange
    assert "Latest message: And how much was it?" in rewrite_prompt


def test_a_follow_up_fails_when_the_answer_comes_from_the_wrong_note(dataset) -> None:
    llm = FakeLLM()
    llm.script = [
        "How much was invoice INV-2025-0418 for the Madrid conference?",
        "Invoice INV-2025-0418 totals 2,150.00 euros [1].",
    ]

    result = run(dataset, llm=llm, answer_ids={"fu-invoice-amount"})

    (outcome,) = result.follow_ups
    assert not outcome.passed
    assert any("2,150" in problem for problem in outcome.problems)


def test_a_follow_up_the_notes_cannot_answer_must_be_refused(dataset) -> None:
    llm = FakeLLM("INSUFFICIENT")
    llm.script = ["How long is the battery life of the router?"]

    result = run(dataset, llm=llm, answer_ids={"fu-no-answer"})

    assert [o.passed for o in result.follow_ups] == [True]


def test_only_the_requested_follow_ups_run(dataset) -> None:
    result = run(dataset, llm=FakeLLM("INSUFFICIENT"), answer_ids={"oats-simmer"})

    assert result.follow_ups == []


def test_a_model_that_cannot_be_reached_fails_the_follow_ups_without_stopping_the_run(
    dataset,
) -> None:
    llm = FakeLLM(error=LLMUnavailableError("Ollama is not reachable"))

    result = run(dataset, llm=llm, answer_ids={"fu-invoice-amount", "fu-printer-serial"})

    assert len(result.follow_ups) == 2
    assert all(not o.passed for o in result.follow_ups)


def test_the_report_lists_failed_follow_ups_with_what_was_searched_for(dataset) -> None:
    llm = FakeLLM("INSUFFICIENT")
    llm.script = ["something else entirely"]

    result = run(dataset, llm=llm, answer_ids={"fu-invoice-amount"})
    text = format_report(result)

    assert "FOLLOW-UP QUESTIONS" in text
    assert "0/1 answered correctly" in text
    assert "searched for 'something else entirely'" in text
    follow_ups = to_dict(result)["follow_ups"]
    assert follow_ups[0]["id"] == "fu-invoice-amount" and follow_ups[0]["passed"] is False
    json.dumps(to_dict(result))  # still plain JSON


# --- summarize, compare and extract jobs -----------------------------------------------------

TASK_IDS = {
    "task-summarize-lisbon",
    "task-summarize-devices",
    "task-compare-invoices",
    "task-extract-invoices",
    "task-extract-devices",
}
LISBON_SUMMARY = (
    "A trip to Lisbon: the 9:40 train, Hotel Alfama for three nights, a 450 euro budget."
)
DEVICES_SUMMARY = (
    "The ZX-400 router is SN-7F3K-9921; the Brightprint BP-30 printer is SN-2B8M-1146."
)
INVOICES_COMPARISON = (
    "In 2026 the Lisbon workshop came to 1,284.50 euros [1]; in 2025 the Madrid conference "
    "came to 2,150.00 euros [2]."
)
INVOICE = re.compile(r"(INV-\d{4}-\d{4}).*?totals ([\d,.]+)")
SERIAL = re.compile(r"(SN-\w+-\w+)")


class NoteReader(FakeLLM):
    """Answers an extraction the way a good model would: by reading the numbered notes it was
    given and writing the table. Every other prompt gets the scripted replies."""

    def generate(self, system: str, user: str) -> str:
        reply = super().generate(system, user)
        if system != EXTRACT.system:
            return reply
        wanted = user.rsplit("Request:", 1)[1]
        rows = []
        for number, body in re.findall(
            r"=== NOTE (\d+) BEGIN \w+ ===\n(.*?)\n=== NOTE \d+ END", user, re.DOTALL
        ):
            if "serial" in wanted:
                found = [(serial, serial) for serial in SERIAL.findall(body)]
            else:
                found = INVOICE.findall(body)
            rows += [{"item": item, "value": value, "note": int(number)} for item, value in found]
        return json.dumps(rows)


def reader() -> NoteReader:
    llm = NoteReader()
    llm.script = [LISBON_SUMMARY, DEVICES_SUMMARY, INVOICES_COMPARISON]
    return llm


def test_jobs_need_the_model_and_are_not_part_of_the_retrieval_numbers(dataset) -> None:
    result = run(dataset)

    assert result.tasks == [] and result.prompts == {}
    assert len(result.retrieval) == len(dataset.questions)


def test_a_model_that_does_each_job_well_passes_every_shipped_task(dataset) -> None:
    # The stand-in embedder scores a natural request low (the invoices come at about 0.1), so the
    # relevance gate is opened; the real model clears it, as the live evaluation shows.
    result = run(dataset, llm=reader(), answer_ids=TASK_IDS, min_score=0.0)

    assert [o.task.id for o in result.tasks] == [t.id for t in dataset.tasks]
    for outcome in result.tasks:
        assert outcome.passed, (outcome.task.id, outcome.problems, outcome.text)
    comparison = next(o for o in result.tasks if o.task.mode == "compare")
    assert set(comparison.cited) == {"invoices-2026.md", "invoices-2025.md"}
    invoices = next(o for o in result.tasks if o.task.id == "task-extract-invoices")
    assert "INV-2026-0418: 1,284.50" in invoices.text and "INV-2025-0533: 780.00" in invoices.text
    assert result.answers == [] and result.follow_ups == []  # only the jobs were asked for


def test_the_run_records_the_prompts_it_was_made_with(dataset) -> None:
    result = run(dataset, llm=FakeLLM("INSUFFICIENT"), answer_ids={"oats-simmer"})

    assert result.prompts == versions()
    assert set(result.prompts) >= {"answer", "rewrite", "summarize", "compare", "extract"}


def test_only_the_requested_jobs_run(dataset) -> None:
    llm = reader()
    llm.script = [DEVICES_SUMMARY]

    result = run(dataset, llm=llm, answer_ids={"task-summarize-devices"})

    assert [o.task.id for o in result.tasks] == ["task-summarize-devices"]
    assert result.tasks[0].passed, result.tasks[0].problems


def test_a_summary_that_loses_a_fact_or_adds_one_fails(dataset) -> None:
    llm = reader()
    llm.script = ["A trip to Berlin by train.", DEVICES_SUMMARY]

    result = run(dataset, llm=llm, answer_ids={"task-summarize-lisbon", "task-summarize-devices"})

    lisbon, devices = result.tasks
    assert not lisbon.passed and devices.passed
    assert any("missing" in problem and "450" in problem for problem in lisbon.problems)
    assert any("forbidden text 'berlin'" in problem for problem in lisbon.problems)


def test_a_comparison_that_cites_one_note_only_fails(dataset) -> None:
    llm = reader()
    llm.script = ["In 2026 the Lisbon workshop was 1,284.50 euros [1]; in 2025 Madrid, 2,150.00."]

    (outcome,) = run(dataset, llm=llm, answer_ids={"task-compare-invoices"}).tasks

    assert outcome.problems == ["did not cite: invoices-2025.md"]
    assert outcome.cited == ("invoices-2026.md",)


def test_a_comparison_that_cites_nothing_is_withheld_and_fails(dataset) -> None:
    llm = reader()
    llm.script = ["The two years differ."]

    (outcome,) = run(dataset, llm=llm, answer_ids={"task-compare-invoices"}).tasks

    assert outcome.problems == ["gave no comparison (no_valid_citation)"]


def test_an_extraction_the_model_cannot_write_as_a_table_fails(dataset) -> None:
    llm = FakeLLM("Here are the invoices: INV-2026-0418.")

    (outcome,) = run(dataset, llm=llm, answer_ids={"task-extract-invoices"}).tasks

    assert outcome.problems == ["found no rows (unreadable)"] and outcome.text == ""


def test_a_forbidden_row_fails_the_extraction(dataset) -> None:
    rows = [
        {"item": "INV-2026-0418", "value": "1,284.50", "note": 1},
        {"item": "INV-2027-0001", "value": "960", "note": 1},
    ]
    llm = FakeLLM(json.dumps(rows))

    (outcome,) = run(dataset, llm=llm, answer_ids={"task-extract-invoices"}).tasks

    assert any("forbidden text 'INV-2027'" in problem for problem in outcome.problems)


def test_a_model_that_cannot_be_reached_fails_the_jobs_without_stopping_the_run(dataset) -> None:
    llm = FakeLLM(error=LLMUnavailableError("Ollama is not reachable"))

    result = run(dataset, llm=llm, answer_ids=TASK_IDS)

    assert len(result.tasks) == len(TASK_IDS)
    assert all(o.problems == ["model error: Ollama is not reachable"] for o in result.tasks)


def test_a_document_that_could_not_be_ingested_fails_its_job_and_nothing_else(
    tmp_path: Path,
) -> None:
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "blank.md").write_text("   \n", encoding="utf-8")
    (tmp_path / "corpus" / "ok.md").write_text("The code is 4711.\n", encoding="utf-8")

    def summarize_job(task_id: str, name: str, phrase: str) -> dict:
        return {
            "id": task_id,
            "mode": "summarize",
            "documents": [name],
            "answer_contains": [[phrase]],
        }

    questions = {
        "questions": [{"id": "q", "type": "unanswerable", "question": "Why?"}],
        "tasks": [
            summarize_job("t-blank", "blank.md", "x"),
            summarize_job("t-ok", "ok.md", "4711"),
        ],
    }
    (tmp_path / "questions.json").write_text(json.dumps(questions), encoding="utf-8")

    result = run(load_dataset(tmp_path), llm=FakeLLM("The code is 4711."))

    blank, ok = result.tasks
    assert blank.problems and blank.problems[0].startswith("could not run:")
    assert ok.passed, ok.problems


def test_checking_a_result_by_its_phrases_and_its_cited_notes() -> None:
    task = EvalTask(
        id="t",
        mode="compare",
        documents=("a.md", "b.md"),
        expected_sources=("a.md", "b.md"),
        answer_contains=(("Alpha",), ("two", "2")),
        answer_must_not_contain=("beta",),
    )

    assert check_task(task, "ALPHA has 2 items", ["b.md", "a.md"]) == []
    assert check_task(task, "alpha", ["a.md", "b.md"]) == ["result is missing: 'two' / '2'"]
    assert check_task(task, "alpha two Beta", ["a.md", "b.md"]) == [
        "result contains forbidden text 'beta'"
    ]
    assert check_task(task, "alpha two", ["a.md"]) == ["did not cite: b.md"]
    assert check_task(task, "", []) == [
        "result is missing: 'Alpha'",
        "result is missing: 'two' / '2'",
        "did not cite: a.md, b.md",
    ]
