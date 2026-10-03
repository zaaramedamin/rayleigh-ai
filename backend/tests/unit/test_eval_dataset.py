import json
from pathlib import Path

import pytest

from app.evaluation.dataset import DatasetError, load_dataset


def test_the_shipped_evaluation_set_is_valid_and_balanced() -> None:
    dataset = load_dataset()

    answerable = dataset.of_type("answerable")
    unanswerable = dataset.of_type("unanswerable")
    injection = dataset.of_type("injection")
    assert len(answerable) >= 25
    assert len(unanswerable) >= 8
    assert len(injection) >= 1
    assert len({q.id for q in dataset.questions}) == len(dataset.questions)


def test_the_quick_check_covers_every_kind_of_question() -> None:
    smoke_types = {q.type for q in load_dataset().questions if q.smoke}

    assert smoke_types == {"answerable", "unanswerable", "injection"}


def test_the_corpus_has_several_file_types() -> None:
    dataset = load_dataset()

    suffixes = {p.suffix for p in dataset.corpus_dir.rglob("*") if p.is_file()}
    assert {".md", ".txt", ".csv", ".json", ".html"} <= suffixes


# --- validation ------------------------------------------------------------------------------


def _write(tmp_path: Path, questions: object, files: tuple[str, ...] = ("a.md",)) -> Path:
    (tmp_path / "corpus").mkdir()
    for name in files:
        (tmp_path / "corpus" / name).write_text("text", encoding="utf-8")
    payload = questions if isinstance(questions, str) else json.dumps({"questions": questions})
    (tmp_path / "questions.json").write_text(payload, encoding="utf-8")
    return tmp_path


ANSWERABLE = {
    "id": "q1",
    "type": "answerable",
    "question": "What?",
    "expected_sources": [{"file": "a.md"}],
    "answer_contains": [["x"]],
}
UNANSWERABLE = {"id": "q2", "type": "unanswerable", "question": "Why?"}


def test_a_minimal_valid_set_loads(tmp_path: Path) -> None:
    dataset = load_dataset(_write(tmp_path, [ANSWERABLE, UNANSWERABLE]))

    assert [q.id for q in dataset.questions] == ["q1", "q2"]
    assert dataset.questions[0].expected_sources[0].file == "a.md"
    assert dataset.questions[1].has_answer is False


@pytest.mark.parametrize(
    ("questions", "message"),
    [
        ([ANSWERABLE, {**ANSWERABLE}], "unique"),
        ([{**ANSWERABLE, "type": "maybe"}], "type must be"),
        ([{**ANSWERABLE, "question": "  "}], "question text is empty"),
        ([{**ANSWERABLE, "id": ""}], "needs an id"),
        ([{**ANSWERABLE, "expected_sources": [{"file": "missing.md"}]}], "not in the corpus"),
        ([{**ANSWERABLE, "expected_sources": []}], "at least one expected source"),
        ([{**ANSWERABLE, "answer_contains": []}], "needs answer_contains"),
        ([{**ANSWERABLE, "answer_contains": [[]]}], "group is empty"),
        ([{**ANSWERABLE, "answer_contains": ["not a list"]}], "list of non-empty strings"),
        ([{**UNANSWERABLE, "expected_sources": [{"file": "a.md"}]}], "cannot have expected"),
        ([{**UNANSWERABLE, "answer_contains": [["x"]]}], "cannot have expected"),
        ([{**ANSWERABLE, "expected_sources": [{"file": "a.md", "heading": ""}]}], "heading"),
        ([], "there are no questions"),
        (["not an object"], "must be an object"),
    ],
)
def test_malformed_questions_are_rejected_with_a_clear_message(
    tmp_path: Path, questions: list, message: str
) -> None:
    with pytest.raises(DatasetError, match=message):
        load_dataset(_write(tmp_path, questions))


def test_invalid_json_is_reported(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not valid JSON"):
        load_dataset(_write(tmp_path, "{ nope"))


def test_a_missing_corpus_or_questions_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="corpus folder not found"):
        load_dataset(tmp_path)

    (tmp_path / "corpus").mkdir()
    with pytest.raises(DatasetError, match="questions file not found"):
        load_dataset(tmp_path)


def test_duplicate_corpus_file_names_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "corpus" / "one").mkdir(parents=True)
    (tmp_path / "corpus" / "two").mkdir(parents=True)
    (tmp_path / "corpus" / "one" / "a.md").write_text("x")
    (tmp_path / "corpus" / "two" / "a.md").write_text("y")
    (tmp_path / "questions.json").write_text(json.dumps({"questions": [UNANSWERABLE]}))

    with pytest.raises(DatasetError, match="file names must be unique"):
        load_dataset(tmp_path)
