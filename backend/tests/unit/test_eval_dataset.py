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


# --- follow-up questions ---------------------------------------------------------------------

HISTORY = [
    {"role": "user", "content": "What is it?"},
    {"role": "assistant", "content": "It is x [1]."},
]
FOLLOW_UP = {**ANSWERABLE, "id": "f1", "question": "And more?", "history": HISTORY}


def _write_with_follow_ups(tmp_path: Path, follow_ups: object) -> Path:
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "a.md").write_text("text", encoding="utf-8")
    payload = {"questions": [ANSWERABLE], "follow_ups": follow_ups}
    (tmp_path / "questions.json").write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def test_the_shipped_set_has_follow_ups_with_an_earlier_exchange() -> None:
    dataset = load_dataset()

    assert len(dataset.follow_ups) >= 5
    assert any(q.type == "unanswerable" for q in dataset.follow_ups)
    for question in dataset.follow_ups:
        assert question.history[0][0] == "user", question.id
    ids = [q.id for q in (*dataset.questions, *dataset.follow_ups)]
    assert len(ids) == len(set(ids))
    # They are kept apart: the retrieval-only measurements must not need the model.
    assert not any(q.history for q in dataset.questions)


def test_a_follow_up_keeps_its_earlier_exchange(tmp_path: Path) -> None:
    dataset = load_dataset(_write_with_follow_ups(tmp_path, [FOLLOW_UP]))

    assert [q.id for q in dataset.questions] == ["q1"]
    assert dataset.follow_ups[0].history == (("user", "What is it?"), ("assistant", "It is x [1]."))


@pytest.mark.parametrize(
    ("follow_up", "message"),
    [
        ({k: v for k, v in FOLLOW_UP.items() if k != "history"}, "history must be a list"),
        ({**FOLLOW_UP, "history": []}, "history must be a list"),
        ({**FOLLOW_UP, "history": [HISTORY[1], HISTORY[0]]}, "must start with the user"),
        ({**FOLLOW_UP, "history": [{"role": "system", "content": "x"}]}, "each history turn"),
        ({**FOLLOW_UP, "history": [{"role": "user", "content": " "}]}, "each history turn"),
        ({**FOLLOW_UP, "history": [HISTORY[0]] * 9}, "history must be a list"),
        ({**FOLLOW_UP, "id": "q1"}, "unique"),
    ],
)
def test_malformed_follow_ups_are_rejected(tmp_path: Path, follow_up: dict, message: str) -> None:
    with pytest.raises(DatasetError, match=message):
        load_dataset(_write_with_follow_ups(tmp_path, [follow_up]))


def test_a_history_in_the_ordinary_questions_is_refused(tmp_path: Path) -> None:
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "a.md").write_text("text", encoding="utf-8")
    payload = {"questions": [{**ANSWERABLE, "history": HISTORY}]}
    (tmp_path / "questions.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatasetError, match="belongs in the follow_ups list"):
        load_dataset(tmp_path)


def test_follow_ups_must_be_a_list(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="'follow_ups' must be a list"):
        load_dataset(_write_with_follow_ups(tmp_path, {"id": "x"}))


# --- task entries: summarize, compare, extract -----------------------------------------------

SUMMARIZE = {
    "id": "t-sum",
    "mode": "summarize",
    "documents": ["a.md"],
    "answer_contains": [["x"]],
}
COMPARE = {
    "id": "t-cmp",
    "mode": "compare",
    "documents": ["a.md", "b.md"],
    "expected_sources": ["a.md", "b.md"],
    "answer_contains": [["x"], ["y", "z"]],
}
EXTRACT = {
    "id": "t-ext",
    "mode": "extract",
    "request": "every total",
    "expected_sources": ["a.md"],
    "answer_contains": [["100"]],
    "answer_must_not_contain": ["999"],
}


def _write_with_tasks(tmp_path: Path, tasks: object) -> Path:
    (tmp_path / "corpus").mkdir()
    for name in ("a.md", "b.md", "c.md", "d.md", "e.md"):
        (tmp_path / "corpus" / name).write_text("text", encoding="utf-8")
    payload = {"questions": [ANSWERABLE], "tasks": tasks}
    (tmp_path / "questions.json").write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def test_the_shipped_set_has_a_task_for_every_mode() -> None:
    dataset = load_dataset()

    assert {task.mode for task in dataset.tasks} == {"summarize", "compare", "extract"}
    ids = [x.id for x in (*dataset.questions, *dataset.follow_ups, *dataset.tasks)]
    assert len(ids) == len(set(ids))
    files = {p.name for p in dataset.corpus_dir.rglob("*") if p.is_file()}
    for task in dataset.tasks:
        assert task.answer_contains, task.id
        assert set(task.documents) | set(task.expected_sources) <= files, task.id


def test_tasks_are_kept_apart_from_the_questions() -> None:
    dataset = load_dataset()

    # Retrieval measurements must not pick them up: they need the model.
    assert not any(q.id.startswith("task-") for q in (*dataset.questions, *dataset.follow_ups))


def test_a_set_without_tasks_has_none(tmp_path: Path) -> None:
    assert load_dataset(_write(tmp_path, [ANSWERABLE])).tasks == ()


def test_valid_tasks_of_each_mode_load(tmp_path: Path) -> None:
    dataset = load_dataset(_write_with_tasks(tmp_path, [SUMMARIZE, COMPARE, EXTRACT]))

    summarize, compare, extract = dataset.tasks
    assert (summarize.mode, summarize.documents) == ("summarize", ("a.md",))
    assert compare.documents == ("a.md", "b.md")
    assert compare.answer_contains == (("x",), ("y", "z"))
    assert extract.request == "every total" and extract.documents == ()
    assert extract.answer_must_not_contain == ("999",)


@pytest.mark.parametrize(
    ("task", "message"),
    [
        ({**SUMMARIZE, "id": ""}, "every task needs an id"),
        ({**SUMMARIZE, "mode": "translate"}, "mode must be one of"),
        ({k: v for k, v in SUMMARIZE.items() if k != "mode"}, "mode must be one of"),
        ({**SUMMARIZE, "documents": []}, "exactly one document"),
        ({**SUMMARIZE, "documents": ["a.md", "b.md"]}, "exactly one document"),
        ({**SUMMARIZE, "documents": ["nothing.md"]}, "not in the corpus"),
        ({**SUMMARIZE, "request": "x"}, "only an extraction has a request"),
        ({**COMPARE, "documents": ["a.md"]}, "two to four documents"),
        ({**COMPARE, "documents": ["a.md", "b.md", "c.md", "d.md", "e.md"]}, "two to four"),
        ({**COMPARE, "documents": ["a.md", "a.md"]}, "must be different"),
        ({**COMPARE, "expected_sources": ["nothing.md"]}, "expected file"),
        ({**EXTRACT, "documents": ["a.md"]}, "takes no documents"),
        ({**EXTRACT, "request": "  "}, "extract needs a request"),
        ({k: v for k, v in EXTRACT.items() if k != "request"}, "extract needs a request"),
        ({**SUMMARIZE, "answer_contains": []}, "needs answer_contains"),
        ({**SUMMARIZE, "answer_contains": [[]]}, "no empty group"),
        ({**SUMMARIZE, "answer_contains": "x"}, "list of lists"),
        ({**SUMMARIZE, "answer_must_not_contain": "x"}, "list of non-empty strings"),
        ("not an object", "must be an object"),
    ],
)
def test_malformed_tasks_are_rejected_with_a_clear_message(
    tmp_path: Path, task: object, message: str
) -> None:
    with pytest.raises(DatasetError, match=message):
        load_dataset(_write_with_tasks(tmp_path, [task]))


def test_a_task_id_cannot_repeat_a_question_id(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="unique"):
        load_dataset(_write_with_tasks(tmp_path, [{**SUMMARIZE, "id": "q1"}]))


def test_tasks_must_be_a_list(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="'tasks' must be a list"):
        load_dataset(_write_with_tasks(tmp_path, {"id": "x"}))
