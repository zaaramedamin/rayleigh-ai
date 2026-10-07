"""The evaluation set: a small corpus of notes and questions with known answers."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = BACKEND_DIR / "eval"

QuestionType = Literal["answerable", "unanswerable", "injection"]
_TYPES = ("answerable", "unanswerable", "injection")


class DatasetError(ValueError):
    """The evaluation set is malformed."""


@dataclass(frozen=True)
class ExpectedSource:
    file: str
    heading: str | None = None  # part of the heading path that must match, e.g. "Cooking"
    page: int | None = None  # for PDFs: a page the chunk must include


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    type: QuestionType
    question: str
    expected_sources: tuple[ExpectedSource, ...] = ()
    # Every group must match; inside a group, any one phrase is enough (case-insensitive).
    answer_contains: tuple[tuple[str, ...], ...] = ()
    answer_must_not_contain: tuple[str, ...] = ()
    smoke: bool = False
    note: str = ""
    # A label for reporting results by kind of question, e.g. "exact-match" or "pdf".
    group: str = ""
    # For a follow-up: what was said before, oldest first, as (role, text).
    history: tuple[tuple[str, str], ...] = ()

    @property
    def has_answer(self) -> bool:
        return self.type != "unanswerable"


TaskMode = Literal["summarize", "compare", "extract"]
_TASK_MODES = ("summarize", "compare", "extract")


@dataclass(frozen=True)
class EvalTask:
    """A job beyond answering a question, with what its result must and must not contain."""

    id: str
    mode: TaskMode
    # summarize: one file; compare: two to four. Names of files in the corpus.
    documents: tuple[str, ...] = ()
    # extract: which facts are wanted.
    request: str = ""
    # Files the result must cite (compare, extract), by name.
    expected_sources: tuple[str, ...] = ()
    # Every group must match; inside a group, any one phrase is enough (case-insensitive).
    answer_contains: tuple[tuple[str, ...], ...] = ()
    answer_must_not_contain: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class Dataset:
    corpus_dir: Path
    questions: tuple[EvalQuestion, ...]
    # Questions that only make sense after an earlier exchange. They need the model, because
    # the follow-up is rewritten before the notes are searched, so only `--answers` runs them.
    follow_ups: tuple[EvalQuestion, ...] = ()
    # Summarize, compare and extract jobs. They need the model, so only `--answers` runs them.
    tasks: tuple[EvalTask, ...] = ()

    def of_type(self, *types: str) -> list[EvalQuestion]:
        return [q for q in self.questions if q.type in types]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetError(message)


def _string_list(value: Any, where: str) -> tuple[str, ...]:
    _require(
        isinstance(value, list) and all(isinstance(v, str) and v for v in value),
        f"{where} must be a list of non-empty strings",
    )
    return tuple(value)


MAX_HISTORY_TURNS = 8


def _parse_history(raw: Any, where: str) -> tuple[tuple[str, str], ...]:
    _require(
        isinstance(raw, list) and 0 < len(raw) <= MAX_HISTORY_TURNS,
        f"{where}: history must be a list of 1 to {MAX_HISTORY_TURNS} turns",
    )
    turns = []
    for turn in raw:
        _require(
            isinstance(turn, dict)
            and turn.get("role") in ("user", "assistant")
            and isinstance(turn.get("content"), str)
            and turn["content"].strip() != "",
            f"{where}: each history turn needs a role (user or assistant) and some text",
        )
        turns.append((turn["role"], turn["content"].strip()))
    _require(turns[0][0] == "user", f"{where}: the history must start with the user")
    return tuple(turns)


def _parse_task(raw: Any, corpus_files: set[str]) -> EvalTask:
    _require(isinstance(raw, dict), "each task must be an object")
    task_id = raw.get("id")
    _require(isinstance(task_id, str) and task_id.strip() != "", "every task needs an id")
    where = f"task {task_id!r}"
    mode = raw.get("mode")
    _require(mode in _TASK_MODES, f"{where}: mode must be one of {_TASK_MODES}")

    documents = _string_list(raw.get("documents", []), f"{where}: documents")
    for name in documents:
        _require(name in corpus_files, f"{where}: document {name!r} is not in the corpus")
    _require(len(set(documents)) == len(documents), f"{where}: documents must be different")
    if mode == "summarize":
        _require(len(documents) == 1, f"{where}: a summary needs exactly one document")
    elif mode == "compare":
        _require(2 <= len(documents) <= 4, f"{where}: a comparison needs two to four documents")
    else:
        _require(not documents, f"{where}: an extraction searches the notes, it takes no documents")

    request = raw.get("request", "")
    if mode == "extract":
        _require(
            isinstance(request, str) and request.strip() != "", f"{where}: extract needs a request"
        )
    else:
        _require(not request, f"{where}: only an extraction has a request")

    sources = _string_list(raw.get("expected_sources", []), f"{where}: expected_sources")
    for name in sources:
        _require(name in corpus_files, f"{where}: expected file {name!r} is not in the corpus")
    groups_raw = raw.get("answer_contains", [])
    _require(isinstance(groups_raw, list), f"{where}: answer_contains must be a list of lists")
    groups = tuple(_string_list(g, f"{where}: an answer_contains group") for g in groups_raw)
    _require(bool(groups) and all(groups), f"{where}: needs answer_contains, with no empty group")
    forbidden = _string_list(
        raw.get("answer_must_not_contain", []), f"{where}: answer_must_not_contain"
    )
    return EvalTask(
        id=task_id,
        mode=mode,
        documents=documents,
        request=request.strip() if isinstance(request, str) else "",
        expected_sources=sources,
        answer_contains=groups,
        answer_must_not_contain=forbidden,
        note=str(raw.get("note", "")),
    )


def _parse_question(raw: Any, corpus_files: set[str], *, follow_up: bool = False) -> EvalQuestion:
    _require(isinstance(raw, dict), "each question must be an object")
    qid = raw.get("id")
    _require(isinstance(qid, str) and qid.strip() != "", "every question needs an id")
    where = f"question {qid!r}"

    qtype = raw.get("type")
    _require(qtype in _TYPES, f"{where}: type must be one of {_TYPES}")
    text = raw.get("question")
    _require(isinstance(text, str) and text.strip() != "", f"{where}: question text is empty")

    history: tuple[tuple[str, str], ...] = ()
    if follow_up:
        history = _parse_history(raw.get("history"), where)
    else:
        _require("history" not in raw, f"{where}: history belongs in the follow_ups list")

    sources_raw = raw.get("expected_sources", [])
    _require(isinstance(sources_raw, list), f"{where}: expected_sources must be a list")
    sources = []
    for item in sources_raw:
        _require(
            isinstance(item, dict) and isinstance(item.get("file"), str),
            f"{where}: each expected source needs a file",
        )
        heading = item.get("heading")
        _require(
            heading is None or (isinstance(heading, str) and heading != ""),
            f"{where}: heading must be a non-empty string",
        )
        page = item.get("page")
        _require(
            page is None or (isinstance(page, int) and not isinstance(page, bool) and page >= 1),
            f"{where}: page must be a positive whole number",
        )
        _require(
            item["file"] in corpus_files,
            f"{where}: expected file {item['file']!r} is not in the corpus",
        )
        sources.append(ExpectedSource(file=item["file"], heading=heading, page=page))

    groups_raw = raw.get("answer_contains", [])
    _require(isinstance(groups_raw, list), f"{where}: answer_contains must be a list of lists")
    groups = tuple(_string_list(g, f"{where}: an answer_contains group") for g in groups_raw)
    _require(
        all(groups),
        f"{where}: an answer_contains group is empty, so it could never match",
    )
    forbidden = _string_list(
        raw.get("answer_must_not_contain", []), f"{where}: answer_must_not_contain"
    )

    if qtype == "unanswerable":
        _require(
            not sources and not groups,
            f"{where}: an unanswerable question cannot have expected sources or answers",
        )
    else:
        _require(bool(sources), f"{where}: needs at least one expected source")
        _require(bool(groups), f"{where}: needs answer_contains")

    return EvalQuestion(
        id=qid,
        type=qtype,
        question=text.strip(),
        expected_sources=tuple(sources),
        answer_contains=groups,
        answer_must_not_contain=forbidden,
        smoke=bool(raw.get("smoke", False)),
        note=str(raw.get("note", "")),
        group=str(raw.get("group", "")),
        history=history,
    )


def load_dataset(directory: Path = DEFAULT_EVAL_DIR) -> Dataset:
    """Load and validate `directory/questions.json` and `directory/corpus/`."""
    corpus_dir = directory / "corpus"
    questions_file = directory / "questions.json"
    _require(corpus_dir.is_dir(), f"corpus folder not found: {corpus_dir}")
    _require(questions_file.is_file(), f"questions file not found: {questions_file}")

    try:
        raw = json.loads(questions_file.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise DatasetError(f"questions.json is not valid JSON: {exc}") from exc
    _require(
        isinstance(raw, dict) and isinstance(raw.get("questions"), list),
        "questions.json needs a 'questions' list",
    )

    names = [p.name for p in corpus_dir.rglob("*") if p.is_file()]
    _require(len(names) == len(set(names)), "corpus file names must be unique")
    corpus_files = set(names)

    questions = tuple(_parse_question(q, corpus_files) for q in raw["questions"])
    follow_ups_raw = raw.get("follow_ups", [])
    _require(isinstance(follow_ups_raw, list), "'follow_ups' must be a list")
    follow_ups = tuple(_parse_question(q, corpus_files, follow_up=True) for q in follow_ups_raw)
    tasks_raw = raw.get("tasks", [])
    _require(isinstance(tasks_raw, list), "'tasks' must be a list")
    tasks = tuple(_parse_task(t, corpus_files) for t in tasks_raw)
    ids = [*(q.id for q in questions), *(q.id for q in follow_ups), *(t.id for t in tasks)]
    _require(len(ids) == len(set(ids)), "question ids must be unique")
    _require(bool(questions), "there are no questions")
    return Dataset(corpus_dir=corpus_dir, questions=questions, follow_ups=follow_ups, tasks=tasks)
