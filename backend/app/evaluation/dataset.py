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

    @property
    def has_answer(self) -> bool:
        return self.type != "unanswerable"


@dataclass(frozen=True)
class Dataset:
    corpus_dir: Path
    questions: tuple[EvalQuestion, ...]

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


def _parse_question(raw: Any, corpus_files: set[str]) -> EvalQuestion:
    _require(isinstance(raw, dict), "each question must be an object")
    qid = raw.get("id")
    _require(isinstance(qid, str) and qid.strip() != "", "every question needs an id")
    where = f"question {qid!r}"

    qtype = raw.get("type")
    _require(qtype in _TYPES, f"{where}: type must be one of {_TYPES}")
    text = raw.get("question")
    _require(isinstance(text, str) and text.strip() != "", f"{where}: question text is empty")

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
        _require(
            item["file"] in corpus_files,
            f"{where}: expected file {item['file']!r} is not in the corpus",
        )
        sources.append(ExpectedSource(file=item["file"], heading=heading))

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
    ids = [q.id for q in questions]
    _require(len(ids) == len(set(ids)), "question ids must be unique")
    _require(bool(questions), "there are no questions")
    return Dataset(corpus_dir=corpus_dir, questions=questions)
