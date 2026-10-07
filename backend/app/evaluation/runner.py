"""Run the evaluation set through the real pipeline and measure the result.

Everything happens in a throwaway database and an in-memory vector store, so the user's own
notes and data folder are never read or changed.
"""

import statistics
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.ai.embeddings.base import EmbeddingProvider
from app.ai.llm.base import ChatMessage, LLMError, LLMProvider, Role
from app.evaluation.dataset import Dataset, EvalQuestion, EvalTask, ExpectedSource
from app.knowledge.answering.prompts import versions
from app.knowledge.answering.rewrite import standalone_question
from app.knowledge.answering.service import Answer, compose_answer
from app.knowledge.answering.tasks import (
    DocumentText,
    compare,
    extract,
    load_document_text,
    summarize,
)
from app.knowledge.indexing.service import index_pending
from app.knowledge.ingestion.service import ingest_folders
from app.knowledge.retrieval.keyword import KeywordSearchUnavailable
from app.knowledge.retrieval.service import SEARCH_MODES, RetrievedChunk, SearchMode, retrieve
from app.storage.database import Base, create_db_engine
from app.storage.models import DOC_ACTIVE, Chunk, Document
from app.storage.vector_store import QdrantVectorStore

RETRIEVAL_CUTOFFS = (1, 3, 5)


# --- results ----------------------------------------------------------------------------------


@dataclass
class RetrievalOutcome:
    question: EvalQuestion
    results: list[RetrievedChunk]
    first_hit_rank: int | None  # 1-based rank of the first expected chunk, None if not retrieved

    @property
    def top_score(self) -> float | None:
        return self.results[0].score if self.results else None


@dataclass
class AnswerOutcome:
    question: EvalQuestion
    answer: Answer | None
    problems: list[str]
    seconds: float

    @property
    def passed(self) -> bool:
        return not self.problems


@dataclass
class FollowUpOutcome:
    """A question asked after an earlier exchange, rewritten before the notes were searched."""

    question: EvalQuestion
    searched_for: str  # what the rewrite turned the message into
    first_hit_rank: int | None  # where the expected note ranked for that rewrite
    answer: Answer | None
    problems: list[str]
    seconds: float

    @property
    def passed(self) -> bool:
        return not self.problems


@dataclass
class TaskOutcome:
    """A summarize, compare or extract job, and what it produced."""

    task: EvalTask
    text: str  # the summary, the comparison, or the extracted rows as lines of "item: value"
    cited: tuple[str, ...]  # file names of the notes the result cites (none for a summary)
    problems: list[str]
    seconds: float

    @property
    def passed(self) -> bool:
        return not self.problems


@dataclass
class EvalRun:
    embedding_model: str
    llm_model: str | None
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_score: float
    documents: int
    chunks: int
    ingest_failures: int
    retrieval: list[RetrievalOutcome]
    answers: list[AnswerOutcome] = field(default_factory=list)
    # The search mode `retrieval` and `answers` were run with, and the retrieval outcomes of
    # every mode that could run, so the modes can be compared on the same questions.
    mode: SearchMode = "vector"
    by_mode: dict[str, list[RetrievalOutcome]] = field(default_factory=dict)
    follow_ups: list[FollowUpOutcome] = field(default_factory=list)
    tasks: list[TaskOutcome] = field(default_factory=list)
    # The version and fingerprint of every prompt the answers were written with, so two
    # reports can be told apart when a prompt changed between them.
    prompts: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class GatePoint:
    threshold: float
    answerable_kept: float  # share of answerable questions whose best note scores >= threshold
    unanswerable_refused: float  # share of unanswerable ones whose best note scores < threshold


@dataclass(frozen=True)
class GateAnalysis:
    answerable_scores: list[float]
    unanswerable_scores: list[float]
    sweep: list[GatePoint]
    best_threshold: float
    best_balanced_accuracy: float

    @property
    def overlap(self) -> bool:
        """True if some unanswerable question scores at least as high as an answerable one."""
        return max(self.unanswerable_scores) >= min(self.answerable_scores)


# --- matching and checking --------------------------------------------------------------------


def chunk_matches(chunk: RetrievedChunk, expected: ExpectedSource) -> bool:
    if chunk.source != expected.file:
        return False
    if expected.heading is not None and expected.heading.lower() not in chunk.heading_path.lower():
        return False
    if expected.page is None:
        return True
    if chunk.start_page is None:
        return False
    return chunk.start_page <= expected.page <= (chunk.end_page or chunk.start_page)


def first_hit_rank(question: EvalQuestion, results: Sequence[RetrievedChunk]) -> int | None:
    for rank, chunk in enumerate(results, start=1):
        if any(chunk_matches(chunk, expected) for expected in question.expected_sources):
            return rank
    return None


def check_answer(question: EvalQuestion, answer: Answer) -> list[str]:
    """What is wrong with this answer? An empty list means it is correct."""
    problems: list[str] = []
    text = answer.text.lower()

    if question.type == "unanswerable":
        if answer.grounded:
            problems.append("answered a question the notes cannot answer")
        return problems

    if not answer.grounded:
        return [f"gave no answer ({answer.reason})"]

    for group in question.answer_contains:
        if not any(phrase.lower() in text for phrase in group):
            problems.append("answer is missing: " + " / ".join(repr(p) for p in group))
    for phrase in question.answer_must_not_contain:
        if phrase.lower() in text:
            problems.append(f"answer contains forbidden text {phrase!r}")

    expected_files = {s.file for s in question.expected_sources}
    cited_files = {source.source for source in answer.sources}
    if not cited_files & expected_files:
        problems.append("cited the wrong note: " + ", ".join(sorted(cited_files)))
    return problems


def check_task(task: EvalTask, text: str, cited: Sequence[str]) -> list[str]:
    """What is wrong with this result? An empty list means it is correct."""
    problems: list[str] = []
    lowered = text.lower()
    for group in task.answer_contains:
        if not any(phrase.lower() in lowered for phrase in group):
            problems.append("result is missing: " + " / ".join(repr(p) for p in group))
    for phrase in task.answer_must_not_contain:
        if phrase.lower() in lowered:
            problems.append(f"result contains forbidden text {phrase!r}")
    missing = sorted(set(task.expected_sources) - set(cited))
    if missing:
        problems.append("did not cite: " + ", ".join(missing))
    return problems


# --- gate analysis ----------------------------------------------------------------------------


def analyse_gate(outcomes: Sequence[RetrievalOutcome]) -> GateAnalysis | None:
    """How well does the best note's score separate answerable from unanswerable questions?"""
    answerable = sorted(
        o.top_score if o.top_score is not None else -1.0 for o in outcomes if o.question.has_answer
    )
    unanswerable = sorted(
        o.top_score if o.top_score is not None else -1.0
        for o in outcomes
        if not o.question.has_answer
    )
    if not answerable or not unanswerable:
        return None

    def point(threshold: float) -> GatePoint:
        kept = sum(score >= threshold for score in answerable) / len(answerable)
        refused = sum(score < threshold for score in unanswerable) / len(unanswerable)
        return GatePoint(round(threshold, 4), kept, refused)

    sweep = [point(t / 100) for t in range(10, 75, 5)]
    everything = sorted(set(answerable + unanswerable))
    midpoints = [(a + b) / 2 for a, b in zip(everything, everything[1:], strict=False)]
    best = max(
        (point(t) for t in midpoints) if midpoints else [point(0.3)],
        key=lambda p: ((p.answerable_kept + p.unanswerable_refused) / 2, -p.threshold),
    )
    return GateAnalysis(
        answerable_scores=answerable,
        unanswerable_scores=unanswerable,
        sweep=sweep,
        best_threshold=best.threshold,
        best_balanced_accuracy=(best.answerable_kept + best.unanswerable_refused) / 2,
    )


# --- metrics ----------------------------------------------------------------------------------


def hit_rate(outcomes: Sequence[RetrievalOutcome], cutoff: int) -> float:
    scored = [o for o in outcomes if o.question.has_answer]
    if not scored:
        return 0.0
    hits = sum(o.first_hit_rank is not None and o.first_hit_rank <= cutoff for o in scored)
    return hits / len(scored)


def mean_reciprocal_rank(outcomes: Sequence[RetrievalOutcome]) -> float:
    scored = [o for o in outcomes if o.question.has_answer]
    if not scored:
        return 0.0
    return statistics.fmean(1 / o.first_hit_rank if o.first_hit_rank else 0.0 for o in scored)


# --- running ----------------------------------------------------------------------------------


@dataclass
class PreparedCorpus:
    session: Session
    store: QdrantVectorStore
    documents: int
    chunks: int
    ingest_failures: int


@contextmanager
def prepared_corpus(
    dataset: Dataset,
    embedder: EmbeddingProvider,
    *,
    chunk_size: int,
    chunk_overlap: int,
    max_file_size_bytes: int = 5 * 1024 * 1024,
) -> Iterator[PreparedCorpus]:
    """Ingest, chunk and index the corpus with the real pipeline, in a temporary location."""
    with tempfile.TemporaryDirectory(prefix="reyleight-eval-", ignore_cleanup_errors=True) as tmp:
        data_dir = Path(tmp) / "data"
        engine = create_db_engine(data_dir)
        Base.metadata.create_all(engine)
        store = QdrantVectorStore.in_memory("eval", embedder.dimension)
        try:
            with Session(engine) as session:
                summary = ingest_folders(
                    session,
                    data_dir,
                    [dataset.corpus_dir],
                    max_file_size_bytes,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                index_pending(session, embedder, store)
                chunks = session.scalar(select(func.count()).select_from(Chunk)) or 0
                yield PreparedCorpus(
                    session=session,
                    store=store,
                    documents=summary.added + summary.unchanged,
                    chunks=chunks,
                    ingest_failures=sum(summary.failed.values()),
                )
        finally:
            store.close()
            engine.dispose()


def run_retrieval(
    corpus: PreparedCorpus,
    embedder: EmbeddingProvider,
    dataset: Dataset,
    top_k: int,
    mode: SearchMode = "vector",
) -> list[RetrievalOutcome]:
    outcomes = []
    for question in dataset.questions:
        results = retrieve(
            corpus.session, embedder, corpus.store, question.question, top_k=top_k, mode=mode
        )
        outcomes.append(RetrievalOutcome(question, results, first_hit_rank(question, results)))
    return outcomes


def run_answers(
    llm: LLMProvider,
    outcomes: Sequence[RetrievalOutcome],
    min_score: float,
    only_ids: set[str] | None = None,
) -> list[AnswerOutcome]:
    results = []
    for outcome in outcomes:
        question = outcome.question
        if only_ids is not None and question.id not in only_ids:
            continue
        started = time.monotonic()
        try:
            answer = compose_answer(llm, question.question, outcome.results, min_score)
            problems = check_answer(question, answer)
        except LLMError as exc:
            answer, problems = None, [f"model error: {exc}"]
        results.append(AnswerOutcome(question, answer, problems, time.monotonic() - started))
    return results


def run_follow_ups(
    corpus: PreparedCorpus,
    embedder: EmbeddingProvider,
    llm: LLMProvider,
    dataset: Dataset,
    *,
    top_k: int,
    min_score: float,
    mode: SearchMode,
    only_ids: set[str] | None = None,
) -> list[FollowUpOutcome]:
    """Each follow-up goes through the same steps as `/ask` with a history: rewrite it into a
    standalone question, search for that, answer from what was found."""
    outcomes = []
    for question in dataset.follow_ups:
        if only_ids is not None and question.id not in only_ids:
            continue
        started = time.monotonic()
        history = [ChatMessage(cast(Role, role), text) for role, text in question.history]
        try:
            searched_for = standalone_question(llm, question.question, history).text
            results = retrieve(
                corpus.session, embedder, corpus.store, searched_for, top_k=top_k, mode=mode
            )
            answer = compose_answer(llm, searched_for, results, min_score)
            problems = check_answer(question, answer)
            rank = first_hit_rank(question, results)
        except LLMError as exc:
            searched_for, rank, answer, problems = (
                question.question,
                None,
                None,
                [f"model error: {exc}"],
            )
        outcomes.append(
            FollowUpOutcome(
                question, searched_for, rank, answer, problems, time.monotonic() - started
            )
        )
    return outcomes


def _document_text(corpus: PreparedCorpus, document_ids: dict[str, int], name: str) -> DocumentText:
    document_id = document_ids.get(name)
    if document_id is None:
        raise LookupError(f"{name} is not in the library: it could not be ingested")
    return load_document_text(corpus.session, document_id)


def _run_task(
    corpus: PreparedCorpus,
    embedder: EmbeddingProvider,
    llm: LLMProvider,
    task: EvalTask,
    document_ids: dict[str, int],
    *,
    top_k: int,
    min_score: float,
    mode: SearchMode,
) -> tuple[str, tuple[str, ...], list[str]]:
    """The text, the cited files and the problems of one job."""
    if task.mode == "summarize":
        text = summarize(llm, _document_text(corpus, document_ids, task.documents[0])).text
        return text, (), check_task(task, text, ())
    if task.mode == "compare":
        comparison = compare(
            llm, [_document_text(corpus, document_ids, name) for name in task.documents]
        )
        cited = tuple(entry.name for entry in comparison.sources)
        if not comparison.grounded:
            return comparison.text, cited, [f"gave no comparison ({comparison.reason})"]
        return comparison.text, cited, check_task(task, comparison.text, cited)
    retrieved = retrieve(
        corpus.session, embedder, corpus.store, task.request, top_k=max(top_k, 8), mode=mode
    )
    extraction = extract(llm, task.request, retrieved, min_score)
    text = "\n".join(f"{row.item}: {row.value}" for row in extraction.rows)
    cited = tuple(source.source for source in extraction.sources)
    if not extraction.rows:
        return text, cited, [f"found no rows ({extraction.reason})"]
    return text, cited, check_task(task, text, cited)


def run_tasks(
    corpus: PreparedCorpus,
    embedder: EmbeddingProvider,
    llm: LLMProvider,
    dataset: Dataset,
    *,
    top_k: int,
    min_score: float,
    mode: SearchMode,
    only_ids: set[str] | None = None,
) -> list[TaskOutcome]:
    """Each job goes through the same code as the `/tasks` routes, over the throwaway library."""
    document_ids = {
        d.original_filename: d.id
        for d in corpus.session.scalars(select(Document).where(Document.status == DOC_ACTIVE))
    }
    outcomes = []
    for task in dataset.tasks:
        if only_ids is not None and task.id not in only_ids:
            continue
        started = time.monotonic()
        try:
            text, cited, problems = _run_task(
                corpus,
                embedder,
                llm,
                task,
                document_ids,
                top_k=top_k,
                min_score=min_score,
                mode=mode,
            )
        except LLMError as exc:
            text, cited, problems = "", (), [f"model error: {exc}"]
        except (LookupError, ValueError) as exc:
            text, cited, problems = "", (), [f"could not run: {exc}"]
        outcomes.append(TaskOutcome(task, text, cited, problems, time.monotonic() - started))
    return outcomes


def run_evaluation(
    embedder: EmbeddingProvider,
    dataset: Dataset,
    *,
    llm: LLMProvider | None,
    top_k: int,
    min_score: float,
    chunk_size: int,
    chunk_overlap: int,
    answer_ids: set[str] | None = None,
    mode: SearchMode = "vector",
) -> EvalRun:
    """Retrieval metrics for every question and every search mode, plus answer checks when an
    LLM is given. `retrieval` and the answers use `mode`."""
    with prepared_corpus(
        dataset, embedder, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ) as corpus:
        by_mode: dict[str, list[RetrievalOutcome]] = {}
        for each in SEARCH_MODES:
            try:
                by_mode[each] = run_retrieval(corpus, embedder, dataset, top_k, each)
            except KeywordSearchUnavailable:
                continue  # this SQLite has no FTS5: the other modes still run
        if mode not in by_mode:
            raise KeywordSearchUnavailable("this SQLite was built without FTS5")
        retrieval = by_mode[mode]
        answers = run_answers(llm, retrieval, min_score, answer_ids) if llm is not None else []
        follow_ups = (
            run_follow_ups(
                corpus,
                embedder,
                llm,
                dataset,
                top_k=top_k,
                min_score=min_score,
                mode=mode,
                only_ids=answer_ids,
            )
            if llm is not None
            else []
        )
        tasks = (
            run_tasks(
                corpus,
                embedder,
                llm,
                dataset,
                top_k=top_k,
                min_score=min_score,
                mode=mode,
                only_ids=answer_ids,
            )
            if llm is not None
            else []
        )
        return EvalRun(
            embedding_model=embedder.model_name,
            llm_model=getattr(llm, "model_name", None) if llm is not None else None,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            min_score=min_score,
            documents=corpus.documents,
            chunks=corpus.chunks,
            ingest_failures=corpus.ingest_failures,
            retrieval=retrieval,
            answers=answers,
            mode=mode,
            by_mode=by_mode,
            follow_ups=follow_ups,
            tasks=tasks,
            prompts=versions() if llm is not None else {},
        )
