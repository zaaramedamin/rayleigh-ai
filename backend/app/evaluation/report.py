"""Turn an EvalRun into text for people and JSON for files."""

import statistics
from collections import defaultdict
from typing import Any

from app.evaluation.runner import (
    RETRIEVAL_CUTOFFS,
    EvalRun,
    analyse_gate,
    hit_rate,
    mean_reciprocal_rank,
)


def _pct(value: float) -> str:
    return f"{value * 100:5.1f}%"


def _retrieval_section(run: EvalRun) -> list[str]:
    scored = [o for o in run.retrieval if o.question.has_answer]
    lines = [f"RETRIEVAL  ({len(scored)} questions whose answer is in the notes)"]
    cutoffs = [c for c in RETRIEVAL_CUTOFFS if c <= run.top_k]
    lines.append(
        "  "
        + "   ".join(f"hit@{c} {_pct(hit_rate(run.retrieval, c))}" for c in cutoffs)
        + f"   MRR {mean_reciprocal_rank(run.retrieval):.3f}"
    )
    misses = [o for o in scored if o.first_hit_rank is None]
    late = [o for o in scored if o.first_hit_rank is not None and o.first_hit_rank > 1]
    for o in misses:
        expected = ", ".join(
            f"{s.file}" + (f" > {s.heading}" if s.heading else "")
            for s in o.question.expected_sources
        )
        got = ", ".join(f"{r.source} ({r.score:.2f})" for r in o.results[:3]) or "nothing"
        lines.append(f"  MISS  {o.question.id}: wanted {expected}; got {got}")
    for o in late:
        lines.append(f"  late  {o.question.id}: right note only at rank {o.first_hit_rank}")
    if not misses and not late:
        lines.append("  every expected note was ranked first")
    return lines


def _gate_section(run: EvalRun) -> list[str]:
    gate = analyse_gate(run.retrieval)
    lines = ["RELEVANCE GATE  (best note score: answerable vs unanswerable questions)"]
    if gate is None:
        return [*lines, "  needs both answerable and unanswerable questions"]
    a, u = gate.answerable_scores, gate.unanswerable_scores
    lines.append(
        f"  answerable    min {a[0]:.2f}  median {statistics.median(a):.2f}  max {a[-1]:.2f}"
    )
    lines.append(
        f"  unanswerable  min {u[0]:.2f}  median {statistics.median(u):.2f}  max {u[-1]:.2f}"
    )
    lines.append("  threshold   answerable kept   unanswerable refused")
    for p in gate.sweep:
        marker = "  <- current" if abs(p.threshold - run.min_score) < 1e-9 else ""
        lines.append(
            f"    {p.threshold:.2f}        {_pct(p.answerable_kept)}            "
            f"{_pct(p.unanswerable_refused)}{marker}"
        )
    note = (
        "scores overlap, so no threshold separates them perfectly"
        if gate.overlap
        else ("the two groups are fully separated")
    )
    lines.append(
        f"  best single threshold: {gate.best_threshold:.2f} "
        f"(balanced accuracy {_pct(gate.best_balanced_accuracy).strip()}; {note})"
    )
    lines.append(f"  current ANSWER_MIN_SCORE: {run.min_score:.2f}")
    return lines


def _answers_section(run: EvalRun) -> list[str]:
    if not run.answers:
        return []
    lines = [f"ANSWERS  (model {run.llm_model}, {len(run.answers)} questions)"]
    by_type: dict[str, list[bool]] = defaultdict(list)
    for outcome in run.answers:
        by_type[outcome.question.type].append(outcome.passed)
    labels = {
        "answerable": "answered correctly with a valid citation",
        "unanswerable": "correctly refused",
        "injection": "answered correctly and ignored the hidden instruction",
    }
    for kind in ("answerable", "unanswerable", "injection"):
        passed = by_type.get(kind)
        if passed:
            lines.append(
                f"  {kind:<13} {sum(passed)}/{len(passed)} {labels[kind]}"
                f"  ({_pct(sum(passed) / len(passed)).strip()})"
            )
    seconds = [o.seconds for o in run.answers]
    lines.append(
        f"  time per question: median {statistics.median(seconds):.1f}s, max {max(seconds):.1f}s"
    )
    refusals = defaultdict(int)
    for outcome in run.answers:
        if outcome.answer is not None and not outcome.answer.grounded:
            refusals[outcome.answer.reason] += 1
    if refusals:
        lines.append(
            "  refusals by reason: " + ", ".join(f"{k}={v}" for k, v in sorted(refusals.items()))
        )
    for outcome in run.answers:
        if not outcome.passed:
            lines.append(f"  FAIL  {outcome.question.id}: " + "; ".join(outcome.problems))
            if outcome.answer is not None:
                lines.append(f"        said: {' '.join(outcome.answer.text.split())[:160]!r}")
    return lines


def format_report(run: EvalRun) -> str:
    header = [
        "EVALUATION",
        f"  corpus: {run.documents} documents, {run.chunks} chunks"
        + (f", {run.ingest_failures} failed to ingest" if run.ingest_failures else ""),
        f"  embedding model: {run.embedding_model}",
        f"  chunk size {run.chunk_size}, overlap {run.chunk_overlap}, top_k {run.top_k}",
    ]
    sections = [header, _retrieval_section(run), _gate_section(run), _answers_section(run)]
    return "\n\n".join("\n".join(section) for section in sections if section)


def to_dict(run: EvalRun) -> dict[str, Any]:
    gate = analyse_gate(run.retrieval)
    return {
        "settings": {
            "embedding_model": run.embedding_model,
            "llm_model": run.llm_model,
            "chunk_size": run.chunk_size,
            "chunk_overlap": run.chunk_overlap,
            "top_k": run.top_k,
            "answer_min_score": run.min_score,
        },
        "corpus": {
            "documents": run.documents,
            "chunks": run.chunks,
            "ingest_failures": run.ingest_failures,
        },
        "retrieval": {
            "hit_rate": {f"hit@{c}": hit_rate(run.retrieval, c) for c in RETRIEVAL_CUTOFFS},
            "mrr": mean_reciprocal_rank(run.retrieval),
            "questions": [
                {
                    "id": o.question.id,
                    "type": o.question.type,
                    "first_hit_rank": o.first_hit_rank,
                    "top_score": o.top_score,
                    "top_results": [
                        {"source": r.source, "heading": r.heading_path, "score": round(r.score, 4)}
                        for r in o.results[:3]
                    ],
                }
                for o in run.retrieval
            ],
        },
        "gate": None
        if gate is None
        else {
            "best_threshold": gate.best_threshold,
            "best_balanced_accuracy": gate.best_balanced_accuracy,
            "sweep": [vars(p) for p in gate.sweep],
        },
        "answers": [
            {
                "id": a.question.id,
                "type": a.question.type,
                "passed": a.passed,
                "problems": a.problems,
                "reason": a.answer.reason if a.answer else None,
                "seconds": round(a.seconds, 2),
            }
            for a in run.answers
        ],
    }
