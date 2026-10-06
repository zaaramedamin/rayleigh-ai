"""Keyword search with SQLite FTS5, for exact things: codes, names, numbers, rare words.

Meaning search finds notes about a topic but can miss an exact string such as "INV-2026-0418" or a
surname. A keyword index finds them.

The index lives **in memory only**. Text in an encrypted library must never sit in plaintext on
disk, and an FTS5 table stores every word it indexes; so the index is built from the chunks (read
and decrypted through the normal columns) and thrown away with the process. It is rebuilt when the
library changes, found by a cheap fingerprint of the active chunks. Nothing is written anywhere.

User text is never passed to FTS5 as its own query language. The question is cut into plain words,
each word is quoted, and the words are joined with OR, so operators, quotes and column filters in a
question are just words.
"""

import logging
import re
import sqlite3
import threading
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.storage.models import DOC_ACTIVE, Chunk, Document
from app.storage.vector_store import normalize_file_type

logger = logging.getLogger(__name__)

MAX_TERMS = 20
INSERT_BATCH = 1000
# A match in a heading counts for more than one in the body.
HEADING_WEIGHT = 3.0

# Words that say nothing about which note is meant. English only, as the notes mostly are; a word
# in another language is simply an ordinary term.
STOP_WORDS = frozenset(
    """a about after all also am an and any are as at be been before but by can could did do does
    doing for from had has have he her his how i if in into is it its just me my not of on or our
    she should so than that the their them then there these they this to too was we were what when
    where which who whom why will with would you your""".split()
)
# Letters and digits only, the same split SQLite makes: "snake_case" is two words, "5503" one.
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


class KeywordSearchUnavailable(RuntimeError):
    """This SQLite has no FTS5, so keyword search cannot be used (meaning search still works)."""


@dataclass(frozen=True)
class KeywordHit:
    document_id: int
    chunk_index: int
    score: float  # BM25, higher is a better match; not comparable with a similarity score
    # The share (0 to 1) of the question's words that this chunk contains.
    coverage: float = 1.0


def _fold(text: str) -> str:
    """Lower case, with accents removed: how a word is compared, as in the index."""
    decomposed = unicodedata.normalize("NFKD", text).lower()
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def query_terms(query: str) -> list[str]:
    """The words of a question that can help find a note, in order, without repeats."""
    terms: list[str] = []
    for word in _WORD.findall(unicodedata.normalize("NFKC", query).lower()):
        if (len(word) > 1 or word.isdigit()) and word not in STOP_WORDS and word not in terms:
            terms.append(word)
    return terms[:MAX_TERMS]


def match_expression(terms: Sequence[str]) -> str:
    """An FTS5 query for these words: each one quoted (so it is only ever a word), joined by OR."""
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)


class KeywordIndex:
    """An FTS5 table in memory, over the chunks of current documents."""

    def __init__(self, fingerprint: tuple[int, int, int]) -> None:
        self.fingerprint = fingerprint
        self._connection = sqlite3.connect(":memory:", check_same_thread=False)
        try:
            self._connection.execute(
                "CREATE VIRTUAL TABLE chunks_fts USING fts5("
                "heading, body, document_id UNINDEXED, chunk_index UNINDEXED, "
                "file_type UNINDEXED, tokenize = 'unicode61 remove_diacritics 2')"
            )
        except sqlite3.OperationalError as exc:
            self._connection.close()
            raise KeywordSearchUnavailable("this SQLite was built without FTS5") from exc
        self.size = 0

    def add(self, rows: Sequence[tuple[str, str, int, int, str]]) -> None:
        self._connection.executemany(
            "INSERT INTO chunks_fts (heading, body, document_id, chunk_index, file_type) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self.size += len(rows)

    def search(
        self,
        terms: Sequence[str],
        *,
        limit: int,
        document_ids: Sequence[int] | None = None,
        file_types: Sequence[str] | None = None,
    ) -> list[KeywordHit]:
        if not terms or limit <= 0:
            return []
        sql = (
            "SELECT document_id, chunk_index, "
            "-bm25(chunks_fts, ?, 1.0, 0.0, 0.0, 0.0) AS score, heading, body "
            "FROM chunks_fts WHERE chunks_fts MATCH ?"
        )
        parameters: list[object] = [HEADING_WEIGHT, match_expression(terms)]
        if document_ids:
            sql += f" AND document_id IN ({','.join('?' * len(document_ids))})"
            parameters += [int(i) for i in document_ids]
        if file_types:
            sql += f" AND file_type IN ({','.join('?' * len(file_types))})"
            parameters += [normalize_file_type(t) for t in file_types]  # 'TXT', 'txt', '.txt'
        sql += " ORDER BY score DESC, document_id, chunk_index LIMIT ?"
        parameters.append(limit)
        try:
            rows = self._connection.execute(sql, parameters).fetchall()
        except sqlite3.OperationalError:
            return []  # a query FTS5 cannot run is "no match", never an error for the person
        wanted = [_fold(term) for term in terms]
        hits = []
        for document_id, chunk_index, score, heading, body in rows:
            present = set(_WORD.findall(_fold(f"{heading} {body}")))
            coverage = sum(term in present for term in wanted) / len(wanted)
            hits.append(KeywordHit(int(document_id), int(chunk_index), float(score), coverage))
        return hits

    def close(self) -> None:
        self._connection.close()


# --- building and caching -----------------------------------------------------------------------

_LOCK = threading.RLock()
_INDEXES: dict[str, KeywordIndex] = {}


def fingerprint(session: Session) -> tuple[int, int, int]:
    """Changes whenever a chunk of a current document is added, removed or replaced."""
    row = session.execute(
        select(
            func.count(Chunk.id),
            func.coalesce(func.max(Chunk.id), 0),
            func.coalesce(func.sum(Chunk.id), 0),
        )
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.status == DOC_ACTIVE)
    ).one()
    return int(row[0]), int(row[1]), int(row[2])


def _file_type(name: str) -> str:
    return Path(name).suffix.lower()


def _build(session: Session, current: tuple[int, int, int]) -> KeywordIndex:
    index = KeywordIndex(current)
    batch: list[tuple[str, str, int, int, str]] = []
    rows = session.execute(
        select(
            Chunk.heading_path,
            Chunk.text,
            Chunk.document_id,
            Chunk.chunk_index,
            Document.original_filename,
        )
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.status == DOC_ACTIVE)
        .execution_options(yield_per=INSERT_BATCH)
    )
    for heading, body, document_id, chunk_index, name in rows:
        batch.append((heading, body, document_id, chunk_index, _file_type(name)))
        if len(batch) >= INSERT_BATCH:
            index.add(batch)
            batch = []
    if batch:
        index.add(batch)
    logger.info("keyword index built chunks=%d", index.size)  # a count, never any text
    return index


def index_for(session: Session) -> KeywordIndex:
    """The keyword index for this library, built now if the library changed since the last one."""
    key = str(session.get_bind().url)  # type: ignore[union-attr]
    current = fingerprint(session)
    with _LOCK:
        cached = _INDEXES.get(key)
        if cached is not None and cached.fingerprint == current:
            return cached
        fresh = _build(session, current)
        if cached is not None:
            cached.close()
        _INDEXES[key] = fresh
        return fresh


def forget_indexes() -> None:
    """Drop every index (tests; and a way to release the memory)."""
    with _LOCK:
        for index in _INDEXES.values():
            index.close()
        _INDEXES.clear()


def keyword_search(
    session: Session,
    query: str,
    *,
    top_k: int,
    document_ids: Sequence[int] | None = None,
    file_types: Sequence[str] | None = None,
) -> list[KeywordHit]:
    """The chunks that best match the words of `query`, best first. Raises
    KeywordSearchUnavailable if this SQLite has no FTS5."""
    terms = query_terms(query)
    if not terms:
        return []
    with _LOCK:
        index = index_for(session)
        return index.search(terms, limit=top_k, document_ids=document_ids, file_types=file_types)
