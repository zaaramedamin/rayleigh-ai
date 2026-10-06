"""Vector storage behind a small interface, implemented with embedded (local-mode) Qdrant.

Embedded Qdrant runs inside this Python process and persists under DATA_DIR/qdrant, so no
server or Docker is needed. Only one process can open that folder at a time.

Points carry only ids and filterable metadata. The chunk text and its provenance stay in
SQLite, which remains the source of truth for citations.
"""

import math
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self, cast

from qdrant_client import QdrantClient, models

# Fixed namespace: a chunk (document_id, chunk_index) always maps to the same point id.
_POINT_NAMESPACE = uuid.UUID("5f2b6c1e-9a47-4d8e-b8a3-2c6f0e7d9b14")


class VectorStoreError(RuntimeError):
    """The vector store cannot be used as configured."""


class VectorStoreBusyError(VectorStoreError):
    """Another process (e.g. the API server or an ingestion run) has the local store open."""


@dataclass(frozen=True)
class VectorPoint:
    document_id: int
    chunk_index: int
    vector: Sequence[float]
    text_hash: str  # fingerprint of the chunk text that was embedded
    file_type: str  # e.g. ".md"


@dataclass(frozen=True)
class VectorHit:
    document_id: int
    chunk_index: int
    text_hash: str
    score: float  # cosine similarity, higher is more similar


class VectorStore(Protocol):
    def upsert(self, points: Sequence[VectorPoint]) -> None: ...

    def delete_document(self, document_id: int) -> None: ...

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        document_ids: Sequence[int] | None = None,
        file_types: Sequence[str] | None = None,
        exclude_document_ids: Sequence[int] | None = None,
    ) -> list[VectorHit]: ...

    def similarities(
        self, vector: Sequence[float], keys: Sequence[tuple[int, int]]
    ) -> dict[tuple[int, int], float]: ...

    def count(self, document_id: int | None = None) -> int: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


def collection_name(model_name: str) -> str:
    """One collection per embedding model, so vectors from different models never mix."""
    return "chunks_" + re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def point_id(document_id: int, chunk_index: int) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{document_id}:{chunk_index}"))


def normalize_file_type(file_type: str) -> str:
    """'MD', 'md' and '.md' all mean '.md'."""
    file_type = file_type.strip().lower()
    return file_type if file_type.startswith(".") else f".{file_type}"


def _open_local_client(path: Path) -> QdrantClient:
    path.mkdir(parents=True, exist_ok=True)
    try:
        return QdrantClient(path=str(path))
    except RuntimeError as exc:
        if "already accessed" in str(exc):
            raise VectorStoreBusyError(
                "the local vector store is in use by another process "
                "(stop the API server or wait for the running command, then retry)"
            ) from exc
        raise


class QdrantVectorStore:
    def __init__(self, client: QdrantClient, collection: str, dimension: int) -> None:
        self._client = client
        self._collection = collection
        self._dimension = dimension
        try:
            self._ensure_collection()
        except BaseException:
            client.close()
            raise

    @classmethod
    def open_local(cls, path: Path, collection: str, dimension: int) -> Self:
        """Embedded Qdrant persisted under `path`."""
        return cls(_open_local_client(path), collection, dimension)

    @classmethod
    def in_memory(cls, collection: str, dimension: int) -> Self:
        """Non-persistent store, for tests."""
        return cls(QdrantClient(location=":memory:"), collection, dimension)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _ensure_collection(self) -> None:
        if self._client.collection_exists(self._collection):
            params = self._client.get_collection(self._collection).config.params.vectors
            size = params.size if isinstance(params, models.VectorParams) else None
            if size != self._dimension:
                raise VectorStoreError(
                    f"collection {self._collection!r} holds {size}-dimensional vectors, "
                    f"but the model produces {self._dimension}; run `python -m app index --rebuild`"
                )
            return
        self._client.create_collection(
            self._collection,
            vectors_config=models.VectorParams(
                size=self._dimension, distance=models.Distance.COSINE
            ),
        )

    @staticmethod
    def _document_filter(document_id: int) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))
            ]
        )

    def upsert(self, points: Sequence[VectorPoint]) -> None:
        if not points:
            return
        self._client.upsert(
            self._collection,
            points=[
                models.PointStruct(
                    id=point_id(p.document_id, p.chunk_index),
                    vector=list(p.vector),
                    payload={
                        "document_id": p.document_id,
                        "chunk_index": p.chunk_index,
                        "text_hash": p.text_hash,
                        "file_type": normalize_file_type(p.file_type),
                    },
                )
                for p in points
            ],
            wait=True,
        )

    def delete_document(self, document_id: int) -> None:
        self._client.delete(
            self._collection,
            points_selector=models.FilterSelector(filter=self._document_filter(document_id)),
            wait=True,
        )

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        document_ids: Sequence[int] | None = None,
        file_types: Sequence[str] | None = None,
        exclude_document_ids: Sequence[int] | None = None,
    ) -> list[VectorHit]:
        """Most similar points first. Empty or None filters mean "no filter"."""
        conditions: list[models.Condition] = []
        excluded: list[models.Condition] = []
        if exclude_document_ids:
            excluded.append(
                models.FieldCondition(
                    key="document_id", match=models.MatchAny(any=list(exclude_document_ids))
                )
            )
        if document_ids:
            conditions.append(
                models.FieldCondition(
                    key="document_id", match=models.MatchAny(any=list(document_ids))
                )
            )
        if file_types:
            conditions.append(
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchAny(any=[normalize_file_type(t) for t in file_types]),
                )
            )
        response = self._client.query_points(
            self._collection,
            query=list(vector),
            limit=top_k,
            query_filter=(
                models.Filter(must=conditions or None, must_not=excluded or None)
                if conditions or excluded
                else None
            ),
            with_payload=True,
        )
        return [
            VectorHit(
                document_id=int(point.payload["document_id"]),
                chunk_index=int(point.payload["chunk_index"]),
                text_hash=str(point.payload["text_hash"]),
                score=float(point.score),
            )
            for point in response.points
            if point.payload is not None
        ]

    def similarities(
        self, vector: Sequence[float], keys: Sequence[tuple[int, int]]
    ) -> dict[tuple[int, int], float]:
        """Cosine similarity of `vector` to the stored vector of each (document_id, chunk_index).

        A chunk with no stored vector is simply left out of the answer."""
        if not keys:
            return {}
        points = self._client.retrieve(
            self._collection,
            ids=[point_id(document_id, chunk_index) for document_id, chunk_index in keys],
            with_payload=True,
            with_vectors=True,
        )
        found: dict[tuple[int, int], float] = {}
        norm = math.sqrt(sum(x * x for x in vector))
        for point in points:
            if point.payload is None or not isinstance(point.vector, list):
                continue
            stored = cast(list[float], point.vector)  # one unnamed vector per point
            stored_norm = math.sqrt(sum(x * x for x in stored))
            if not norm or not stored_norm:
                continue
            dot = sum(a * b for a, b in zip(vector, stored, strict=True))
            key = (int(point.payload["document_id"]), int(point.payload["chunk_index"]))
            found[key] = dot / (norm * stored_norm)
        return found

    def count(self, document_id: int | None = None) -> int:
        count_filter = self._document_filter(document_id) if document_id is not None else None
        return self._client.count(self._collection, count_filter=count_filter, exact=True).count

    def reset(self) -> None:
        """Delete every vector in this collection.

        The points are deleted, not the collection: in Qdrant's on-disk mode, deleting a
        collection and creating it again under the same name brings the old points back.
        """
        while True:
            points, _ = self._client.scroll(
                self._collection, limit=1000, with_payload=False, with_vectors=False
            )
            if not points:
                return
            self._client.delete(
                self._collection,
                points_selector=models.PointIdsList(points=[p.id for p in points]),
                wait=True,
            )

    def close(self) -> None:
        self._client.close()


def count_local_vectors(path: Path, collection: str) -> int:
    """Number of vectors in a local collection, without knowing the model dimension."""
    if not path.exists():
        return 0
    client = _open_local_client(path)
    try:
        if not client.collection_exists(collection):
            return 0
        return client.count(collection, exact=True).count
    finally:
        client.close()
