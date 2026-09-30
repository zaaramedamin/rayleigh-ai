from fastapi import APIRouter
from pydantic import BaseModel

from app.knowledge.ingestion.file_types import FILE_TYPES

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


class FileTypeInfo(BaseModel):
    name: str
    extensions: list[str]
    description: str


@router.get("/file-types")
def list_file_types() -> list[FileTypeInfo]:
    """File types the ingestion pipeline can read."""
    return [
        FileTypeInfo(name=ft.name, extensions=list(ft.extensions), description=ft.description)
        for ft in FILE_TYPES
    ]
