import sys

from sqlalchemy import inspect
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.knowledge.ingestion.service import IngestSummary, ingest_folders
from app.storage.database import create_db_engine


def _print_summary(summary: IngestSummary) -> None:
    print(f"added:                     {summary.added}")
    print(f"unchanged:                 {summary.unchanged}")
    print(f"skipped (empty):           {summary.skipped_empty}")
    print(f"skipped (too large):       {summary.skipped_too_large}")
    print(f"skipped (unsupported):     {summary.skipped_unsupported}")
    print(f"skipped (outside allowed): {summary.skipped_outside_allowlist}")
    print(f"allowed folders missing:   {summary.folders_missing}")
    failures = ", ".join(f"{reason}={count}" for reason, count in sorted(summary.failed.items()))
    print(f"failed:                    {sum(summary.failed.values())} {failures}".rstrip())


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)

    if not settings.allowed_folders:
        print("ALLOWED_FOLDERS is empty. Nothing to ingest. Set it in .env.")
        return 0

    engine = create_db_engine(settings.data_dir)
    if "documents" not in inspect(engine).get_table_names():
        print("Database not initialised. Run `alembic upgrade head` from backend/ first.")
        return 1

    with Session(engine) as session:
        summary = ingest_folders(
            session,
            settings.data_dir,
            settings.allowed_folders,
            settings.max_file_size_mb * 1024 * 1024,
        )
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
