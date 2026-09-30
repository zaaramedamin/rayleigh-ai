import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from app.knowledge.ingestion.file_types import file_type_for

logger = logging.getLogger(__name__)

IGNORED_DIR_NAMES = frozenset({"node_modules", "__pycache__"})
NO_EXTENSION = "(none)"


@dataclass
class ScanResult:
    files: list[Path] = field(default_factory=list)
    folders_missing: int = 0
    skipped_outside_allowlist: int = 0
    skipped_unsupported: int = 0
    # Extension -> count of skipped files. Extensions only; never file names.
    unsupported_by_extension: Counter[str] = field(default_factory=Counter)


def resolve_roots(allowed_folders: list[Path]) -> list[Path]:
    """Resolve the allow-list once, collapsing `..` and symlinks."""
    return [folder.resolve() for folder in allowed_folders]


def is_within_allowed(path: Path, roots: list[Path]) -> bool:
    """True only if `path`, fully resolved (symlinks, `..`), is inside an allowed root."""
    resolved = path.resolve()
    return any(resolved.is_relative_to(root) for root in roots)


def _is_hidden_or_ignored(name: str) -> bool:
    return name.startswith(".") or name in IGNORED_DIR_NAMES


def scan_allowed_folders(allowed_folders: list[Path]) -> ScanResult:
    """List ingestible files under the allow-listed folders, and only those.

    Entries (files or folders) that resolve outside their allowed root, for example through a
    symlink or junction, are skipped. Allow-listed folders that don't exist are skipped, never
    created.
    """
    result = ScanResult()
    roots = resolve_roots(allowed_folders)
    seen: set[Path] = set()

    for root in roots:
        if not root.is_dir():
            logger.warning("allowed folder missing or not a directory")
            result.folders_missing += 1
            continue

        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            kept_dirs = []
            for name in dirnames:
                if _is_hidden_or_ignored(name):
                    continue
                if not is_within_allowed(Path(dirpath, name), roots):
                    result.skipped_outside_allowlist += 1
                    continue
                kept_dirs.append(name)
            dirnames[:] = kept_dirs

            for name in filenames:
                path = Path(dirpath, name)
                if name.startswith("."):
                    continue
                if file_type_for(path.suffix) is None:
                    result.skipped_unsupported += 1
                    result.unsupported_by_extension[path.suffix.lower() or NO_EXTENSION] += 1
                    continue
                if not is_within_allowed(path, roots):
                    result.skipped_outside_allowlist += 1
                    continue
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    result.files.append(resolved)

    return result
