"""Find resume files in the input folder and detect duplicate files."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass(frozen=True)
class ResumeFile:
    path: Path
    file_hash: str
    duplicate_of: str | None = None   # name of the first file with this hash


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover(input_dir: Path) -> list[ResumeFile]:
    """Return every supported resume file, in a stable order.

    Byte-identical files are kept in the list but marked as duplicates so the
    batch summary can report them without processing them twice.
    """
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    files: list[ResumeFile] = []
    first_seen: dict[str, str] = {}
    for path in sorted(input_dir.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.name.startswith((".", "~$")):
            continue
        if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            continue
        try:
            digest = file_sha256(path)
        except OSError:
            digest = ""   # unreadable on disk; the parser will report it
        duplicate_of = first_seen.get(digest) if digest else None
        if digest and duplicate_of is None:
            first_seen[digest] = path.name
        files.append(ResumeFile(path=path, file_hash=digest, duplicate_of=duplicate_of))
    return files
