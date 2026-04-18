from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.tools.repo_loader import is_text_file


@dataclass
class IndexedChunk:
    """
    A chunk of text from a source file.
    """

    path: str
    chunk_id: int
    text: str
    start_line: int
    end_line: int


def read_file_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 20) -> list[dict[str, int | str]]:
    """
    Split text into overlapping line-based chunks.
    """
    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[dict[str, int | str]] = []
    start = 0

    while start < len(lines):
        end = min(start + chunk_size, len(lines))
        chunk_lines = lines[start:end]
        chunk_text_value = "\n".join(chunk_lines).strip()

        if chunk_text_value:
            chunks.append(
                {
                    "text": chunk_text_value,
                    "start_line": start + 1,
                    "end_line": end,
                }
            )