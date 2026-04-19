from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from models.retrieval_chunk import RetrievalChunk


@dataclass(frozen=True)
class LoadedChunkBundle:
    """
    Container for chunks loaded from a repository.
    """

    chunks: list[RetrievalChunk]


class RepoChunkLoader:
    """
    Loads repository files and converts them into retrieval chunks.
    """

    def __init__(
        self,
        chunk_size_lines: int = 80,
        overlap_lines: int = 15,
        max_file_lines: int = 2_000,
    ) -> None:
        self.chunk_size_lines = chunk_size_lines
        self.overlap_lines = overlap_lines
        self.max_file_lines = max_file_lines

    def load_from_files(self, repo_root: str, files: list[str]) -> LoadedChunkBundle:
        """
        Load chunks from a repository root and a list of relative file paths.
        """
        root_path = Path(repo_root)
        chunks: list[RetrievalChunk] = []

        for file_path in files:
            absolute_path = root_path / file_path
            if not absolute_path.is_file():
                continue

            try:
                text = absolute_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            except OSError:
                continue

            file_chunks = self._chunk_text(
                text=text,
                file_path=file_path,
                language=self._infer_language(file_path),
            )
            chunks.extend(file_chunks)

        return LoadedChunkBundle(chunks=chunks)

    def load_from_root(self, repo_root: str) -> LoadedChunkBundle:
        """
        Load chunks from every supported text file under a repository root.
        """
        root_path = Path(repo_root)
        files = [
            str(path.relative_to(root_path))
            for path in root_path.rglob("*")
            if path.is_file()
        ]
        return self.load_from_files(repo_root, files)

    def _chunk_text(
        self,
        text: str,
        file_path: str,
        language: str | None = None,
    ) -> list[RetrievalChunk]:
        lines = text.splitlines()
        if not lines:
            return []

        if len(lines) > self.max_file_lines:
            lines = lines[: self.max_file_lines]

        chunks: list[RetrievalChunk] = []
        step = max(1, self.chunk_size_lines - self.overlap_lines)

        start = 0
        chunk_index = 0
        while start < len(lines):
            end = min(len(lines), start + self.chunk_size_lines)
            chunk_lines = lines[start:end]

            if not chunk_lines:
                break

            chunk_text = "\n".join(chunk_lines).strip()
            if chunk_text:
                chunk_id = f"{file_path}::chunk-{chunk_index}"
                chunks.append(
                    RetrievalChunk(
                        text=chunk_text,
                        file_path=file_path,
                        chunk_id=chunk_id,
                        language=language,
                        start_line=start + 1,
                        end_line=end,
                    )
                )

            chunk_index += 1
            start += step

        return chunks

    def _infer_language(self, file_path: str) -> str | None:
        suffix = Path(file_path).suffix.lower()

        language_map = {
            ".py": "Python",
            ".js": "JavaScript",
            ".jsx": "JavaScript React",
            ".ts": "TypeScript",
            ".tsx": "TypeScript React",
            ".java": "Java",
            ".kt": "Kotlin",
            ".go": "Go",
            ".rs": "Rust",
            ".md": "Markdown",
            ".txt": "Text",
            ".json": "JSON",
            ".yaml": "YAML",
            ".yml": "YAML",
            ".toml": "TOML",
            ".ini": "INI",
        }

        return language_map.get(suffix)