from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from models.retrieval_chunk import RetrievalChunk
from rag.chunker import Chunker


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

    def __init__(self, chunker: Chunker | None = None) -> None:
        self.chunker = chunker or Chunker()

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

            language = self._infer_language(file_path)
            chunks.extend(
                self.chunker.chunk_text(
                    text=text,
                    file_path=file_path,
                    language=language,
                )
            )

        return LoadedChunkBundle(chunks=chunks)

    def load_from_root(self, repo_root: str) -> LoadedChunkBundle:
        """
        Load chunks from every file under a repository root.
        """
        root_path = Path(repo_root)
        files = [
            str(path.relative_to(root_path))
            for path in root_path.rglob("*")
            if path.is_file()
        ]
        return self.load_from_files(repo_root, files)

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