from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from zipfile import ZipFile, BadZipFile


SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".go", ".rs",
    ".cpp", ".c", ".h", ".hpp", ".cs", ".php", ".rb", ".swift",
    ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini"
}


@dataclass
class RepositoryData:
    repo_name: str
    file_tree: dict[str, Any]
    files: list[str]
    root_dir: str | None = None


@dataclass
class ExtractedRepository:
    """
    Represents a zip-based repository that has been extracted to disk.
    The TemporaryDirectory object must be kept alive while using the repo.
    """
    data: RepositoryData
    temp_dir: TemporaryDirectory


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def build_file_tree(root: Path) -> dict[str, Any]:
    """
    Build a simple nested directory tree representation.
    """
    tree: dict[str, Any] = {}

    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue

        if not is_text_file(path):
            continue

        relative_parts = path.relative_to(root).parts
        cursor = tree

        for part in relative_parts[:-1]:
            cursor = cursor.setdefault(part, {})

        cursor[relative_parts[-1]] = {
            "path": str(path.relative_to(root)),
            "name": path.name,
        }

    return tree


def list_repository_files(root: Path) -> list[str]:
    """
    Return a list of text file paths relative to the repo root.
    """
    files: list[str] = []

    for path in sorted(root.rglob("*")):
        if path.is_file() and is_text_file(path):
            files.append(str(path.relative_to(root)))

    return files


def load_repository(repo_path: str) -> RepositoryData:
    """
    Load a repository from disk and return a lightweight snapshot.

    Args:
        repo_path: Path to the repository root.

    Returns:
        RepositoryData containing repo name, file tree, and file list.

    Raises:
        FileNotFoundError: if the path does not
    """
    root = Path(repo_path).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Repository not found: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory: {root}")

    return RepositoryData(
        repo_name=root.name,
        file_tree=build_file_tree(root),
        files=list_repository_files(root),
        root_dir=str(root),
    )


def load_repository_from_zip(zip_bytes: bytes, zip_name: str = "repository.zip") -> ExtractedRepository:
    """
    Extract a zip archive into a temporary directory and return repo metadata.

    The returned TemporaryDirectory must stay in scope for as long as the
    extracted files are needed.
    """
    temp_dir = TemporaryDirectory(prefix="agentic_code_analysis_")
    temp_root = Path(temp_dir.name)
    zip_path = temp_root / zip_name
    zip_path.write_bytes(zip_bytes)

    try:
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_root)
    except BadZipFile as exc:
        temp_dir.cleanup()
        raise ValueError("The uploaded file is not a valid zip archive.") from exc

    extracted_root = _find_extracted_root(temp_root)
    repo_data = RepositoryData(
        repo_name=extracted_root.name,
        file_tree=build_file_tree(extracted_root),
        files=list_repository_files(extracted_root),
        root_dir=str(extracted_root),
    )

    return ExtractedRepository(data=repo_data, temp_dir=temp_dir)


def _find_extracted_root(temp_root: Path) -> Path:
    """
    Find the most likely repository root after extraction.

    If the zip contains a single top-level folder, use it.
    Otherwise, use the extraction directory itself.
    """
    entries = [p for p in temp_root.iterdir() if p.name != "__MACOSX" and p.name != temp_root.name]

    dirs = [p for p in entries if p.is_dir()]
    files = [p for p in entries if p.is_file()]

    if len(dirs) == 1 and not files:
        return dirs[0]

    return temp_root