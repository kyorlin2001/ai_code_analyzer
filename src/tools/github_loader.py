from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, BadZipFile

import requests

from tools.repo_loader import RepositoryData, build_file_tree, list_repository_files


@dataclass
class ExtractedGitHubRepository:
    data: RepositoryData
    temp_dir: TemporaryDirectory


def load_repository_from_github(
    owner: str,
    repo: str,
    branch: str = "main",
    token: str | None = None,
) -> ExtractedGitHubRepository:
    """
    Download a GitHub repository archive and normalize it into RepositoryData.
    """
    archive_url = f"https://api.github.com/repos/{owner}/{repo}/zipball/{branch}"

    headers = {
        "Accept": "application/vnd.github+json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(archive_url, headers=headers, timeout=60)
    response.raise_for_status()

    temp_dir = TemporaryDirectory(prefix="agentic_code_analysis_github_")
    temp_root = Path(temp_dir.name)
    zip_path = temp_root / f"{repo}.zip"
    zip_path.write_bytes(response.content)

    try:
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_root)
    except BadZipFile as exc:
        temp_dir.cleanup()
        raise ValueError("GitHub archive could not be extracted.") from exc

    extracted_root = _find_extracted_root(temp_root)

    repo_data = RepositoryData(
        repo_name=repo,
        file_tree=build_file_tree(extracted_root),
        files=list_repository_files(extracted_root),
        root_dir=str(extracted_root),
    )

    return ExtractedGitHubRepository(data=repo_data, temp_dir=temp_dir)


def _find_extracted_root(temp_root: Path) -> Path:
    entries = [p for p in temp_root.iterdir() if p.name != "__MACOSX"]

    dirs = [p for p in entries if p.is_dir()]
    files = [p for p in entries if p.is_file()]

    if len(dirs) == 1 and not files:
        return dirs[0]

    return temp_root