from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMPORT_PATTERNS = {
    ".py": re.compile(r"^\s*(?:from\s+\S+\s+import\s+\S+|import\s+\S+)", re.MULTILINE),
    ".js": re.compile(r"^\s*import\s+.*|^\s*const\s+\w+\s*=\s*require\(", re.MULTILINE),
    ".ts": re.compile(r"^\s*import\s+.*|^\s*const\s+\w+\s*=\s*require\(", re.MULTILINE),
    ".jsx": re.compile(r"^\s*import\s+.*|^\s*const\s+\w+\s*=\s*require\(", re.MULTILINE),
    ".tsx": re.compile(r"^\s*import\s+.*|^\s*const\s+\w+\s*=\s*require\(", re.MULTILINE),
}


@dataclass
class ParsedCodeInfo:
    path: str
    language: str
    line_count: int
    imports: list[str]
    todos: list[str]
    classes: list[str]
    functions: list[str]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def detect_language(path: str) -> str:
    suffix = Path(path).suffix.lower()

    mapping = {
        ".py": "Python",
        ".js": "JavaScript",
        ".jsx": "JavaScript React",
        ".ts": "TypeScript",
        ".tsx": "TypeScript React",
        ".java": "Java",
        ".kt": "Kotlin",
        ".go": "Go",
        ".rs": "Rust",
        ".c": "C",
        ".cpp": "C++",
        ".h": "C/C++ Header",
        ".hpp": "C++ Header",
        ".cs": "C#",
        ".php": "PHP",
        ".rb": "Ruby",
        ".swift": "Swift",
    }

    return mapping.get(suffix, "Unknown")


def extract_imports(text: str, path: str) -> list[str]:
    suffix = Path(path).suffix.lower()
    pattern = IMPORT_PATTERNS.get(suffix)
    if not pattern:
        return []

    matches = pattern.findall(text)
    return [m.strip() for m in matches if m.strip()]


def extract_todos(text: str) -> list[str]:
    todos: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if "TODO" in upper or "FIXME" in upper or "XXX" in upper:
            todos.append(stripped)
    return todos


def extract_python_symbols(text: str) -> tuple[list[str], list[str]]:
    classes = re.findall(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.MULTILINE)
    functions = re.findall(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.MULTILINE)
    return classes, functions


def extract_js_symbols(text: str) -> tuple[list[str], list[str]]:
    classes = re.findall(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.MULTILINE)
    functions = re.findall(
        r"^\s*(?:function\s+([A-Za-z_][A-Za-z0-9_]*)|const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\()",
        text,
        re.MULTILINE,
    )

    flat_functions: list[str] = []
    for fn_a, fn_b in functions:
        name = fn_a or fn_b
        if name:
            flat_functions.append(name)

    return classes, flat_functions


def parse_code_file(path: str) -> ParsedCodeInfo:
    file_path = Path(path)
    text = read_text(file_path)
    language = detect_language(path)

    imports = extract_imports(text, path)
    todos = extract_todos(text)

    classes: list[str] = []
    functions: list[str] = []

    if file_path.suffix.lower() == ".py":
        classes, functions = extract_python_symbols(text)
    elif file_path.suffix.lower() in {".js", ".jsx", ".ts", ".tsx"}:
        classes, functions = extract_js_symbols(text)

    return ParsedCodeInfo(
        path=path,
        language=language,
        line_count=len(text.splitlines()),
        imports=imports,
        todos=todos,
        classes=classes,
        functions=functions,
    )


def parse_repository_files(files: list[str], root_dir: str) -> list[ParsedCodeInfo]:
    """
    Parse a list of repository-relative file paths into structured code metadata.
    """
    root = Path(root_dir).expanduser().resolve()
    parsed: list[ParsedCodeInfo] = []

    for rel_path in files:
        full_path = root / rel_path
        if full_path.is_file():
            parsed.append(parse_code_file(str(full_path)))

    return parsed