import os
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import streamlit as st
from smolagents import InferenceClientModel


st.title("Agentic Code Analyst 🤖")
st.write("Step 1: Analyze a local codebase and ask questions about it.")


SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".go", ".rs",
    ".cpp", ".c", ".h", ".hpp", ".cs", ".php", ".rb", ".swift",
    ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini"
}


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def read_file_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 20):
    lines = text.splitlines()
    if not lines:
        return []

    chunks = []
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
        if end == len(lines):
            break
        start = max(end - overlap, start + 1)

    return chunks


def scan_repository(root_dir: str):
    root = Path(root_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return [], f"Folder not found: {root}"

    docs = []
    for path in root.rglob("*"):
        if path.is_file() and is_text_file(path):
            text = read_file_text(path)
            if not text.strip():
                continue

            file_chunks = chunk_text(text)
            for i, chunk in enumerate(file_chunks, start=1):
                docs.append(
                    {
                        "path": str(path.relative_to(root)),
                        "chunk_id": i,
                        "text": chunk["text"],
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                    }
                )

    return docs, None


def scan_zip_repository(uploaded_zip):
    temp_dir = Path(tempfile.mkdtemp(prefix="code_analysis_zip_"))
    extract_dir = temp_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        zip_path = temp_dir / uploaded_zip.name
        zip_path.write_bytes(uploaded_zip.getbuffer())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
    except zipfile.BadZipFile:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return [], None, "The uploaded file is not a valid zip archive."
    except Exception as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return [], None, f"Failed to process zip file: {exc}"

    docs = []
    for path in extract_dir.rglob("*"):
        if path.is_file() and is_text_file(path):
            text = read_file_text(path)
            if not text.strip():
                continue

            file_chunks = chunk_text(text)
            for i, chunk in enumerate(file_chunks, start=1):
                docs.append(
                    {
                        "path": str(path.relative_to(extract_dir)),
                        "chunk_id": i,
                        "text": chunk["text"],
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                    }
                )

    return docs, str(temp_dir), None


def tokenize(text: str):
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower())


def score_chunk(query: str, chunk_text_value: str) -> float:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    chunk_tokens = tokenize(chunk_text_value)
    if not chunk_tokens:
        return 0.0

    q = Counter(query_tokens)
    c = Counter(chunk_tokens)

    score = 0
    for token, count in q.items():
        score += min(count, c.get(token, 0))

    lower_chunk = chunk_text_value.lower()
    substring_bonus = sum(1 for token in set(query_tokens) if token in lower_chunk)
    return float(score + (0.25 * substring_bonus))


def retrieve(query: str, docs, top_k: int = 5):
    scored = []
    for doc in docs:
        s = score_chunk(query, doc["text"])
        if s > 0:
            scored.append((s, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def build_context(query: str, docs):
    hits = retrieve(query, docs, top_k=5)
    if not hits:
        return "", []

    context_blocks = []
    results = []

    for score, doc in hits:
        results.append(doc)
        context_blocks.append(
            f"FILE: {doc['path']}\n"
            f"LINES: {doc['start_line']}-{doc['end_line']}\n"
            f"SCORE: {score:.2f}\n"
            f"CODE:\n{doc['text']}\n"
        )

    return "\n---\n".join(context_blocks), results


def get_model():
    token = os.getenv("HF_TOKEN")
    return InferenceClientModel(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        token=token,
    )


def synthesize_answer(query: str, context: str):
    if not context.strip():
        return "I couldn't find anything relevant in the indexed files."

    model = get_model()

    prompt = f"""
You are an expert code analysis assistant.

Answer the user's question using ONLY the provided code context.
If the answer is uncertain, say so.
Cite the relevant files and line ranges in your response.

User question:
{query}

Code context:
{context}

Write a concise, useful answer with bullet points and citations.
""".strip()

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    return model.generate(messages)


if "docs" not in st.session_state:
    st.session_state.docs = []
if "repo_path" not in st.session_state:
    st.session_state.repo_path = ""
if "temp_zip_dir" not in st.session_state:
    st.session_state.temp_zip_dir = None


st.markdown("### Load a repository")

input_mode = st.radio(
    "Choose input type",
    ["Local folder path", "Zip file upload"],
    horizontal=True,
)

repo_path = ""
uploaded_zip = None

if input_mode == "Local folder path":
    repo_path = st.text_input(
        "Local repository folder path",
        value=st.session_state.repo_path,
        placeholder=r"C:\path\to\your\repo"
    )
else:
    st.write("Upload a zip file containing your codebase:")
    uploaded_zip = st.file_uploader(
        "Browse zip file",
        type=["zip"],
        accept_multiple_files=False,
    )

col1, col2 = st.columns([1, 1])

with col1:
    index_button = st.button("Index repository")

with col2:
    clear_button = st.button("Clear index")

if clear_button:
    st.session_state.docs = []
    st.session_state.repo_path = ""
    if st.session_state.temp_zip_dir:
        shutil.rmtree(st.session_state.temp_zip_dir, ignore_errors=True)
        st.session_state.temp_zip_dir = None
    st.success("Index cleared.")

if index_button:
    if input_mode == "Local folder path":
        if not repo_path.strip():
            st.error("Please enter a folder path.")
        else:
            with st.spinner("Scanning repository..."):
                docs, error = scan_repository(repo_path)
                if error:
                    st.error(error)
                else:
                    if st.session_state.temp_zip_dir:
                        shutil.rmtree(st.session_state.temp_zip_dir, ignore_errors=True)
                        st.session_state.temp_zip_dir = None
                    st.session_state.docs = docs
                    st.session_state.repo_path = repo_path
                    st.success(f"Indexed {len(docs)} chunks from the repository.")
    else:
        if uploaded_zip is None:
            st.error("Please upload a zip file.")
        else:
            with st.spinner("Scanning zip file..."):
                docs, temp_dir, error = scan_zip_repository(uploaded_zip)
                if error:
                    st.error(error)
                else:
                    if st.session_state.temp_zip_dir:
                        shutil.rmtree(st.session_state.temp_zip_dir, ignore_errors=True)
                    st.session_state.temp_zip_dir = temp_dir
                    st.session_state.docs = docs
                    st.success(f"Indexed {len(docs)} chunks from the zip file.")

if st.session_state.docs:
    st.info(f"Indexed chunks: {len(st.session_state.docs)}")

    query = st.text_input("Ask a question about the codebase")
    ask_button = st.button("Search")

    if ask_button and query.strip():
        with st.spinner("Searching..."):
            context, results = build_context(query, st.session_state.docs)
            answer = synthesize_answer(query, context)
            st.markdown(answer)

            st.subheader("Evidence")
            for doc in results:
                st.markdown(
                    f"**File:** `{doc['path']}`  \n"
                    f"**Lines:** {doc['start_line']}-{doc['end_line']}  \n"
                )
                st.code(doc["text"][:2000], language="python")
else:
    st.write("Index a repository to begin asking questions.")