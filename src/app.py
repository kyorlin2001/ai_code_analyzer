from __future__ import annotations

import json

import streamlit as st

from orchestrator import AnalysisOrchestrator
from tools.github_loader import load_repository_from_github
from tools.repo_loader import load_repository_from_zip


st.set_page_config(page_title="Agentic Code Analyst", layout="wide")
st.title("Agentic Code Analyst 🤖")
st.write("Analyze a zip archive, or GitHub repository.")


if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "zip_temp_dir" not in st.session_state:
    st.session_state.zip_temp_dir = None
if "github_temp_dir" not in st.session_state:
    st.session_state.github_temp_dir = None
if "repo_root_path" not in st.session_state:
    st.session_state.repo_root_path = None


input_mode = st.radio(
    "Choose input type",
    ["Zip file upload", "GitHub repository"],
    horizontal=True,
)

repo_path = ""
uploaded_zip = None
github_owner = ""
github_repo = ""
github_branch = "main"
github_token = ""

if input_mode == "Local folder path":
    repo_path = st.text_input(
        "Local repository folder path",
        placeholder=r"C:\\path\\to\\your\\repo",
    )
elif input_mode == "Zip file upload":
    uploaded_zip = st.file_uploader(
        "Upload a zip file containing your codebase",
        type=["zip"],
        accept_multiple_files=False,
    )
else:
    github_owner = st.text_input("GitHub owner", placeholder="octocat")
    github_repo = st.text_input("GitHub repo", placeholder="hello-world")
    github_branch = st.text_input("Branch", value="main")
    github_token = st.text_input("GitHub token (optional)", type="password")

focus = st.selectbox(
    "Analysis focus",
    ["full", "architecture", "dependencies", "issues"],
    index=0,
)

enable_rag = st.checkbox("Enable RAG follow-up questions", value=True)

rag_question = ""
if enable_rag:
    rag_question = st.text_area(
        "Optional RAG question",
        placeholder="What should I improve in this repository?",
        height=100,
    )

baseline_file = st.file_uploader(
    "Optional baseline findings JSON for regression analysis",
    type=["json"],
    accept_multiple_files=False,
)

baseline_findings = []
if baseline_file is not None:
    try:
        baseline_findings = json.loads(baseline_file.getvalue().decode("utf-8"))
        if not isinstance(baseline_findings, list):
            st.warning("Baseline JSON must be a list of finding objects.")
            baseline_findings = []
    except Exception as exc:
        st.warning(f"Could not parse baseline JSON: {exc}")
        baseline_findings = []

run_button = st.button("Run analysis")

if run_button:
    try:
        with st.spinner("Analyzing repository..."):
            orchestrator = AnalysisOrchestrator(enable_rag=enable_rag)
            repo_data = None
            resolved_repo_path = ""

            if input_mode == "Local folder path":
                if not repo_path.strip():
                    st.error("Please enter a folder path.")
                    st.stop()

                resolved_repo_path = repo_path.strip()
                result = orchestrator.run_analysis(
                    repo_path=resolved_repo_path,
                    focus=focus,
                    baseline_findings=baseline_findings,
                    rag_question=rag_question.strip() or None,
                )

            elif input_mode == "Zip file upload":
                if uploaded_zip is None:
                    st.error("Please upload a zip file.")
                    st.stop()

                loaded = load_repository_from_zip(
                    uploaded_zip.getvalue(),
                    uploaded_zip.name,
                )
                st.session_state.zip_temp_dir = loaded.temp_dir
                repo_data = loaded.data
                resolved_repo_path = loaded.data.root_dir or ""

                result = orchestrator.run_analysis(
                    repo_path=resolved_repo_path,
                    focus=focus,
                    baseline_findings=baseline_findings,
                    repo_data=repo_data,
                    rag_question=rag_question.strip() or None,
                )

            else:
                if not github_owner.strip() or not github_repo.strip():
                    st.error("Please enter both GitHub owner and repo.")
                    st.stop()

                loaded = load_repository_from_github(
                    owner=github_owner.strip(),
                    repo=github_repo.strip(),
                    branch=github_branch.strip() or "main",
                    token=github_token.strip() or None,
                )
                st.session_state.github_temp_dir = loaded.temp_dir
                repo_data = loaded.data
                resolved_repo_path = loaded.data.root_dir or ""

                result = orchestrator.run_analysis(
                    repo_path=resolved_repo_path,
                    focus=focus,
                    baseline_findings=baseline_findings,
                    repo_data=repo_data,
                    rag_question=rag_question.strip() or None,
                )

            st.session_state.last_result = result
            st.session_state.repo_root_path = resolved_repo_path

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")


result = st.session_state.last_result

if result:
    st.subheader("Summary")
    st.markdown(result.summary)

    st.subheader("Recommendations")
    for rec in result.recommendations:
        st.write(f"- {rec}")

    st.subheader("Findings")
    if result.findings:
        for finding in result.findings:
            severity = finding.get("severity", "info")
            message = finding.get("message", "")
            st.write(f"- **[{severity}]** {message}")
    else:
        st.write("No findings reported.")

    if result.rag_answer:
        st.divider()
        st.subheader("RAG Answer")
        st.write(result.rag_answer)

        if result.rag_suggestions:
            st.subheader("RAG Suggestions")
            for suggestion in result.rag_suggestions:
                st.write(f"- {suggestion}")

        if result.rag_citations:
            st.subheader("RAG Citations")
            for citation in result.rag_citations:
                st.write(f"- {citation.get('label', citation)}")

        if result.rag_follow_up_questions:
            st.subheader("RAG Follow-up Questions")
            for item in result.rag_follow_up_questions:
                st.write(f"- {item}")

        if result.rag_notes:
            st.subheader("RAG Notes")
            for note in result.rag_notes:
                st.write(f"- {note}")

        rag_debug = {}
        if isinstance(result.metadata, dict):
            rag_debug = result.metadata.get("rag", {}) or {}

        if rag_debug:
            with st.expander("RAG Debug Info"):
                st.json(rag_debug)

                prompt_preview = rag_debug.get("prompt_preview")
                if prompt_preview:
                    st.subheader("Prompt Preview")
                    st.text(prompt_preview)

                selected_files = rag_debug.get("selected_files", [])
                if selected_files:
                    st.subheader("Selected Files")
                    for file_path in selected_files:
                        st.write(f"- {file_path}")

                selected_chunk_previews = rag_debug.get("selected_chunk_previews", [])
                if selected_chunk_previews:
                    st.subheader("Selected Chunk Previews")
                    for i, preview in enumerate(selected_chunk_previews, start=1):
                        st.markdown(f"**Chunk {i}**")
                        st.code(preview)

    with st.expander("Raw metadata"):
        st.json(result.metadata)
else:
    st.info("Run an analysis to see results here.")