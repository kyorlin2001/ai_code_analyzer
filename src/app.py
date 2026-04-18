from __future__ import annotations

import json

import streamlit as st

from orchestrator import AnalysisOrchestrator
from tools.github_loader import load_repository_from_github
from tools.repo_loader import load_repository_from_zip


st.set_page_config(page_title="Agentic Code Analyst", layout="wide")
st.title("Agentic Code Analyst 🤖")
st.write("Analyze a local repo, zip archive, or GitHub repository.")


if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "zip_temp_dir" not in st.session_state:
    st.session_state.zip_temp_dir = None
if "github_temp_dir" not in st.session_state:
    st.session_state.github_temp_dir = None


input_mode = st.radio(
    "Choose input type",
    ["Local folder path", "Zip file upload", "GitHub repository"],
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
            orchestrator = AnalysisOrchestrator()

            if input_mode == "Local folder path":
                if not repo_path.strip():
                    st.error("Please enter a folder path.")
                    st.stop()

                result = orchestrator.run_analysis(
                    repo_path=repo_path.strip(),
                    focus=focus,
                    baseline_findings=baseline_findings,
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

                result = orchestrator.run_analysis(
                    repo_path=loaded.data.root_dir or "",
                    focus=focus,
                    baseline_findings=baseline_findings,
                    repo_data=loaded.data,
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

                result = orchestrator.run_analysis(
                    repo_path=loaded.data.root_dir or "",
                    focus=focus,
                    baseline_findings=baseline_findings,
                    repo_data=loaded.data,
                )

            st.session_state.last_result = result

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

    with st.expander("Raw metadata"):
        st.json(result.metadata)
else:
    st.info("Run an analysis to see results here.")