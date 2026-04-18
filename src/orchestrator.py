from __future__ import annotations

from typing import Optional

from agents.architecture_agent import ArchitectureAgent
from agents.dependency_agent import DependencyAgent
from agents.intake_agent import IntakeAgent
from agents.issue_agent import IssueAgent
from agents.report_agent import ReportAgent
from agents.regression_agent import RegressionAgent
from models.analysis_result import AnalysisResult
from models.analysis_state import AnalysisState
from tools.repo_loader import RepositoryData, load_repository


class AnalysisOrchestrator:
    """
    Coordinates the full analysis workflow.
    """

    def __init__(
        self,
        intake_agent: Optional[IntakeAgent] = None,
        dependency_agent: Optional[DependencyAgent] = None,
        architecture_agent: Optional[ArchitectureAgent] = None,
        issue_agent: Optional[IssueAgent] = None,
        report_agent: Optional[ReportAgent] = None,
        regression_agent: Optional[RegressionAgent] = None,
    ) -> None:
        self.intake_agent = intake_agent or IntakeAgent()
        self.dependency_agent = dependency_agent or DependencyAgent()
        self.architecture_agent = architecture_agent or ArchitectureAgent()
        self.issue_agent = issue_agent or IssueAgent()
        self.report_agent = report_agent or ReportAgent()
        self.regression_agent = regression_agent or RegressionAgent()

    def run_analysis(
        self,
        repo_path: str,
        focus: str | None = None,
        baseline_findings: list[dict] | None = None,
        repo_data: RepositoryData | None = None,
    ) -> AnalysisResult:
        if repo_data is None:
            repo_data = load_repository(repo_path)

        state = AnalysisState(
            repo_path=repo_path,
            repo_name=repo_data.repo_name,
            file_tree=repo_data.file_tree,
            files=repo_data.files,
            metadata={
                "focus": focus,
            },
            baseline_findings=baseline_findings or [],
        )

        state = self._run_intake(state)

        if focus in (None, "dependencies", "full"):
            state = self._run_dependencies(state)

        if focus in (None, "architecture", "full"):
            state = self._run_architecture(state)

        if focus in (None, "issues", "full"):
            state = self._run_issues(state)

        if baseline_findings:
            state = self._run_regression(state)

        return self._build_report(state)

    def _run_intake(self, state: AnalysisState) -> AnalysisState:
        output = self.intake_agent.run(state)
        state.agent_outputs["intake"] = output
        return state

    def _run_dependencies(self, state: AnalysisState) -> AnalysisState:
        output = self.dependency_agent.run(state)
        state.agent_outputs["dependencies"] = output
        return state

    def _run_architecture(self, state: AnalysisState) -> AnalysisState:
        output = self.architecture_agent.run(state)
        state.agent_outputs["architecture"] = output
        return state

    def _run_issues(self, state: AnalysisState) -> AnalysisState:
        output = self.issue_agent.run(state)
        state.agent_outputs["issues"] = output
        return state

    def _run_regression(self, state: AnalysisState) -> AnalysisState:
        output = self.regression_agent.run(state)
        state.agent_outputs["regression"] = output
        return state

    def _build_report(self, state: AnalysisState) -> AnalysisResult:
        report_output = self.report_agent.run(state)

        return AnalysisResult(
            repo_name=state.repo_name,
            summary=report_output.summary,
            findings=report_output.findings,
            recommendations=report_output.recommendations,
            metadata={
                "repo_path": state.repo_path,
                "focus": state.metadata.get("focus"),
                "agent_outputs": state.agent_outputs,
            },
        )