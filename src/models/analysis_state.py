from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnalysisState:
    """
    Shared mutable state passed through the agent pipeline.

    The orchestrator owns this object and each agent can read from it and
    add its own outputs into `agent_outputs`.
    """

    repo_path: str
    repo_name: str
    file_tree: Any = None
    files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    agent_outputs: dict[str, Any] = field(default_factory=dict)
    findings: list[dict[str, Any]] = field(default_factory=list)
    summaries: dict[str, str] = field(default_factory=dict)

    # Optional regression support
    baseline_findings: list[dict[str, Any]] = field(default_factory=list)
    baseline_summary: str = ""