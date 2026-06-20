"""Generic helpers for authoring Livepeer-backed Roboflow workflow JSON.

This module intentionally avoids task-specific topology. It helps callers build
valid Roboflow workflow dictionaries from chosen blocks, references, and outputs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Union

PathLike = Union[str, Path]

SESSION_ROOT_ENV = "LIVEPEER_WORKFLOW_SESSION_ROOT"
ARTIFACT_ROOT_ENV = "LIVEPEER_WORKFLOW_ARTIFACT_ROOT"
DEFAULT_SESSION_ROOT = "/workspace/authoring-test"
DEFAULT_ARTIFACT_ROOT = f"{DEFAULT_SESSION_ROOT}/artifacts"


def input_ref(name: str) -> str:
    """Return a Roboflow workflow input reference."""
    return f"$inputs.{name}"


def step_ref(step_name: str, output_name: str) -> str:
    """Return a Roboflow workflow step output reference."""
    return f"$steps.{step_name}.{output_name}"


def json_output(name: str, selector: str) -> Dict[str, str]:
    """Create a JsonField workflow output projection."""
    return {"type": "JsonField", "name": name, "selector": selector}


def session_root() -> str:
    """Return the container-visible workflow session root."""
    return os.getenv(SESSION_ROOT_ENV, DEFAULT_SESSION_ROOT)


def artifact_root() -> str:
    """Return the container-visible root where block artifacts should be written."""
    return os.getenv(ARTIFACT_ROOT_ENV, DEFAULT_ARTIFACT_ROOT)


def artifact_path(*parts: str) -> str:
    """Return a path under the canonical persisted artifact root.

    Use this in authored workflow step parameters such as output_dir. The path
    is container-visible when run through the Docker workbench and maps to the
    host directory mounted at /workspace/authoring-test.
    """
    root = artifact_root().rstrip("/")
    suffix = "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))
    if not suffix:
        return root
    return f"{root}/{suffix}"


@dataclass(frozen=True)
class StepHandle:
    """Reference handle returned after adding a workflow step."""

    name: str

    def output(self, output_name: str) -> str:
        return step_ref(self.name, output_name)


class WorkflowBuilder:
    """Small builder for Roboflow workflow JSON.

    The builder only owns workflow mechanics:
    inputs, step dictionaries, output selectors, serialization, and basic
    validation. It does not choose blocks or impose a workflow shape.
    """

    def __init__(self, *, version: str = "1.0", artifact_root: Optional[str] = None) -> None:
        self.version = version
        self.artifact_root = artifact_root or artifact_path()
        self._inputs: List[Dict[str, str]] = []
        self._steps: List[Dict[str, Any]] = []
        self._outputs: List[Dict[str, str]] = []
        self._step_names: set[str] = set()
        self._input_names: set[str] = set()
        self._output_names: set[str] = set()

    def artifact_path(self, *parts: str) -> str:
        suffix = "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))
        if not suffix:
            return self.artifact_root
        return f"{self.artifact_root.rstrip('/')}/{suffix}"

    def input(self, name: str, *, input_type: str = "WorkflowParameter") -> str:
        if not name:
            raise ValueError("input name must be non-empty")
        if name in self._input_names:
            return input_ref(name)
        self._inputs.append({"type": input_type, "name": name})
        self._input_names.add(name)
        return input_ref(name)

    def step(self, name: str, block_type: str, **parameters: Any) -> StepHandle:
        if not name:
            raise ValueError("step name must be non-empty")
        if name in self._step_names:
            raise ValueError(f"duplicate step name: {name}")
        if not block_type:
            raise ValueError("step block type must be non-empty")
        step: Dict[str, Any] = {"type": block_type, "name": name}
        step.update(parameters)
        self._steps.append(step)
        self._step_names.add(name)
        return StepHandle(name)

    def output(self, name: str, selector: str) -> None:
        if not name:
            raise ValueError("output name must be non-empty")
        if name in self._output_names:
            raise ValueError(f"duplicate output name: {name}")
        if not selector:
            raise ValueError("output selector must be non-empty")
        self._outputs.append(json_output(name, selector))
        self._output_names.add(name)

    def extend_outputs(self, outputs: List[Mapping[str, str]]) -> None:
        for output in outputs:
            self.output(str(output["name"]), str(output["selector"]))

    def to_dict(self) -> Dict[str, Any]:
        workflow: Dict[str, Any] = {"version": self.version}
        if self._inputs:
            workflow["inputs"] = list(self._inputs)
        workflow["steps"] = list(self._steps)
        workflow["outputs"] = list(self._outputs)
        validate_workflow_dict(workflow)
        return workflow

    def write_json(self, path: PathLike, *, indent: int = 2) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        workflow = self.to_dict()
        output_path.write_text(
            json.dumps(workflow, indent=indent, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output_path


def validate_workflow_dict(workflow: Mapping[str, Any]) -> None:
    """Run structural validation that is cheap and runtime-independent."""
    if not workflow.get("version"):
        raise ValueError("workflow must include a version")
    steps = workflow.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("workflow must include at least one step")
    seen_steps: set[str] = set()
    for index, step in enumerate(steps):
        if not isinstance(step, MutableMapping):
            raise ValueError(f"step {index} must be an object")
        step_type = step.get("type")
        step_name = step.get("name")
        if not isinstance(step_type, str) or not step_type:
            raise ValueError(f"step {index} must include a non-empty type")
        if not isinstance(step_name, str) or not step_name:
            raise ValueError(f"step {index} must include a non-empty name")
        if step_name in seen_steps:
            raise ValueError(f"duplicate step name: {step_name}")
        seen_steps.add(step_name)

    outputs = workflow.get("outputs", [])
    if not isinstance(outputs, list):
        raise ValueError("workflow outputs must be a list")
    seen_outputs: set[str] = set()
    for index, output in enumerate(outputs):
        if not isinstance(output, MutableMapping):
            raise ValueError(f"output {index} must be an object")
        if output.get("type") != "JsonField":
            raise ValueError(f"output {index} must use type JsonField")
        output_name = output.get("name")
        selector = output.get("selector")
        if not isinstance(output_name, str) or not output_name:
            raise ValueError(f"output {index} must include a non-empty name")
        if output_name in seen_outputs:
            raise ValueError(f"duplicate output name: {output_name}")
        if not isinstance(selector, str) or not selector:
            raise ValueError(f"output {index} must include a non-empty selector")
        seen_outputs.add(output_name)


def load_workflow(path: PathLike) -> Dict[str, Any]:
    workflow = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_workflow_dict(workflow)
    return workflow


__all__ = [
    "StepHandle",
    "WorkflowBuilder",
    "ARTIFACT_ROOT_ENV",
    "DEFAULT_ARTIFACT_ROOT",
    "DEFAULT_SESSION_ROOT",
    "SESSION_ROOT_ENV",
    "artifact_path",
    "artifact_root",
    "input_ref",
    "json_output",
    "load_workflow",
    "session_root",
    "step_ref",
    "validate_workflow_dict",
]
