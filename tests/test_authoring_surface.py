from pathlib import Path

from roboflow_livepeer_blocks.authoring import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_SESSION_ROOT,
    WorkflowBuilder,
    artifact_path,
)


def test_artifact_path_defaults_to_container_mount_root():
    assert DEFAULT_SESSION_ROOT == "/workspace/authoring-test"
    assert DEFAULT_ARTIFACT_ROOT == "/workspace/authoring-test/artifacts"
    assert artifact_path("audio-true-streaming") == (
        "/workspace/authoring-test/artifacts/audio-true-streaming"
    )


def test_workflow_builder_exposes_artifact_path(tmp_path: Path):
    workflow_path = tmp_path / "workflow.json"
    builder = WorkflowBuilder()
    source = builder.input("source")
    media = builder.step("media", "roboflow_livepeer_blocks/example@v1", source=source)
    builder.output("stream_id", media.output("stream_id"))
    assert builder.artifact_path("visual-analysis") == (
        "/workspace/authoring-test/artifacts/visual-analysis"
    )
    builder.write_json(workflow_path)
    assert workflow_path.exists()
