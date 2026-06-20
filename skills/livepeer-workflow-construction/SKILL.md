---
name: livepeer-workflow-construction
description: "Construct, run, validate, or explain Livepeer Roboflow media workflows from composable blocks. Use when an agent needs to design workflow JSON or a framework-backed workflow pack for live/browser media capture, audio transcription, visual analysis, session artifacts, or runtime execution without relying on a task-shaped production workflow."
---

# Livepeer Workflow Construction

Use this skill to turn a media task into a workflow contract, compose the needed Livepeer-aware blocks, run the workflow through a generic runtime, and inspect session artifacts.

## Core Shape

Keep the product contract centered on:

```text
input + workflow + session = events + artifacts + summary
```

Start from the user's requested behavior:

1. Define the input contract: source type, runtime parameters, duration/window, credentials/signaling, and expected media.
2. Choose the smallest set of blocks that satisfy the requested behavior.
3. Use Livepeer Modules via Cloudspe as the default provider. Discover capabilities dynamically when a local repo copy and `LIVEPEER_OPEN_CLEARINGHOUSE_API_KEY` are available.
4. Set backend/capability/offering values from the discovered provider catalog, not from a hardcoded model list.
5. Materialize workflow JSON as an artifact. Prefer the generic builder when a local repo copy is available.
6. Execute the materialized workflow with runtime parameters. Prefer the generic session runner when a local repo copy is available.
7. Report status, events, transcript/audio artifacts, visual artifacts, and any blocker by layer.

## References

Read only the references needed for the task:

- `references/source-diagnostics.md`: VDO/browser stream preflight, host/container URL mapping, source resolution.
- `references/authoring-surface.md`: generic workflow builder and session runner available when a local `livepeer-roboflow/` repo copy exists.
- `references/provider-discovery.md`: Livepeer Modules / Cloudspe provider discovery and fast-fail behavior.
- `references/raw-block-catalog.md`: available Livepeer workflow blocks, inputs, outputs, and common parameters.
- `references/runtime-contract.md`: generic Docker/runtime command contracts for running authored workflow JSON.
- `references/artifact-contract.md`: status, event, transcript, visual, and summary artifact expectations.

## Authoring Principles

- Prefer composing documented blocks over copying a complete workflow.
- If `livepeer-roboflow/` is available, use `roboflow_livepeer_blocks.authoring.WorkflowBuilder` to construct workflow JSON instead of hand-assembling large dictionaries.
- If `livepeer-roboflow/` is available, use `livepeer-poc run-session` for preflight, runtime execution, status, result, and summary sidecars instead of creating a task-specific run harness.
- Use Livepeer Modules through Cloudspe by default. Do not silently fall back to local/self-hosted runners; if Cloudspe credentials, capability discovery, or routing fail, report that failure and mention self-hosting as an explicit alternative.
- Keep user-facing source values as runtime parameters when practical.
- Keep output roots explicit so artifacts survive container execution.
- Separate authoring from execution: the authored `workflow.json` should be inspectable before it is run.
- Treat preflight failures, runner route failures, empty media, and workflow execution errors as different failure layers.

## Minimal Handoff

For every run, report:

- Authored workflow path.
- Runtime command or script path.
- Final status and failure reason, if any.
- Resolved source and signaling server.
- Session/run ID.
- Artifact root.
- Event log path.
- Transcript path or transcript failure reason.
- Visual result path or visual failure reason.
- Brief content summary when transcript or visual artifacts exist.
