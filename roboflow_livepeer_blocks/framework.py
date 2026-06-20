"""Small authoring layer for Livepeer-backed Roboflow workflow packs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

PathLike = Union[str, Path]

WORKFLOW_SOURCE_INPUT = "source"
VDO_TRANSCRIPTION_PACK_ID = "vdo-transcription"


@dataclass(frozen=True)
class CapabilityProfile:
    """Concrete backend choice for a logical workflow role."""

    role: str
    backend: str
    capability: str
    offering: str


DEFAULT_AUDIO_TRANSCRIPTION_PROFILE = CapabilityProfile(
    role="audio.transcription",
    backend="livepeer_remote_http",
    capability="openai:audio-transcriptions",
    offering="nemo-meeting",
)

DEFAULT_VISION_ANALYSIS_PROFILE = CapabilityProfile(
    role="vision.screen_slide_analysis",
    backend="livepeer_remote",
    capability="openai:vision",
    offering="florence-2-large",
)


@dataclass(frozen=True)
class TranscriptionProfiles:
    """Logical capability routing selected by a transcription workflow author."""

    audio_transcription: CapabilityProfile = DEFAULT_AUDIO_TRANSCRIPTION_PROFILE


@dataclass(frozen=True)
class SourceCaptureConfig:
    """Coarse VDO/browser capture role consumed by the runtime kernel."""

    signaling_server: str = ""
    password: str = ""
    include_visual: bool = True
    buffer_ms: int = 300
    audio_sample_rate: int = 48000
    audio_channels: int = 1
    video_frame_interval_seconds: float = 5.0

    @property
    def video_frame_rate(self) -> float:
        if not self.include_visual:
            return 0.0
        return 1.0 / self.video_frame_interval_seconds


@dataclass(frozen=True)
class AudioTranscriptionConfig:
    """Audio normalization and transcription role."""

    output_dir: PathLike
    duration_seconds: float = 30.0
    startup_timeout_seconds: float = 15.0
    language: str = "en"
    preset: str = "meeting"
    max_speakers: int = 4
    ingest_mode: str = "segmented_wav"
    segment_duration_seconds: float = 30.0
    segment_startup_seconds: float = 8.0
    pcm_sample_rate: int = 16000
    pcm_channels: int = 1
    pcm_frame_duration_seconds: float = 0.08
    session_id: str = ""


@dataclass(frozen=True)
class VisionAnalysisConfig:
    """Screen capture and Florence-style visual analysis role."""

    capture_output_dir: PathLike
    analysis_output_dir: PathLike
    duration_seconds: float = 30.0
    startup_seconds: float = 8.0
    frame_interval_seconds: float = 5.0
    max_frames: int = 7
    min_slide_gap_seconds: float = 4.0
    slide_change_threshold: float = 0.72
    model_id: str = "florence-2-large"
    runner_url: str = ""


@dataclass(frozen=True)
class VDOTranscriptionWorkflowSpec:
    """Author-facing spec for materializing a basic VDO transcription workflow."""

    capture: SourceCaptureConfig
    audio: AudioTranscriptionConfig
    profiles: TranscriptionProfiles = TranscriptionProfiles()
    input_name: str = WORKFLOW_SOURCE_INPUT
    pack_id: str = VDO_TRANSCRIPTION_PACK_ID


def path_string(path: PathLike) -> str:
    return str(path)


def vdo_transcription_workflow_spec(
    *,
    output_root: PathLike,
    signaling_server: str = "",
    password: str = "",
    duration_seconds: float = 30.0,
    audio_startup_timeout_seconds: float = 15.0,
    audio_ingest_mode: str = "segmented_wav",
    audio_segment_duration_seconds: float = 30.0,
    audio_segment_startup_seconds: float = 8.0,
    max_speakers: int = 4,
    profiles: TranscriptionProfiles = TranscriptionProfiles(),
) -> VDOTranscriptionWorkflowSpec:
    """Build the normalized VDO transcription authoring spec from coarse roles."""
    artifact_root = Path(output_root)
    return VDOTranscriptionWorkflowSpec(
        capture=SourceCaptureConfig(
            signaling_server=signaling_server,
            password=password,
            include_visual=False,
        ),
        audio=AudioTranscriptionConfig(
            output_dir=artifact_root / "audio-true-streaming",
            duration_seconds=duration_seconds,
            startup_timeout_seconds=audio_startup_timeout_seconds,
            max_speakers=max_speakers,
            ingest_mode=audio_ingest_mode,
            segment_duration_seconds=audio_segment_duration_seconds,
            segment_startup_seconds=audio_segment_startup_seconds,
        ),
        profiles=profiles,
    )


def materialize_vdo_transcription_workflow(spec: VDOTranscriptionWorkflowSpec) -> Dict[str, Any]:
    """Materialize a basic VDO transcription authoring spec as Roboflow workflow JSON."""
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": spec.input_name},
        ],
        "steps": _transcription_steps(
            input_name=spec.input_name,
            capture=spec.capture,
            audio=spec.audio,
            audio_profile=spec.profiles.audio_transcription,
        ),
        "outputs": _transcription_outputs(),
    }


def _transcription_steps(
    *,
    input_name: str,
    capture: SourceCaptureConfig,
    audio: AudioTranscriptionConfig,
    audio_profile: CapabilityProfile,
) -> List[Dict[str, Any]]:
    return [
        _vdo_media_source_step(input_name=input_name, capture=capture),
        _pcm16_transform_step(audio=audio),
        _true_streaming_transcription_step(audio=audio, profile=audio_profile),
        _transcript_output_step(),
    ]


def _vdo_media_source_step(*, input_name: str, capture: SourceCaptureConfig) -> Dict[str, Any]:
    return {
        "type": "roboflow_livepeer_blocks/livepeer_vdo_ninja_media_source@v1",
        "name": "vdo_media_source",
        "source": f"$inputs.{input_name}",
        "signaling_server": capture.signaling_server,
        "password": capture.password,
        "buffer_ms": capture.buffer_ms,
        "audio_enabled": True,
        "video_enabled": capture.include_visual,
        "audio_sample_rate": capture.audio_sample_rate,
        "audio_channels": capture.audio_channels,
        "video_frame_rate": capture.video_frame_rate,
    }


def _pcm16_transform_step(*, audio: AudioTranscriptionConfig) -> Dict[str, Any]:
    return {
        "type": "roboflow_livepeer_blocks/livepeer_pcm16_audio_transform@v1",
        "name": "pcm16_transform",
        "source_descriptor": "$steps.vdo_media_source.audio_source_descriptor",
        "sample_rate": audio.pcm_sample_rate,
        "channels": audio.pcm_channels,
        "frame_duration_seconds": audio.pcm_frame_duration_seconds,
    }


def _true_streaming_transcription_step(
    *,
    audio: AudioTranscriptionConfig,
    profile: CapabilityProfile,
) -> Dict[str, Any]:
    return {
        "type": "roboflow_livepeer_blocks/livepeer_true_streaming_transcription_session@v1",
        "name": "true_streaming_transcription",
        "pcm_descriptor": "$steps.pcm16_transform.pcm_descriptor",
        "duration_seconds": audio.duration_seconds,
        "startup_timeout_seconds": audio.startup_timeout_seconds,
        "output_dir": path_string(audio.output_dir),
        "session_id": audio.session_id,
        "language": audio.language,
        "preset": audio.preset,
        "max_speakers": audio.max_speakers,
        "transcription_backend": profile.backend,
        "livepeer_capability": profile.capability,
        "livepeer_offering": profile.offering,
        "vdo_ingest_mode": audio.ingest_mode,
        "vdo_segment_duration_seconds": audio.segment_duration_seconds,
        "vdo_segment_startup_seconds": audio.segment_startup_seconds,
    }


def _transcript_output_step() -> Dict[str, Any]:
    return {
        "type": "roboflow_livepeer_blocks/livepeer_transcript_output@v1",
        "name": "transcript_output",
        "transcription_session": "$steps.true_streaming_transcription.transcription_session",
    }


def _screen_slide_capture_step(vision: VisionAnalysisConfig) -> Dict[str, Any]:
    return {
        "type": "roboflow_livepeer_blocks/livepeer_screen_slide_capture@v1",
        "name": "screen_slide_capture",
        "video_source_descriptor": "$steps.vdo_media_source.video_source_descriptor",
        "duration_seconds": vision.duration_seconds,
        "startup_seconds": vision.startup_seconds,
        "frame_interval_seconds": vision.frame_interval_seconds,
        "max_frames": vision.max_frames,
        "min_slide_gap_seconds": vision.min_slide_gap_seconds,
        "slide_change_threshold": vision.slide_change_threshold,
        "output_dir": path_string(vision.capture_output_dir),
    }


def _florence2_screen_slide_analysis_step(
    *,
    vision: VisionAnalysisConfig,
    profile: CapabilityProfile,
) -> Dict[str, Any]:
    return {
        "type": "roboflow_livepeer_blocks/livepeer_florence2_screen_slide_analysis@v1",
        "name": "florence2_screen_slide_analysis",
        "capture_descriptor": "$steps.screen_slide_capture.capture_descriptor",
        "output_dir": path_string(vision.analysis_output_dir),
        "model_id": vision.model_id,
        "vision_backend": profile.backend,
        "florence2_runner_url": vision.runner_url,
        "livepeer_capability": profile.capability,
        "livepeer_offering": profile.offering,
    }


def _transcription_outputs() -> List[Dict[str, str]]:
    return [
        _json_output("run_source", "$steps.vdo_media_source.source"),
        _json_output("stream_id", "$steps.vdo_media_source.stream_id"),
        _json_output("tracks", "$steps.vdo_media_source.tracks"),
        _json_output("media_descriptor", "$steps.vdo_media_source.media_descriptor"),
        _json_output("audio_status", "$steps.transcript_output.status"),
        _json_output("audio_session_id", "$steps.transcript_output.session_id"),
        _json_output("sent_audio_seconds", "$steps.transcript_output.sent_audio_seconds"),
        _json_output("sent_frame_count", "$steps.transcript_output.sent_frame_count"),
        _json_output("text", "$steps.transcript_output.text"),
        _json_output("speaker_count", "$steps.transcript_output.speaker_count"),
        _json_output("transcript_events", "$steps.transcript_output.transcript_events"),
        _json_output("transcript_events_jsonl_path", "$steps.transcript_output.events_jsonl_path"),
        _json_output("transcript_result_json_path", "$steps.transcript_output.result_json_path"),
        _json_output("transcript_text_path", "$steps.transcript_output.transcript_text_path"),
    ]


def _json_output(name: str, selector: str) -> Dict[str, str]:
    return {"type": "JsonField", "name": name, "selector": selector}
