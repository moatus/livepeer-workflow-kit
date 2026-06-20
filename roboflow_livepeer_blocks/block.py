"""Roboflow static workflow block for Livepeer audio transcription."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, Field

try:
    from inference.core.workflows.execution_engine.entities.base import OutputDefinition
    from inference.core.workflows.execution_engine.entities.types import (
        BOOLEAN_KIND,
        DICTIONARY_KIND,
        FLOAT_KIND,
        INTEGER_KIND,
        LIST_OF_VALUES_KIND,
        STRING_KIND,
        Selector,
    )
    from inference.core.workflows.prototypes.block import (
        AirGappedAvailability,
        BlockResult,
        WorkflowBlock,
        WorkflowBlockManifest,
    )
except ModuleNotFoundError:
    BOOLEAN_KIND = "Boolean"
    DICTIONARY_KIND = "Dictionary"
    FLOAT_KIND = "Float"
    INTEGER_KIND = "Integer"
    LIST_OF_VALUES_KIND = "List"
    STRING_KIND = "String"
    BlockResult = dict[str, Any]

    @dataclass(frozen=True)
    class OutputDefinition:
        name: str
        kind: list[str]

    @dataclass(frozen=True)
    class AirGappedAvailability:
        available: bool
        reason: str

    class WorkflowBlock:
        @classmethod
        def get_init_parameters(cls) -> List[str]:
            return []

    class WorkflowBlockManifest(BaseModel):
        pass

    def Selector(kind: list[str]) -> type[Any]:
        if BOOLEAN_KIND in kind:
            return bool
        if FLOAT_KIND in kind:
            return float
        if INTEGER_KIND in kind:
            return int
        return str

from .client import (
    DEFAULT_CAPABILITY,
    DEFAULT_OFFERING,
    DEFAULT_RESPONSE_FORMAT,
    LivepeerOpenClearinghouseClient,
)
from .config import (
    DEFAULT_FLORENCE2_RUNNER_URL,
    DEFAULT_OPEN_CLEARINGHOUSE_URL,
    DEFAULT_TRUE_STREAMING_CAPABILITY,
    DEFAULT_TRUE_STREAMING_OFFERING,
    DEFAULT_TRUE_STREAMING_TRANSCRIPTION_BACKEND,
    DEFAULT_VISION_BACKEND,
    DEFAULT_VISION_CAPABILITY,
    DEFAULT_VISION_OFFERING,
    init_florence2_runner_url,
    init_local_audio_ingest_url,
    init_nemo_diarized_runner_url,
    init_open_clearinghouse_api_key,
    init_open_clearinghouse_url,
    init_true_streaming_capability,
    init_true_streaming_offering,
    init_true_streaming_transcription_backend,
    init_vision_backend,
    init_vision_capability,
    init_vision_offering,
    init_vdo_signaling_server_url,
)
from .ingest import (
    DEFAULT_INGEST_OUTPUT_DIR,
    capture_rolling_audio_segments,
    parse_vdo_stream_id,
    record_vdo_segment,
    resolve_vdo_stream_source,
)
from .livepeer_http_chunking_client import (
    LivepeerRemoteFallbackTransportClient,
    LivepeerRemoteHttpChunkingClient,
    fallback_offering_for_streaming,
)
from .local_ingest import parse_local_audio_ingest_source
from .nemo_client import NemoDiarizedTranscriptionClient
from .streaming import LivepeerVDONinjaAudioSegmentSource
from .true_streaming import (
    LocalAudioIngestWebSocketClient,
    NemoTrueStreamingWebSocketClient,
    build_local_audio_ingest_true_streaming_runner,
    build_vdo_direct_true_streaming_runner,
    build_vdo_true_streaming_runner,
)
from .vision import (
    DEFAULT_FLORENCE2_MODEL_ID,
    DEFAULT_FRAME_INTERVAL_SECONDS,
    DEFAULT_MEETING_CONTEXT_PROMPT,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MIN_SLIDE_GAP_SECONDS,
    DEFAULT_SLIDE_CHANGE_THRESHOLD,
    Florence2InferenceAnalyzer,
    copy_slide_frame,
    extract_video_frames,
    make_slide_signal_text,
    presentation_likelihood,
    separate_meeting_visual_text,
    similarity_ratio,
    summarize_meeting_visual_events,
    write_json,
    write_jsonl,
)


TRANSCRIBE_LONG_DESCRIPTION = """
Transcribe a local audio file by chunking it and sending each chunk through the
Livepeer Open Clearinghouse handoff flow. This block directly performs
POST /v1/jobs, multipart POST /v1/cap, and POST /v1/jobs/{id}/settle using httpx.
"""

INGEST_LONG_DESCRIPTION = """
Capture rolling audio segments from a VDO.Ninja stream ID using Raspberry.Ninja
and GStreamer under the hood. The block accepts a direct stream ID or a viewer
URL, records one or more bounded segments without launching a browser, extracts
16 kHz mono WAV audio, and returns segment metadata plus paths suitable for
downstream Livepeer transcription blocks.
"""

DIARIZED_TRANSCRIBE_LONG_DESCRIPTION = """
Transcribe and diarize a local audio file by sending it directly to the
standalone audio-diarized-transcription runner over multipart HTTP. By default
this block uses the runner's OpenAI-compatible `POST /v1/audio/transcriptions`
route and normalizes the Livepeer diarization extension back into Roboflow
workflow fields.
"""

LIVE_DIARIZED_SESSION_LONG_DESCRIPTION = """
Capture audio chunks from a VDO.Ninja source and feed them into one stateful
live diarization session on the standalone audio-diarized-transcription runner.
The block persists Roboflow-side session events and final transcript artifacts
without changing the existing segmented transcription workflow.
"""

TRUE_STREAMING_SESSION_LONG_DESCRIPTION = """
Bounded Roboflow compatibility block for the Nemotron true-streaming path. The
block captures a finite number of VDO.Ninja audio segments, but sends their PCM
frames through one persistent runner WebSocket session instead of treating the
WebSocket as a normal per-step workflow call.
"""

DIRECT_TRUE_STREAMING_SESSION_LONG_DESCRIPTION = """
Direct live Roboflow block for the runner's true-streaming path. The block uses
Raspberry.Ninja fdsink mode to pull decoded VDO.Ninja audio as a live PCM pipe,
resamples it to 16 kHz mono PCM, and feeds frames into one persistent
`WS /v1/audio/transcriptions/stream` session without creating intermediate
segment WAV/WebM files.
"""

VDO_LIVE_AUDIO_SOURCE_LONG_DESCRIPTION = """
Composable source descriptor block for a live VDO.Ninja audio stream. This block
does not start capture by itself; it resolves the stream ID and emits a typed
descriptor that downstream workflow blocks can transform or replace.
"""

VDO_MEDIA_SOURCE_LONG_DESCRIPTION = """
Composable source descriptor block for a live VDO.Ninja media stream. This block
does not start capture by itself; it resolves the stream ID once and emits
separate audio and video descriptors so workflows can branch into transcription,
frame sampling, slide scanning, OCR, or other media-specific paths.
"""

LOCAL_INGEST_LIVE_AUDIO_SOURCE_LONG_DESCRIPTION = """
Composable source descriptor block for a localhost audio ingest session. The
source is the local WebSocket URL that the browser extension publishes PCM16
audio into, for example `ws://127.0.0.1:8876/v1/ingest/audio/test-session`.
"""

PCM16_AUDIO_TRANSFORM_LONG_DESCRIPTION = """
Composable audio transform descriptor block. It accepts a live audio source
descriptor and declares the PCM16 mono sample rate, frame cadence, and resampler
needed by the true-streaming runner WebSocket.
"""

TRUE_STREAMING_TRANSCRIPTION_LONG_DESCRIPTION = """
Composable runner block for `WS /v1/audio/transcriptions/stream`. It consumes a
PCM16 stream descriptor, opens the live VDO.Ninja pipe described upstream, and
feeds realtime-paced frames into the local audio-diarized-transcription runner.
"""

LOCAL_INGEST_TRUE_STREAMING_TRANSCRIPTION_LONG_DESCRIPTION = """
Composable runner block for `WS /v1/audio/transcriptions/stream` backed by the
localhost ingest service. It consumes a PCM16 descriptor emitted from a local
ingest source block and forwards live PCM frames into the transcription runner
without VDO.Ninja, Raspberry.Ninja, or local ffmpeg subprocesses.
"""

TRANSCRIPT_OUTPUT_LONG_DESCRIPTION = """
Composable transcript output block. It consumes a true-streaming transcription
session result and returns normalized text, speaker, event, and artifact fields
for downstream Roboflow blocks or workflow outputs.
"""

SCREEN_SLIDE_CAPTURE_LONG_DESCRIPTION = """
Composable screen and slide capture descriptor block. It consumes a live video
source descriptor and declares the bounded frame sampling, slide debounce, and
artifact contract that a downstream Roboflow or Livepeer visual runner should
execute. The block intentionally keeps capture ownership with the downstream
consumer so audio and visual branches can share one workflow source.
"""

FLORENCE2_SCREEN_SLIDE_ANALYSIS_LONG_DESCRIPTION = """
Composable visual analysis descriptor block for screen and slide captures. The
block uses Florence-2 large as the base visual model contract and emits a
normalized session envelope with caption, OCR, and slide event fields that
downstream workflow blocks can consume or replace with a concrete visual runner.
"""

DEFAULT_VDO_TRUE_STREAMING_INGEST_MODE = "direct_pcm"
DEFAULT_VDO_SEGMENT_DURATION_SECONDS = 30.0
DEFAULT_VDO_SEGMENT_STARTUP_SECONDS = 8.0
MAX_LIVEPEER_REMOTE_HTTP_CHUNK_SIZE_SECONDS = 16.08


def _normalize_transcription_backend(
    backend: str,
    *,
    fallback: str = DEFAULT_TRUE_STREAMING_TRANSCRIPTION_BACKEND,
) -> str:
    resolved = str(backend or fallback).strip().lower().replace("-", "_")
    if resolved in {"", "default"}:
        resolved = fallback
    aliases = {
        "remote_http": "livepeer_remote_http",
        "livepeer_http": "livepeer_remote_http",
        "http_chunking": "livepeer_remote_http",
        "livepeer_remote_http_chunking": "livepeer_remote_http",
    }
    resolved = aliases.get(resolved, resolved)
    if resolved not in {"local", "livepeer_remote", "livepeer_remote_http"}:
        raise ValueError(
            "transcription_backend must be 'local', 'livepeer_remote', or "
            "'livepeer_remote_http'"
        )
    return resolved


def _remote_session_units(
    *,
    duration_seconds: float,
    estimated_runway_units: Optional[int],
    max_total_units: Optional[int],
) -> tuple[int, int]:
    derived_estimate = (
        int(estimated_runway_units)
        if estimated_runway_units is not None
        else max(1, int(round(float(duration_seconds))))
    )
    if derived_estimate <= 0:
        raise ValueError("livepeer_remote requires positive estimated_runway_units")
    derived_cap = (
        int(max_total_units)
        if max_total_units is not None
        else max(derived_estimate + 30, int(math.ceil(derived_estimate * 1.25)))
    )
    if derived_cap < derived_estimate:
        raise ValueError("livepeer_max_total_units must be >= estimated runway units")
    return derived_estimate, derived_cap


def _resolve_true_streaming_client(
    *,
    backend: str,
    local_client_cls: Any,
    runner_url: str,
    api_key: Optional[str],
    base_url: str,
    livepeer_capability: str,
    livepeer_offering: str,
    duration_seconds: float,
    livepeer_estimated_runway_units: Optional[int],
    livepeer_max_total_units: Optional[int],
) -> tuple[Any, str, Dict[str, Any]]:
    resolved_backend = _normalize_transcription_backend(backend)
    if resolved_backend == "local":
        return local_client_cls, runner_url, {}
    if resolved_backend == "livepeer_remote_http":
        return (
            LivepeerRemoteHttpChunkingClient,
            base_url,
            {
                "api_key": api_key,
                "capability": livepeer_capability,
                "offering": fallback_offering_for_streaming(livepeer_offering),
            },
        )
    estimated_units, max_units = _remote_session_units(
        duration_seconds=duration_seconds,
        estimated_runway_units=livepeer_estimated_runway_units,
        max_total_units=livepeer_max_total_units,
    )
    return (
        LivepeerRemoteFallbackTransportClient,
        base_url,
        {
            "api_key": api_key,
            "capability": livepeer_capability,
            "realtime_offering": livepeer_offering,
            "estimated_runway_units": estimated_units,
            "max_total_units": max_units,
        },
    )


def _livepeer_remote_http_chunk_size_seconds(
    *,
    window_seconds: float,
    frame_duration_seconds: float,
) -> float:
    return min(
        MAX_LIVEPEER_REMOTE_HTTP_CHUNK_SIZE_SECONDS,
        max(10.0, float(window_seconds) + float(frame_duration_seconds)),
    )


def _normalize_vdo_true_streaming_ingest_mode(
    ingest_mode: str,
    *,
    fallback: str = DEFAULT_VDO_TRUE_STREAMING_INGEST_MODE,
) -> str:
    resolved = str(ingest_mode or fallback).strip().lower().replace("-", "_")
    if resolved in {"", "default"}:
        resolved = fallback
    aliases = {
        "direct": "direct_pcm",
        "direct_pcm": "direct_pcm",
        "live_pcm": "direct_pcm",
        "segmented": "segmented_wav",
        "segmented_wav": "segmented_wav",
        "chunked": "segmented_wav",
        "chunked_wav": "segmented_wav",
    }
    normalized = aliases.get(resolved)
    if normalized is None:
        raise ValueError(
            "vdo_ingest_mode must be 'direct_pcm' or 'segmented_wav'"
        )
    return normalized


def _segmented_vdo_transport_metrics(
    *,
    captured_segments: List[Dict[str, Any]],
    frame_duration_seconds: float,
) -> tuple[float, int]:
    sent_audio_seconds = 0.0
    sent_frame_count = 0
    for segment in captured_segments:
        duration_seconds = float(
            segment.get("audio_duration_seconds")
            or segment.get("requested_duration_seconds")
            or 0.0
        )
        if duration_seconds <= 0:
            continue
        sent_audio_seconds += duration_seconds
        sent_frame_count += max(1, int(math.ceil(duration_seconds / frame_duration_seconds)))
    return round(sent_audio_seconds, 6), sent_frame_count


class LivepeerAudioTranscribeManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Livepeer Audio Transcribe",
            "version": "v1",
            "short_description": "Transcribe local audio via Livepeer Open Clearinghouse.",
            "long_description": TRANSCRIBE_LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["livepeer", "audio", "transcription", "whisper"],
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-waveform-lines",
                "blockPriority": 5,
            },
        }
    )

    type: Literal[
        "roboflow_livepeer_blocks/livepeer_audio_transcribe@v1",
        "LivepeerAudioTranscribe",
    ]
    audio_path: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Local path to the audio file to transcribe.",
        examples=["/tmp/audio.mp3", "$inputs.audio_path"],
    )
    chunk_size_seconds: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        default=10,
        description="Chunk size in seconds for local audio splitting.",
        examples=[10],
    )
    offering: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=DEFAULT_OFFERING,
        description="Livepeer offering/model to request.",
        examples=[DEFAULT_OFFERING],
    )
    capability: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=DEFAULT_CAPABILITY,
        description="Livepeer capability to request.",
        examples=[DEFAULT_CAPABILITY],
    )
    response_format: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=DEFAULT_RESPONSE_FORMAT,
        description="Broker transcription response format.",
        examples=[DEFAULT_RESPONSE_FORMAT],
    )
    max_total_units_per_chunk: Optional[int] = Field(
        default=None,
        description="Optional max_total_units override for each opened chunk job.",
        examples=[20],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="text", kind=[STRING_KIND]),
            OutputDefinition(name="chunks", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="actual_units", kind=[INTEGER_KIND]),
            OutputDefinition(name="job_ids", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="work_ids", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="raw_responses", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="result", kind=[DICTIONARY_KIND]),
        ]

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=False, reason="requires_internet")

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class LivepeerAudioTranscribeV1(WorkflowBlock):
    def __init__(
        self,
        api_key: Optional[str],
        base_url: str = DEFAULT_OPEN_CLEARINGHOUSE_URL,
        client_cls: Type[LivepeerOpenClearinghouseClient] = LivepeerOpenClearinghouseClient,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._client_cls = client_cls

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["api_key", "base_url"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LivepeerAudioTranscribeManifest

    def run(
        self,
        audio_path: str,
        chunk_size_seconds: int = 10,
        offering: str = DEFAULT_OFFERING,
        capability: str = DEFAULT_CAPABILITY,
        response_format: str = DEFAULT_RESPONSE_FORMAT,
        max_total_units_per_chunk: Optional[int] = None,
    ) -> BlockResult:
        with self._client_cls(api_key=self._api_key, base_url=self._base_url) as client:
            result = client.transcribe_audio_file(
                audio_path=audio_path,
                chunk_size_seconds=float(chunk_size_seconds),
                capability=capability,
                offering=offering,
                response_format=response_format,
                max_total_units_per_chunk=max_total_units_per_chunk,
            )
        return {
            "text": result["text"],
            "chunks": result["chunks"],
            "actual_units": result["actual_units"],
            "job_ids": result["job_ids"],
            "work_ids": result["work_ids"],
            "raw_responses": result["raw_responses"],
            "result": result,
        }


class LivepeerAudioDiarizedTranscribeLocalManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Livepeer Audio Diarized Transcribe Local",
            "version": "v1",
            "short_description": "Transcribe local audio with speaker labels via standalone runner.",
            "long_description": DIARIZED_TRANSCRIBE_LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["livepeer", "audio", "transcription", "diarization", "nemo"],
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-users",
                "blockPriority": 6,
            },
        }
    )

    type: Literal[
        "roboflow_livepeer_blocks/livepeer_audio_diarized_transcribe_local@v1",
        "LivepeerAudioDiarizedTranscribeLocal",
    ]
    audio_path: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Local path to the audio file to transcribe and diarize.",
        examples=["/tmp/audio.wav", "$inputs.audio_path"],
    )
    language: Union[Selector(kind=[STRING_KIND]), str] = Field(default="en", examples=["en"])
    preset: Union[Selector(kind=[STRING_KIND]), str] = Field(default="meeting", examples=["meeting"])
    num_speakers: Optional[int] = Field(
        default=None,
        description="Optional known speaker count. Leave empty to estimate.",
        examples=[2],
    )
    max_speakers: Union[Selector(kind=[INTEGER_KIND]), int] = Field(default=8, examples=[8])
    model: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default="nemo-diarized-transcription-meeting-v0",
        examples=["nemo-diarized-transcription-meeting-v0"],
    )
    include_words: Union[Selector(kind=[BOOLEAN_KIND]), bool] = Field(default=True, examples=[True])
    include_artifacts: Union[Selector(kind=[BOOLEAN_KIND]), bool] = Field(default=True, examples=[True])
    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="text", kind=[STRING_KIND]),
            OutputDefinition(name="speaker_count", kind=[INTEGER_KIND]),
            OutputDefinition(name="speakers", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="segments", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="words", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="artifacts", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="actual_units", kind=[INTEGER_KIND]),
            OutputDefinition(name="api_endpoint", kind=[STRING_KIND]),
            OutputDefinition(name="result", kind=[DICTIONARY_KIND]),
        ]

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=True, reason="standalone_audio_diarized_runner")

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class LivepeerAudioDiarizedTranscribeLocalV1(WorkflowBlock):
    def __init__(
        self,
        runner_url: Optional[str] = None,
        client_cls: Type[NemoDiarizedTranscriptionClient] = NemoDiarizedTranscriptionClient,
    ) -> None:
        self._runner_url = runner_url or init_nemo_diarized_runner_url()
        self._client_cls = client_cls

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["runner_url"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LivepeerAudioDiarizedTranscribeLocalManifest

    def run(
        self,
        audio_path: str,
        language: str = "en",
        preset: str = "meeting",
        num_speakers: Optional[int] = None,
        max_speakers: int = 8,
        model: str = "nemo-diarized-transcription-meeting-v0",
        include_words: bool = True,
        include_artifacts: bool = True,
    ) -> BlockResult:
        with self._client_cls(base_url=self._runner_url) as client:
            result = client.diarized_transcribe_audio_file(
                audio_path=audio_path,
                model=model,
                language=language,
                preset=preset,
                num_speakers=num_speakers,
                max_speakers=int(max_speakers),
                response_format="json",
                include_words=bool(include_words),
                include_artifacts=bool(include_artifacts),
            )
        return {
            "text": result.get("text", ""),
            "speaker_count": result.get("speaker_count", 0),
            "speakers": result.get("speakers", []),
            "segments": result.get("segments", []),
            "words": result.get("words", []),
            "artifacts": result.get("artifacts", {}),
            "actual_units": result.get("actual_units", result.get("usage", {}).get("work_units", 0)),
            "api_endpoint": result.get("api_endpoint", ""),
            "result": result,
        }


class LivepeerVDONinjaLiveDiarizedSessionManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Livepeer VDO.Ninja Live Diarized Session",
            "version": "v1",
            "short_description": "Drive the standalone stateful live diarization runner from VDO.Ninja audio.",
            "long_description": LIVE_DIARIZED_SESSION_LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "livepeer",
                "vdo.ninja",
                "audio",
                "diarization",
                "live",
                "session",
                "nemo",
            ],
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-users-viewfinder",
                "blockPriority": 7,
            },
        }
    )

    type: Literal[
        "roboflow_livepeer_blocks/livepeer_vdo_ninja_live_diarized_session@v1",
        "LivepeerVDONinjaLiveDiarizedSession",
    ]
    source: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Raw VDO.Ninja stream ID or viewer URL containing ?view=...",
        examples=["stream_mzadj0spa", "https://vdo.ninja/?view=stream_mzadj0spa"],
    )
    segment_count: Union[Selector(kind=[INTEGER_KIND]), int] = Field(default=16, examples=[16])
    segment_duration_seconds: Union[Selector(kind=[FLOAT_KIND, INTEGER_KIND]), float] = Field(
        default=30.0,
        examples=[30.0],
    )
    startup_seconds: Union[Selector(kind=[FLOAT_KIND, INTEGER_KIND]), float] = Field(
        default=8.0,
        examples=[8.0],
    )
    output_dir: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=str(DEFAULT_INGEST_OUTPUT_DIR / "live-diarized-sessions"),
        examples=[str(DEFAULT_INGEST_OUTPUT_DIR / "live-diarized-sessions")],
    )
    session_id: Union[Selector(kind=[STRING_KIND]), str] = Field(default="", examples=[""])
    password: Union[Selector(kind=[STRING_KIND]), str] = Field(default="", examples=[""])
    buffer_ms: Union[Selector(kind=[INTEGER_KIND]), int] = Field(default=300, examples=[300])
    audio_only: Union[Selector(kind=[BOOLEAN_KIND]), bool] = Field(default=True, examples=[True])
    language: Union[Selector(kind=[STRING_KIND]), str] = Field(default="en", examples=["en"])
    preset: Union[Selector(kind=[STRING_KIND]), str] = Field(default="meeting", examples=["meeting"])
    num_speakers: Optional[int] = Field(default=None, examples=[2])
    max_speakers: Union[Selector(kind=[INTEGER_KIND]), int] = Field(default=8, examples=[8])
    vad_strategy: Union[Selector(kind=[STRING_KIND]), str] = Field(default="energy", examples=["energy"])
    rolling_window_seconds: Union[Selector(kind=[FLOAT_KIND, INTEGER_KIND]), float] = Field(
        default=60.0,
        examples=[60.0],
    )
    energy_threshold: Union[Selector(kind=[FLOAT_KIND, INTEGER_KIND]), float] = Field(
        default=0.012,
        examples=[0.012],
    )
    final_transcription: Union[Selector(kind=[BOOLEAN_KIND]), bool] = Field(
        default=True,
        examples=[True],
    )
    include_words: Union[Selector(kind=[BOOLEAN_KIND]), bool] = Field(default=True, examples=[True])
    include_artifacts: Union[Selector(kind=[BOOLEAN_KIND]), bool] = Field(default=True, examples=[True])

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="session_id", kind=[STRING_KIND]),
            OutputDefinition(name="stream_id", kind=[STRING_KIND]),
            OutputDefinition(name="status", kind=[STRING_KIND]),
            OutputDefinition(name="captured_segments", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="audio_paths", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="live_segments", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="text", kind=[STRING_KIND]),
            OutputDefinition(name="speaker_count", kind=[INTEGER_KIND]),
            OutputDefinition(name="speakers", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="words", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="final_audio_path", kind=[STRING_KIND]),
            OutputDefinition(name="events_jsonl_path", kind=[STRING_KIND]),
            OutputDefinition(name="provisional_transcript_jsonl_path", kind=[STRING_KIND]),
            OutputDefinition(name="transcript_events", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="transcript_event_count", kind=[INTEGER_KIND]),
            OutputDefinition(name="result_json_path", kind=[STRING_KIND]),
            OutputDefinition(name="transcript_text_path", kind=[STRING_KIND]),
            OutputDefinition(name="result", kind=[DICTIONARY_KIND]),
        ]

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=False, reason="requires_vdo_ninja_network")

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class LivepeerVDONinjaLiveDiarizedSessionV1(WorkflowBlock):
    def __init__(
        self,
        runner_url: Optional[str] = None,
        client_cls: Type[NemoDiarizedTranscriptionClient] = NemoDiarizedTranscriptionClient,
    ) -> None:
        self._runner_url = runner_url or init_nemo_diarized_runner_url()
        self._client_cls = client_cls

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["runner_url"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LivepeerVDONinjaLiveDiarizedSessionManifest

    def run(
        self,
        source: str,
        segment_count: int = 16,
        segment_duration_seconds: float = 30.0,
        startup_seconds: float = 8.0,
        output_dir: str = str(DEFAULT_INGEST_OUTPUT_DIR / "live-diarized-sessions"),
        session_id: str = "",
        password: str = "",
        buffer_ms: int = 300,
        audio_only: bool = True,
        language: str = "en",
        preset: str = "meeting",
        num_speakers: Optional[int] = None,
        max_speakers: int = 8,
        vad_strategy: str = "energy",
        rolling_window_seconds: float = 60.0,
        energy_threshold: float = 0.012,
        final_transcription: bool = True,
        include_words: bool = True,
        include_artifacts: bool = True,
    ) -> BlockResult:
        artifact_dir = _prepare_live_artifact_dir(output_dir=output_dir, source=source)
        events_path = artifact_dir / "session-events.jsonl"
        provisional_transcript_path = artifact_dir / "provisional-transcript.jsonl"
        result_path = artifact_dir / "session-result.json"

        stream_source = LivepeerVDONinjaAudioSegmentSource(
            source=source,
            output_dir=artifact_dir,
            segment_duration_seconds=float(segment_duration_seconds),
            startup_seconds=float(startup_seconds),
            password=password,
            buffer_ms=int(buffer_ms),
            audio_only=bool(audio_only),
            max_segments=int(segment_count),
        )
        events: List[Dict[str, Any]] = []
        transcript_events: List[Dict[str, Any]] = []
        captured_segments: List[Dict[str, Any]] = []
        created: Dict[str, Any] = {}
        finished: Dict[str, Any] = {}

        with self._client_cls(base_url=self._runner_url) as client:
            created = client.create_live_session(
                session_id=session_id or None,
                language=language,
                preset=preset,
                num_speakers=num_speakers,
                max_speakers=int(max_speakers),
                vad_strategy=vad_strategy,
                rolling_window_seconds=float(rolling_window_seconds),
                energy_threshold=float(energy_threshold),
            )
            live_session_id = str(created["session_id"])
            _append_live_event(events, events_path, created)
            _append_live_transcript_events(
                transcript_events,
                provisional_transcript_path,
                _transcript_events_from_live_response(created),
            )
            stream_source.open()
            try:
                for segment in stream_source.segments():
                    segment_payload = segment.as_dict()
                    captured_segments.append(segment_payload)
                    _append_live_event(
                        events,
                        events_path,
                        {
                            "event_type": "source.audio_chunk",
                            "session_id": live_session_id,
                            "stream_id": segment.stream_id,
                            "source_segment": segment_payload,
                        },
                    )
                    ingested = client.ingest_live_audio_file(
                        session_id=live_session_id,
                        audio_path=str(segment.audio_path),
                        sequence_index=segment.index,
                    )
                    ingested["source_segment"] = segment_payload
                    _append_live_event(events, events_path, ingested)
                    _append_live_transcript_events(
                        transcript_events,
                        provisional_transcript_path,
                        _transcript_events_from_live_response(
                            ingested,
                            source_segment=segment_payload,
                        ),
                    )
            finally:
                stream_source.close()
                finished = client.finish_live_session(
                    session_id=str(created["session_id"]),
                    run_final_transcription=bool(final_transcription),
                    include_words=bool(include_words),
                    include_artifacts=bool(include_artifacts),
                )
                _append_live_event(events, events_path, finished)
                _append_live_transcript_events(
                    transcript_events,
                    provisional_transcript_path,
                    _transcript_events_from_live_response(finished),
                )

        final_transcription_result = finished.get("final_transcription") or {}
        transcript_text_path = _write_transcript_text(
            artifact_dir=artifact_dir,
            final_transcription=final_transcription_result,
        )
        result = _live_session_result_payload(
            created=created,
            finished=finished,
            events=events,
            transcript_events=transcript_events,
            captured_segments=captured_segments,
            artifact_dir=artifact_dir,
            events_path=events_path,
            provisional_transcript_path=provisional_transcript_path,
            result_path=result_path,
            transcript_text_path=transcript_text_path,
        )
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "session_id": result["session_id"],
            "stream_id": result["stream_id"],
            "status": result["status"],
            "captured_segments": result["captured_segments"],
            "audio_paths": result["audio_paths"],
            "live_segments": result["live_segments"],
            "text": result["text"],
            "speaker_count": result["speaker_count"],
            "speakers": result["speakers"],
            "words": result["words"],
            "final_audio_path": result["final_audio_path"],
            "events_jsonl_path": result["events_jsonl_path"],
            "provisional_transcript_jsonl_path": result["provisional_transcript_jsonl_path"],
            "transcript_events": result["transcript_events"],
            "transcript_event_count": result["transcript_event_count"],
            "result_json_path": result["result_json_path"],
            "transcript_text_path": result["transcript_text_path"],
            "result": result,
        }


class LivepeerVDONinjaTrueStreamingSessionManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Livepeer VDO.Ninja Nemotron True Streaming Session",
            "version": "v1",
            "short_description": "Drive one persistent Nemotron true-streaming WebSocket from VDO.Ninja audio.",
            "long_description": TRUE_STREAMING_SESSION_LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "livepeer",
                "vdo.ninja",
                "audio",
                "transcription",
                "true-streaming",
                "websocket",
                "nemotron",
            ],
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-tower-broadcast",
                "blockPriority": 8,
            },
        }
    )

    type: Literal[
        "roboflow_livepeer_blocks/livepeer_vdo_ninja_true_streaming_session@v1",
        "LivepeerVDONinjaTrueStreamingSession",
    ]
    source: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Raw VDO.Ninja stream ID or viewer URL containing ?view=...",
        examples=["stream_mzadj0spa", "https://vdo.ninja/?view=stream_mzadj0spa"],
    )
    segment_count: Union[Selector(kind=[INTEGER_KIND]), int] = Field(default=16, examples=[16])
    segment_duration_seconds: Union[Selector(kind=[FLOAT_KIND, INTEGER_KIND]), float] = Field(
        default=5.0,
        examples=[5.0],
    )
    startup_seconds: Union[Selector(kind=[FLOAT_KIND, INTEGER_KIND]), float] = Field(
        default=8.0,
        examples=[8.0],
    )
    output_dir: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=str(DEFAULT_INGEST_OUTPUT_DIR / "true-streaming"),
        examples=[str(DEFAULT_INGEST_OUTPUT_DIR / "true-streaming")],
    )
    session_id: Union[Selector(kind=[STRING_KIND]), str] = Field(default="", examples=[""])
    password: Union[Selector(kind=[STRING_KIND]), str] = Field(default="", examples=[""])
    buffer_ms: Union[Selector(kind=[INTEGER_KIND]), int] = Field(default=300, examples=[300])
    audio_only: Union[Selector(kind=[BOOLEAN_KIND]), bool] = Field(default=True, examples=[True])
    language: Union[Selector(kind=[STRING_KIND]), str] = Field(default="en", examples=["en"])
    preset: Union[Selector(kind=[STRING_KIND]), str] = Field(default="meeting", examples=["meeting"])
    max_speakers: Union[Selector(kind=[INTEGER_KIND]), int] = Field(default=4, examples=[4])
    sample_rate: Union[Selector(kind=[INTEGER_KIND]), int] = Field(default=16000, examples=[16000])
    frame_duration_seconds: Union[Selector(kind=[FLOAT_KIND, INTEGER_KIND]), float] = Field(
        default=0.08,
        examples=[0.08],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="session_id", kind=[STRING_KIND]),
            OutputDefinition(name="stream_id", kind=[STRING_KIND]),
            OutputDefinition(name="status", kind=[STRING_KIND]),
            OutputDefinition(name="captured_segments", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="audio_paths", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="text", kind=[STRING_KIND]),
            OutputDefinition(name="speaker_count", kind=[INTEGER_KIND]),
            OutputDefinition(name="speakers", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="transcript_events", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="transcript_event_count", kind=[INTEGER_KIND]),
            OutputDefinition(name="events_jsonl_path", kind=[STRING_KIND]),
            OutputDefinition(name="result_json_path", kind=[STRING_KIND]),
            OutputDefinition(name="transcript_text_path", kind=[STRING_KIND]),
            OutputDefinition(name="result", kind=[DICTIONARY_KIND]),
        ]

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=False, reason="requires_vdo_ninja_and_runner_websocket")

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class LivepeerVDONinjaTrueStreamingSessionV1(WorkflowBlock):
    def __init__(
        self,
        runner_url: Optional[str] = None,
        client_cls: Type[NemoTrueStreamingWebSocketClient] = NemoTrueStreamingWebSocketClient,
    ) -> None:
        self._runner_url = runner_url or init_nemo_diarized_runner_url()
        self._client_cls = client_cls

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["runner_url"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LivepeerVDONinjaTrueStreamingSessionManifest

    def run(
        self,
        source: str,
        segment_count: int = 16,
        segment_duration_seconds: float = 5.0,
        startup_seconds: float = 8.0,
        output_dir: str = str(DEFAULT_INGEST_OUTPUT_DIR / "true-streaming"),
        session_id: str = "",
        password: str = "",
        buffer_ms: int = 300,
        audio_only: bool = True,
        language: str = "en",
        preset: str = "meeting",
        max_speakers: int = 4,
        sample_rate: int = 16000,
        frame_duration_seconds: float = 0.08,
    ) -> BlockResult:
        runner = build_vdo_true_streaming_runner(
            source=source,
            runner_url=self._runner_url,
            output_dir=output_dir,
            segment_count=int(segment_count),
            segment_duration_seconds=float(segment_duration_seconds),
            startup_seconds=float(startup_seconds),
            session_id=session_id,
            password=password,
            buffer_ms=int(buffer_ms),
            audio_only=bool(audio_only),
            language=language,
            preset=preset,
            max_speakers=int(max_speakers),
            sample_rate=int(sample_rate),
            frame_duration_seconds=float(frame_duration_seconds),
            client_cls=self._client_cls,
        )
        result = runner.run()
        return {
            "session_id": result["session_id"],
            "stream_id": result["stream_id"],
            "status": result["status"],
            "captured_segments": result["captured_segments"],
            "audio_paths": result["audio_paths"],
            "text": result["text"],
            "speaker_count": result["speaker_count"],
            "speakers": result["speakers"],
            "transcript_events": result["transcript_events"],
            "transcript_event_count": result["transcript_event_count"],
            "events_jsonl_path": result["events_jsonl_path"],
            "result_json_path": result["result_json_path"],
            "transcript_text_path": result["transcript_text_path"],
            "result": result,
        }


class LivepeerVDONinjaMediaSourceManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Livepeer VDO.Ninja Media Source",
            "version": "v1",
            "short_description": "Describe a live VDO.Ninja A/V source for composable workflows.",
            "long_description": VDO_MEDIA_SOURCE_LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "source",
            "search_keywords": [
                "livepeer",
                "vdo.ninja",
                "audio",
                "video",
                "media",
                "source",
                "stream",
                "composable",
            ],
            "ui_manifest": {
                "section": "sources",
                "icon": "fal fa-tower-broadcast",
                "blockPriority": 9,
            },
        }
    )

    type: Literal[
        "roboflow_livepeer_blocks/livepeer_vdo_ninja_media_source@v1",
        "LivepeerVDONinjaMediaSource",
    ]
    source: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default="auto",
        description='Raw VDO.Ninja stream ID, viewer URL containing ?view=..., or "auto" to select the newest live bridge stream.',
        examples=["auto", "stream_9xc43b5s6", "https://vdo.ninja/?view=stream_9xc43b5s6"],
    )
    signaling_server: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default="",
        description='Optional custom VDO-compatible signaling server, for example "wss://localhost:9443".',
        examples=["wss://localhost:9443", ""],
    )
    password: Union[Selector(kind=[STRING_KIND]), str] = Field(default="", examples=[""])
    buffer_ms: Union[Selector(kind=[INTEGER_KIND]), int] = Field(default=300, examples=[300])
    audio_enabled: Union[Selector(kind=[BOOLEAN_KIND]), bool] = Field(default=True, examples=[True])
    video_enabled: Union[Selector(kind=[BOOLEAN_KIND]), bool] = Field(default=True, examples=[True])
    audio_sample_rate: Union[Selector(kind=[INTEGER_KIND]), int] = Field(default=48000, examples=[48000])
    audio_channels: Union[Selector(kind=[INTEGER_KIND]), int] = Field(default=1, examples=[1])
    video_frame_rate: Union[Selector(kind=[FLOAT_KIND, INTEGER_KIND]), float] = Field(
        default=1.0,
        description="Future default sampling cadence for downstream video blocks.",
        examples=[1.0],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="media_descriptor", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="audio_source_descriptor", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="video_source_descriptor", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="source", kind=[STRING_KIND]),
            OutputDefinition(name="stream_id", kind=[STRING_KIND]),
            OutputDefinition(name="tracks", kind=[LIST_OF_VALUES_KIND]),
        ]

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=False, reason="requires_vdo_ninja_network")

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class LivepeerVDONinjaMediaSourceV1(WorkflowBlock):
    def __init__(self, vdo_signaling_server_url: str = "") -> None:
        self._default_signaling_server = vdo_signaling_server_url or init_vdo_signaling_server_url()

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["vdo_signaling_server_url"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LivepeerVDONinjaMediaSourceManifest

    def run(
        self,
        source: str = "auto",
        signaling_server: str = "",
        password: str = "",
        buffer_ms: int = 300,
        audio_enabled: bool = True,
        video_enabled: bool = True,
        audio_sample_rate: int = 48000,
        audio_channels: int = 1,
        video_frame_rate: float = 1.0,
    ) -> BlockResult:
        if int(audio_channels) != 1:
            raise ValueError("VDO.Ninja live audio source currently emits mono PCM only")
        if not audio_enabled and not video_enabled:
            raise ValueError("at least one media track must be enabled")
        resolved = resolve_vdo_stream_source(
            source=source,
            signaling_server=str(signaling_server or self._default_signaling_server or ""),
            password=password,
            timeout_seconds=20.0,
        )
        resolved_source = str(resolved["source"])
        stream_id = str(resolved["stream_id"])
        resolved_signaling_server = str(resolved.get("signaling_server") or "")
        resolved_password = str(resolved.get("password", password))
        media_ref = {
            "schema_version": "livepeer.vdo_ninja_media_ref.v1",
            "kind": "live_media_source_ref",
            "source_type": "vdo_ninja",
            "source": resolved_source,
            "stream_id": stream_id,
            "transport": "webrtc",
        }
        if resolved_signaling_server:
            media_ref["signaling_server"] = resolved_signaling_server
        if resolved.get("auto_resolved"):
            media_ref["requested_source"] = resolved.get("requested_source", source)
            media_ref["auto_resolved"] = True
            media_ref["bridge_status_url"] = resolved.get("status_url", "")
            media_ref["bridge_stream_id"] = resolved.get("bridge_stream_id", stream_id)
        tracks = []
        audio_descriptor: Dict[str, Any] = {}
        video_descriptor: Dict[str, Any] = {}
        if audio_enabled:
            tracks.append("audio")
            audio_descriptor = {
                "schema_version": "livepeer.live_audio_source.v1",
                "kind": "live_audio_source",
                "source_type": "vdo_ninja",
                "source": resolved_source,
                "stream_id": stream_id,
                "media_source": media_ref,
                "available_tracks": ["audio", "video"],
                "selected_tracks": ["audio"],
                "publisher": "raspberry_ninja_fdsink",
                "sample_format": "s16le",
                "sample_rate": int(audio_sample_rate),
                "channels": int(audio_channels),
                "password": resolved_password,
                "buffer_ms": int(buffer_ms),
            }
            if resolved_signaling_server:
                audio_descriptor["signaling_server"] = resolved_signaling_server
            if resolved.get("auto_resolved"):
                audio_descriptor["requested_source"] = resolved.get("requested_source", source)
                audio_descriptor["auto_resolved"] = True
                audio_descriptor["bridge_status_url"] = resolved.get("status_url", "")
                audio_descriptor["bridge_stream_id"] = resolved.get("bridge_stream_id", stream_id)
        if video_enabled:
            tracks.append("video")
            video_descriptor = {
                "schema_version": "livepeer.live_video_source.v1",
                "kind": "live_video_source",
                "source_type": "vdo_ninja",
                "source": resolved_source,
                "stream_id": stream_id,
                "media_source": media_ref,
                "available_tracks": ["audio", "video"],
                "selected_tracks": ["video"],
                "transport": "webrtc",
                "consumer": "future_frame_sampler",
                "frame_outputs": ["sampled_frames", "raw_frames"],
                "default_frame_rate": float(video_frame_rate),
                "password": resolved_password,
                "buffer_ms": int(buffer_ms),
            }
            if resolved_signaling_server:
                video_descriptor["signaling_server"] = resolved_signaling_server
            if resolved.get("auto_resolved"):
                video_descriptor["requested_source"] = resolved.get("requested_source", source)
                video_descriptor["auto_resolved"] = True
                video_descriptor["bridge_status_url"] = resolved.get("status_url", "")
                video_descriptor["bridge_stream_id"] = resolved.get("bridge_stream_id", stream_id)
        media_descriptor = {
            "schema_version": "livepeer.vdo_ninja_media_source.v1",
            "kind": "live_media_source",
            "source_type": "vdo_ninja",
            "source": resolved_source,
            "stream_id": stream_id,
            "transport": "webrtc",
            "tracks": tracks,
            "audio_source_descriptor": audio_descriptor,
            "video_source_descriptor": video_descriptor,
            "execution": {
                "starts_capture": False,
                "capture_owned_by_consumer_block": True,
            },
        }
        if resolved_signaling_server:
            media_descriptor["signaling_server"] = resolved_signaling_server
        if resolved.get("auto_resolved"):
            media_descriptor["requested_source"] = resolved.get("requested_source", source)
            media_descriptor["auto_resolved"] = True
            media_descriptor["bridge_status_url"] = resolved.get("status_url", "")
            media_descriptor["bridge_stream_id"] = resolved.get("bridge_stream_id", stream_id)
        return {
            "media_descriptor": media_descriptor,
            "audio_source_descriptor": audio_descriptor,
            "video_source_descriptor": video_descriptor,
            "source": resolved_source,
            "stream_id": stream_id,
            "tracks": tracks,
        }


class LivepeerVDONinjaLiveAudioSourceManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Livepeer VDO.Ninja Live Audio Source",
            "version": "v1",
            "short_description": "Describe a live VDO.Ninja audio source for composable workflows.",
            "long_description": VDO_LIVE_AUDIO_SOURCE_LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "source",
            "search_keywords": [
                "livepeer",
                "vdo.ninja",
                "audio",
                "source",
                "stream",
                "composable",
            ],
            "ui_manifest": {
                "section": "sources",
                "icon": "fal fa-satellite-dish",
                "blockPriority": 10,
            },
        }
    )

    type: Literal[
        "roboflow_livepeer_blocks/livepeer_vdo_ninja_live_audio_source@v1",
        "LivepeerVDONinjaLiveAudioSource",
    ]
    source: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default="auto",
        description='Raw VDO.Ninja stream ID, viewer URL containing ?view=..., or "auto" to select the newest live bridge stream.',
        examples=["auto", "stream_9xc43b5s6", "https://vdo.ninja/?view=stream_9xc43b5s6"],
    )
    signaling_server: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default="",
        description='Optional custom VDO-compatible signaling server, for example "wss://localhost:9443".',
        examples=["wss://localhost:9443", ""],
    )
    password: Union[Selector(kind=[STRING_KIND]), str] = Field(default="", examples=[""])
    buffer_ms: Union[Selector(kind=[INTEGER_KIND]), int] = Field(default=300, examples=[300])
    sample_rate: Union[Selector(kind=[INTEGER_KIND]), int] = Field(default=48000, examples=[48000])
    channels: Union[Selector(kind=[INTEGER_KIND]), int] = Field(default=1, examples=[1])

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="source_descriptor", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="source", kind=[STRING_KIND]),
            OutputDefinition(name="stream_id", kind=[STRING_KIND]),
            OutputDefinition(name="sample_rate", kind=[INTEGER_KIND]),
            OutputDefinition(name="channels", kind=[INTEGER_KIND]),
        ]

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=False, reason="requires_vdo_ninja_network")

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class LivepeerVDONinjaLiveAudioSourceV1(WorkflowBlock):
    def __init__(self, vdo_signaling_server_url: str = "") -> None:
        self._default_signaling_server = vdo_signaling_server_url or init_vdo_signaling_server_url()

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["vdo_signaling_server_url"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LivepeerVDONinjaLiveAudioSourceManifest

    def run(
        self,
        source: str = "auto",
        signaling_server: str = "",
        password: str = "",
        buffer_ms: int = 300,
        sample_rate: int = 48000,
        channels: int = 1,
    ) -> BlockResult:
        if int(channels) != 1:
            raise ValueError("VDO.Ninja live audio source currently emits mono PCM only")
        resolved = resolve_vdo_stream_source(
            source=source,
            signaling_server=str(signaling_server or self._default_signaling_server or ""),
            password=password,
            timeout_seconds=20.0,
        )
        resolved_source = str(resolved["source"])
        stream_id = str(resolved["stream_id"])
        resolved_signaling_server = str(resolved.get("signaling_server") or "")
        resolved_password = str(resolved.get("password", password))
        media_ref = {
            "schema_version": "livepeer.vdo_ninja_media_ref.v1",
            "kind": "live_media_source_ref",
            "source_type": "vdo_ninja",
            "source": resolved_source,
            "stream_id": stream_id,
            "transport": "webrtc",
        }
        if resolved_signaling_server:
            media_ref["signaling_server"] = resolved_signaling_server
        if resolved.get("auto_resolved"):
            media_ref["requested_source"] = resolved.get("requested_source", source)
            media_ref["auto_resolved"] = True
            media_ref["bridge_status_url"] = resolved.get("status_url", "")
            media_ref["bridge_stream_id"] = resolved.get("bridge_stream_id", stream_id)
        descriptor = {
            "schema_version": "livepeer.live_audio_source.v1",
            "kind": "live_audio_source",
            "source_type": "vdo_ninja",
            "source": resolved_source,
            "stream_id": stream_id,
            "media_source": media_ref,
            "available_tracks": ["audio", "video"],
            "selected_tracks": ["audio"],
            "publisher": "raspberry_ninja_fdsink",
            "sample_format": "s16le",
            "sample_rate": int(sample_rate),
            "channels": int(channels),
            "password": resolved_password,
            "buffer_ms": int(buffer_ms),
        }
        if resolved_signaling_server:
            descriptor["signaling_server"] = resolved_signaling_server
        if resolved.get("auto_resolved"):
            descriptor["requested_source"] = resolved.get("requested_source", source)
            descriptor["auto_resolved"] = True
            descriptor["bridge_status_url"] = resolved.get("status_url", "")
            descriptor["bridge_stream_id"] = resolved.get("bridge_stream_id", stream_id)
        return {
            "source_descriptor": descriptor,
            "source": resolved_source,
            "stream_id": stream_id,
            "sample_rate": int(sample_rate),
            "channels": int(channels),
        }


class LivepeerLocalAudioIngressLiveAudioSourceManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Livepeer Local Audio Ingest Source",
            "version": "v1",
            "short_description": "Describe a localhost ingest session published by the browser extension.",
            "long_description": LOCAL_INGEST_LIVE_AUDIO_SOURCE_LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "source",
            "search_keywords": [
                "livepeer",
                "localhost",
                "websocket",
                "ingest",
                "audio",
                "composable",
            ],
            "ui_manifest": {
                "section": "source",
                "icon": "fal fa-plug-circle-bolt",
                "blockPriority": 10,
            },
        }
    )

    type: Literal[
        "roboflow_livepeer_blocks/livepeer_local_audio_ingest_source@v1",
        "LivepeerLocalAudioIngressLiveAudioSource",
    ]
    source: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Local ingest WebSocket URL or session id.",
        examples=[
            "ws://local-audio-ingest:8876/v1/ingest/audio/test-session",
            "test-session",
        ],
    )
    sample_rate: Union[Selector(kind=[INTEGER_KIND]), int] = Field(default=16000, examples=[16000])
    channels: Union[Selector(kind=[INTEGER_KIND]), int] = Field(default=1, examples=[1])

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="source_descriptor", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="source", kind=[STRING_KIND]),
            OutputDefinition(name="stream_id", kind=[STRING_KIND]),
            OutputDefinition(name="sample_rate", kind=[INTEGER_KIND]),
            OutputDefinition(name="channels", kind=[INTEGER_KIND]),
        ]

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=True, reason="localhost_only")

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class LivepeerLocalAudioIngressLiveAudioSourceV1(WorkflowBlock):
    def __init__(self, local_audio_ingest_url: Optional[str] = None) -> None:
        self._local_audio_ingest_url = local_audio_ingest_url or init_local_audio_ingest_url()

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["local_audio_ingest_url"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LivepeerLocalAudioIngressLiveAudioSourceManifest

    def run(
        self,
        source: str,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> BlockResult:
        if int(channels) != 1:
            raise ValueError("local ingest audio source currently emits mono PCM only")
        info = parse_local_audio_ingest_source(
            source=source,
            default_base_url=self._local_audio_ingest_url,
        )
        descriptor = {
            "schema_version": "livepeer.live_audio_source.v1",
            "kind": "live_audio_source",
            "source_type": "localhost_ingest",
            "source": info["ingest_url"],
            "stream_id": info["session_id"],
            "publisher": "localhost_ingest_ws",
            "sample_format": "s16le",
            "sample_rate": int(sample_rate),
            "channels": int(channels),
            "status_url": info["status_url"],
            "consume_url": info["consume_url"],
        }
        return {
            "source_descriptor": descriptor,
            "source": info["ingest_url"],
            "stream_id": info["session_id"],
            "sample_rate": int(sample_rate),
            "channels": int(channels),
        }


class LivepeerPCM16AudioTransformManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Livepeer PCM16 Audio Transform",
            "version": "v1",
            "short_description": "Declare PCM16 mono transform settings for streaming audio.",
            "long_description": PCM16_AUDIO_TRANSFORM_LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "search_keywords": [
                "livepeer",
                "pcm",
                "audio",
                "resample",
                "transform",
                "composable",
            ],
            "ui_manifest": {
                "section": "transformations",
                "icon": "fal fa-waveform",
                "blockPriority": 11,
            },
        }
    )

    type: Literal[
        "roboflow_livepeer_blocks/livepeer_pcm16_audio_transform@v1",
        "LivepeerPCM16AudioTransform",
    ]
    source_descriptor: Union[Selector(kind=[DICTIONARY_KIND]), Dict[str, Any]] = Field(
        description="Live audio source descriptor emitted by a source block.",
        examples=["$steps.vdo_source.source_descriptor"],
    )
    sample_rate: Union[Selector(kind=[INTEGER_KIND]), int] = Field(default=16000, examples=[16000])
    channels: Union[Selector(kind=[INTEGER_KIND]), int] = Field(default=1, examples=[1])
    frame_duration_seconds: Union[Selector(kind=[FLOAT_KIND, INTEGER_KIND]), float] = Field(
        default=0.08,
        examples=[0.08],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="pcm_descriptor", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="source_descriptor", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="sample_rate", kind=[INTEGER_KIND]),
            OutputDefinition(name="channels", kind=[INTEGER_KIND]),
            OutputDefinition(name="frame_duration_seconds", kind=[FLOAT_KIND]),
        ]

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=False, reason="inherits_source_network_requirements")

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class LivepeerPCM16AudioTransformV1(WorkflowBlock):
    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return []

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LivepeerPCM16AudioTransformManifest

    def run(
        self,
        source_descriptor: Dict[str, Any],
        sample_rate: int = 16000,
        channels: int = 1,
        frame_duration_seconds: float = 0.08,
    ) -> BlockResult:
        if not isinstance(source_descriptor, dict):
            raise ValueError("source_descriptor must be a dictionary")
        if source_descriptor.get("kind") != "live_audio_source":
            raise ValueError("source_descriptor must describe a live_audio_source")
        if int(channels) != 1:
            raise ValueError("true-streaming runner currently expects mono PCM")
        descriptor = {
            "schema_version": "livepeer.pcm16_audio_stream.v1",
            "kind": "pcm16_audio_stream",
            "source_descriptor": source_descriptor,
            "input_sample_format": str(source_descriptor.get("sample_format") or "s16le"),
            "input_sample_rate": int(source_descriptor.get("sample_rate") or 48000),
            "input_channels": int(source_descriptor.get("channels") or 1),
            "sample_format": "s16le",
            "sample_rate": int(sample_rate),
            "channels": int(channels),
            "frame_duration_seconds": float(frame_duration_seconds),
            "resampler": "ffmpeg",
        }
        return {
            "pcm_descriptor": descriptor,
            "source_descriptor": source_descriptor,
            "sample_rate": int(sample_rate),
            "channels": int(channels),
            "frame_duration_seconds": float(frame_duration_seconds),
        }


class LivepeerTrueStreamingTranscriptionSessionManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Livepeer True Streaming Transcription Session",
            "version": "v1",
            "short_description": "Run a PCM16 live audio descriptor through the local true-streaming runner.",
            "long_description": TRUE_STREAMING_TRANSCRIPTION_LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "livepeer",
                "audio",
                "transcription",
                "true-streaming",
                "websocket",
                "runner",
                "composable",
            ],
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-tower-broadcast",
                "blockPriority": 12,
            },
        }
    )

    type: Literal[
        "roboflow_livepeer_blocks/livepeer_true_streaming_transcription_session@v1",
        "LivepeerTrueStreamingTranscriptionSession",
    ]
    pcm_descriptor: Union[Selector(kind=[DICTIONARY_KIND]), Dict[str, Any]] = Field(
        description="PCM16 descriptor emitted by the PCM transform block.",
        examples=["$steps.pcm16_transform.pcm_descriptor"],
    )
    duration_seconds: Union[Selector(kind=[FLOAT_KIND, INTEGER_KIND]), float] = Field(
        default=60.0,
        description="Seconds of live audio to capture. Use 0 to run until the source stream ends.",
        examples=[60.0],
    )
    startup_timeout_seconds: Union[Selector(kind=[FLOAT_KIND, INTEGER_KIND]), float] = Field(
        default=20.0,
        examples=[20.0],
    )
    output_dir: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=str(DEFAULT_INGEST_OUTPUT_DIR / "direct-true-streaming"),
        examples=[str(DEFAULT_INGEST_OUTPUT_DIR / "direct-true-streaming")],
    )
    session_id: Union[Selector(kind=[STRING_KIND]), str] = Field(default="", examples=[""])
    language: Union[Selector(kind=[STRING_KIND]), str] = Field(default="en", examples=["en"])
    preset: Union[Selector(kind=[STRING_KIND]), str] = Field(default="meeting", examples=["meeting"])
    max_speakers: Union[Selector(kind=[INTEGER_KIND]), int] = Field(default=4, examples=[4])
    transcription_backend: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=DEFAULT_TRUE_STREAMING_TRANSCRIPTION_BACKEND,
        description=(
            "Transcription transport backend: local runner WebSocket, legacy "
            "Livepeer remote ws-realtime with HTTP fallback, or HTTP-only "
            "Livepeer remote chunking."
        ),
        examples=["local", "livepeer_remote", "livepeer_remote_http"],
    )
    vdo_ingest_mode: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=DEFAULT_VDO_TRUE_STREAMING_INGEST_MODE,
        description="VDO ingest strategy: direct live PCM pipe or bounded segmented WAV capture.",
        examples=["direct_pcm", "segmented_wav"],
    )
    vdo_segment_duration_seconds: Union[Selector(kind=[FLOAT_KIND, INTEGER_KIND]), float] = Field(
        default=DEFAULT_VDO_SEGMENT_DURATION_SECONDS,
        description="Segment duration for segmented VDO ingest mode.",
        examples=[30.0],
    )
    vdo_segment_startup_seconds: Union[Selector(kind=[FLOAT_KIND, INTEGER_KIND]), float] = Field(
        default=DEFAULT_VDO_SEGMENT_STARTUP_SECONDS,
        description="Initial warmup before the first segmented VDO capture begins.",
        examples=[8.0],
    )
    livepeer_capability: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=DEFAULT_TRUE_STREAMING_CAPABILITY,
        examples=[DEFAULT_TRUE_STREAMING_CAPABILITY],
    )
    livepeer_offering: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=DEFAULT_TRUE_STREAMING_OFFERING,
        examples=[DEFAULT_TRUE_STREAMING_OFFERING],
    )
    livepeer_estimated_runway_units: Optional[int] = Field(default=None, examples=[105])
    livepeer_max_total_units: Optional[int] = Field(default=None, examples=[135])

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="transcription_session", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="pcm_descriptor", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="session_id", kind=[STRING_KIND]),
            OutputDefinition(name="stream_id", kind=[STRING_KIND]),
            OutputDefinition(name="status", kind=[STRING_KIND]),
            OutputDefinition(name="source_mode", kind=[STRING_KIND]),
            OutputDefinition(name="source", kind=[STRING_KIND]),
            OutputDefinition(name="vdo_ingest_mode", kind=[STRING_KIND]),
            OutputDefinition(name="sent_audio_seconds", kind=[FLOAT_KIND]),
            OutputDefinition(name="sent_frame_count", kind=[INTEGER_KIND]),
            OutputDefinition(name="text", kind=[STRING_KIND]),
            OutputDefinition(name="speaker_count", kind=[INTEGER_KIND]),
            OutputDefinition(name="speakers", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="transcript_events", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="transcript_event_count", kind=[INTEGER_KIND]),
            OutputDefinition(name="events_jsonl_path", kind=[STRING_KIND]),
            OutputDefinition(name="result_json_path", kind=[STRING_KIND]),
            OutputDefinition(name="transcript_text_path", kind=[STRING_KIND]),
            OutputDefinition(name="publisher_log_path", kind=[STRING_KIND]),
            OutputDefinition(name="ffmpeg_log_path", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=False, reason="requires_vdo_ninja_and_runner_websocket")

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class LivepeerTrueStreamingTranscriptionSessionV1(WorkflowBlock):
    def __init__(
        self,
        runner_url: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_OPEN_CLEARINGHOUSE_URL,
        transcription_backend: str = DEFAULT_TRUE_STREAMING_TRANSCRIPTION_BACKEND,
        livepeer_capability: str = DEFAULT_TRUE_STREAMING_CAPABILITY,
        livepeer_offering: str = DEFAULT_TRUE_STREAMING_OFFERING,
        client_cls: Type[NemoTrueStreamingWebSocketClient] = NemoTrueStreamingWebSocketClient,
    ) -> None:
        self._runner_url = runner_url or init_nemo_diarized_runner_url()
        self._api_key = api_key or init_open_clearinghouse_api_key()
        self._base_url = base_url or init_open_clearinghouse_url()
        self._transcription_backend = (
            transcription_backend or init_true_streaming_transcription_backend()
        )
        self._livepeer_capability = livepeer_capability or init_true_streaming_capability()
        self._livepeer_offering = livepeer_offering or init_true_streaming_offering()
        self._client_cls = client_cls

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return [
            "runner_url",
            "api_key",
            "base_url",
            "transcription_backend",
            "livepeer_capability",
            "livepeer_offering",
        ]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LivepeerTrueStreamingTranscriptionSessionManifest

    def run(
        self,
        pcm_descriptor: Dict[str, Any],
        duration_seconds: float = 60.0,
        startup_timeout_seconds: float = 20.0,
        output_dir: str = str(DEFAULT_INGEST_OUTPUT_DIR / "direct-true-streaming"),
        session_id: str = "",
        language: str = "en",
        preset: str = "meeting",
        max_speakers: int = 4,
        transcription_backend: str = "",
        vdo_ingest_mode: str = DEFAULT_VDO_TRUE_STREAMING_INGEST_MODE,
        vdo_segment_duration_seconds: float = DEFAULT_VDO_SEGMENT_DURATION_SECONDS,
        vdo_segment_startup_seconds: float = DEFAULT_VDO_SEGMENT_STARTUP_SECONDS,
        livepeer_capability: str = "",
        livepeer_offering: str = "",
        livepeer_estimated_runway_units: Optional[int] = None,
        livepeer_max_total_units: Optional[int] = None,
    ) -> BlockResult:
        if not isinstance(pcm_descriptor, dict):
            raise ValueError("pcm_descriptor must be a dictionary")
        if pcm_descriptor.get("kind") != "pcm16_audio_stream":
            raise ValueError("pcm_descriptor must describe a pcm16_audio_stream")
        source_descriptor = pcm_descriptor.get("source_descriptor")
        if not isinstance(source_descriptor, dict):
            raise ValueError("pcm_descriptor.source_descriptor must be a dictionary")
        if source_descriptor.get("source_type") != "vdo_ninja":
            raise ValueError("only vdo_ninja live sources are supported by this runner block")
        resolved_ingest_mode = _normalize_vdo_true_streaming_ingest_mode(vdo_ingest_mode)
        client_cls, client_base_url, client_init_kwargs = _resolve_true_streaming_client(
            backend=transcription_backend or self._transcription_backend,
            local_client_cls=self._client_cls,
            runner_url=self._runner_url,
            api_key=self._api_key,
            base_url=self._base_url,
            livepeer_capability=livepeer_capability or self._livepeer_capability,
            livepeer_offering=livepeer_offering or self._livepeer_offering,
            duration_seconds=float(duration_seconds),
            livepeer_estimated_runway_units=livepeer_estimated_runway_units,
            livepeer_max_total_units=livepeer_max_total_units,
        )
        resolved_sample_rate = int(pcm_descriptor.get("sample_rate") or 16000)
        resolved_frame_duration_seconds = float(
            pcm_descriptor.get("frame_duration_seconds") or 0.08
        )
        source = str(source_descriptor["source"])
        password = str(source_descriptor.get("password") or "")
        signaling_server = str(source_descriptor.get("signaling_server") or "")
        buffer_ms = int(source_descriptor.get("buffer_ms") or 300)
        if resolved_ingest_mode == "segmented_wav":
            if float(duration_seconds) <= 0:
                raise ValueError(
                    "segmented_wav VDO ingest mode requires positive duration_seconds"
                )
            effective_segment_duration_seconds = min(
                float(vdo_segment_duration_seconds),
                float(duration_seconds),
            )
            if effective_segment_duration_seconds <= 0:
                raise ValueError("vdo_segment_duration_seconds must be positive")
            effective_segment_count = max(
                1,
                int(math.ceil(float(duration_seconds) / effective_segment_duration_seconds)),
            )
            if client_cls is not LivepeerRemoteHttpChunkingClient:
                client_init_kwargs = {
                    **client_init_kwargs,
                    "initial_receive_timeout_seconds": float(startup_timeout_seconds),
                }
            if client_cls in {
                LivepeerRemoteFallbackTransportClient,
                LivepeerRemoteHttpChunkingClient,
            }:
                client_init_kwargs = {
                    **client_init_kwargs,
                    "chunk_size_seconds": _livepeer_remote_http_chunk_size_seconds(
                        window_seconds=effective_segment_duration_seconds,
                        frame_duration_seconds=resolved_frame_duration_seconds,
                    ),
                }
            runner = build_vdo_true_streaming_runner(
                source=source,
                runner_url=client_base_url,
                output_dir=output_dir,
                segment_count=effective_segment_count,
                segment_duration_seconds=effective_segment_duration_seconds,
                startup_seconds=float(vdo_segment_startup_seconds),
                session_id=session_id,
                password=password,
                signaling_server=signaling_server,
                buffer_ms=buffer_ms,
                audio_only=True,
                language=language,
                preset=preset,
                max_speakers=int(max_speakers),
                sample_rate=resolved_sample_rate,
                frame_duration_seconds=resolved_frame_duration_seconds,
                client_cls=client_cls,
                client_init_kwargs=client_init_kwargs,
            )
            result = runner.run()
            sent_audio_seconds, sent_frame_count = _segmented_vdo_transport_metrics(
                captured_segments=list(result.get("captured_segments") or []),
                frame_duration_seconds=resolved_frame_duration_seconds,
            )
            result.update(
                {
                    "source_mode": "vdo_ninja_segmented_wav",
                    "source": source,
                    "sent_audio_seconds": sent_audio_seconds,
                    "sent_frame_count": sent_frame_count,
                    "publisher_log_path": "",
                    "ffmpeg_log_path": "",
                    "vdo_ingest_mode": resolved_ingest_mode,
                }
            )
        else:
            runner = build_vdo_direct_true_streaming_runner(
                source=source,
                runner_url=client_base_url,
                output_dir=output_dir,
                duration_seconds=float(duration_seconds),
                startup_timeout_seconds=float(startup_timeout_seconds),
                session_id=session_id,
                password=password,
                signaling_server=signaling_server,
                buffer_ms=buffer_ms,
                language=language,
                preset=preset,
                max_speakers=int(max_speakers),
                sample_rate=resolved_sample_rate,
                frame_duration_seconds=resolved_frame_duration_seconds,
                client_cls=client_cls,
                client_init_kwargs=client_init_kwargs,
            )
            result = runner.run()
            result["vdo_ingest_mode"] = resolved_ingest_mode
        result["pcm_descriptor"] = pcm_descriptor
        return {
            "transcription_session": result,
            "pcm_descriptor": pcm_descriptor,
            "session_id": result["session_id"],
            "stream_id": result["stream_id"],
            "status": result["status"],
            "source_mode": result["source_mode"],
            "source": result["source"],
            "vdo_ingest_mode": result["vdo_ingest_mode"],
            "sent_audio_seconds": result["sent_audio_seconds"],
            "sent_frame_count": result["sent_frame_count"],
            "text": result["text"],
            "speaker_count": result["speaker_count"],
            "speakers": result["speakers"],
            "transcript_events": result["transcript_events"],
            "transcript_event_count": result["transcript_event_count"],
            "events_jsonl_path": result["events_jsonl_path"],
            "result_json_path": result["result_json_path"],
            "transcript_text_path": result["transcript_text_path"],
            "publisher_log_path": result["publisher_log_path"],
            "ffmpeg_log_path": result["ffmpeg_log_path"],
        }


class LivepeerLocalAudioIngressTrueStreamingTranscriptionSessionManifest(
    WorkflowBlockManifest
):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Livepeer Local Ingest True Streaming Transcription Session",
            "version": "v1",
            "short_description": "Run a localhost ingest PCM16 descriptor through the true-streaming runner.",
            "long_description": LOCAL_INGEST_TRUE_STREAMING_TRANSCRIPTION_LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "livepeer",
                "localhost",
                "audio",
                "transcription",
                "true-streaming",
                "websocket",
                "ingest",
            ],
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-waveform-lines",
                "blockPriority": 13,
            },
        }
    )

    type: Literal[
        "roboflow_livepeer_blocks/livepeer_local_audio_ingest_true_streaming_transcription_session@v1",
        "LivepeerLocalAudioIngressTrueStreamingTranscriptionSession",
    ]
    pcm_descriptor: Union[Selector(kind=[DICTIONARY_KIND]), Dict[str, Any]] = Field(
        description="PCM16 descriptor emitted by the local ingest source block.",
        examples=["$steps.pcm16_transform.pcm_descriptor"],
    )
    duration_seconds: Union[Selector(kind=[FLOAT_KIND, INTEGER_KIND]), float] = Field(
        default=60.0,
        examples=[60.0],
    )
    startup_timeout_seconds: Union[Selector(kind=[FLOAT_KIND, INTEGER_KIND]), float] = Field(
        default=20.0,
        examples=[20.0],
    )
    output_dir: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=str(DEFAULT_INGEST_OUTPUT_DIR / "local-ingest-true-streaming"),
        examples=[str(DEFAULT_INGEST_OUTPUT_DIR / "local-ingest-true-streaming")],
    )
    session_id: Union[Selector(kind=[STRING_KIND]), str] = Field(default="", examples=[""])
    language: Union[Selector(kind=[STRING_KIND]), str] = Field(default="en", examples=["en"])
    preset: Union[Selector(kind=[STRING_KIND]), str] = Field(default="meeting", examples=["meeting"])
    max_speakers: Union[Selector(kind=[INTEGER_KIND]), int] = Field(default=4, examples=[4])
    transcription_backend: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=DEFAULT_TRUE_STREAMING_TRANSCRIPTION_BACKEND,
        examples=["local", "livepeer_remote", "livepeer_remote_http"],
    )
    livepeer_capability: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=DEFAULT_TRUE_STREAMING_CAPABILITY,
        examples=[DEFAULT_TRUE_STREAMING_CAPABILITY],
    )
    livepeer_offering: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=DEFAULT_TRUE_STREAMING_OFFERING,
        examples=[DEFAULT_TRUE_STREAMING_OFFERING],
    )
    livepeer_estimated_runway_units: Optional[int] = Field(default=None, examples=[105])
    livepeer_max_total_units: Optional[int] = Field(default=None, examples=[135])

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return LivepeerTrueStreamingTranscriptionSessionManifest.describe_outputs()

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=True, reason="localhost_and_runner_only")

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class LivepeerLocalAudioIngressTrueStreamingTranscriptionSessionV1(WorkflowBlock):
    def __init__(
        self,
        runner_url: Optional[str] = None,
        local_audio_ingest_url: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_OPEN_CLEARINGHOUSE_URL,
        transcription_backend: str = "",
        livepeer_capability: str = "",
        livepeer_offering: str = "",
        client_cls: Type[NemoTrueStreamingWebSocketClient] = NemoTrueStreamingWebSocketClient,
        ingest_client_cls: Type[LocalAudioIngestWebSocketClient] = LocalAudioIngestWebSocketClient,
    ) -> None:
        self._runner_url = runner_url or init_nemo_diarized_runner_url()
        self._local_audio_ingest_url = local_audio_ingest_url or init_local_audio_ingest_url()
        self._api_key = api_key or init_open_clearinghouse_api_key()
        self._base_url = base_url or init_open_clearinghouse_url()
        self._transcription_backend = (
            transcription_backend or init_true_streaming_transcription_backend()
        )
        self._livepeer_capability = livepeer_capability or init_true_streaming_capability()
        self._livepeer_offering = livepeer_offering or init_true_streaming_offering()
        self._client_cls = client_cls
        self._ingest_client_cls = ingest_client_cls

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return [
            "runner_url",
            "local_audio_ingest_url",
            "api_key",
            "base_url",
            "transcription_backend",
            "livepeer_capability",
            "livepeer_offering",
        ]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LivepeerLocalAudioIngressTrueStreamingTranscriptionSessionManifest

    def run(
        self,
        pcm_descriptor: Dict[str, Any],
        duration_seconds: float = 60.0,
        startup_timeout_seconds: float = 20.0,
        output_dir: str = str(DEFAULT_INGEST_OUTPUT_DIR / "local-ingest-true-streaming"),
        session_id: str = "",
        language: str = "en",
        preset: str = "meeting",
        max_speakers: int = 4,
        transcription_backend: str = "",
        livepeer_capability: str = "",
        livepeer_offering: str = "",
        livepeer_estimated_runway_units: Optional[int] = None,
        livepeer_max_total_units: Optional[int] = None,
    ) -> BlockResult:
        if not isinstance(pcm_descriptor, dict):
            raise ValueError("pcm_descriptor must be a dictionary")
        if pcm_descriptor.get("kind") != "pcm16_audio_stream":
            raise ValueError("pcm_descriptor must describe a pcm16_audio_stream")
        source_descriptor = pcm_descriptor.get("source_descriptor")
        if not isinstance(source_descriptor, dict):
            raise ValueError("pcm_descriptor.source_descriptor must be a dictionary")
        if source_descriptor.get("source_type") != "localhost_ingest":
            raise ValueError("only localhost_ingest live sources are supported by this runner block")
        client_cls, client_base_url, client_init_kwargs = _resolve_true_streaming_client(
            backend=transcription_backend or self._transcription_backend,
            local_client_cls=self._client_cls,
            runner_url=self._runner_url,
            api_key=self._api_key,
            base_url=self._base_url,
            livepeer_capability=livepeer_capability or self._livepeer_capability,
            livepeer_offering=livepeer_offering or self._livepeer_offering,
            duration_seconds=float(duration_seconds),
            livepeer_estimated_runway_units=livepeer_estimated_runway_units,
            livepeer_max_total_units=livepeer_max_total_units,
        )
        runner = build_local_audio_ingest_true_streaming_runner(
            source=str(source_descriptor["source"]),
            runner_url=client_base_url,
            local_audio_ingest_url=self._local_audio_ingest_url,
            output_dir=output_dir,
            duration_seconds=float(duration_seconds),
            startup_timeout_seconds=float(startup_timeout_seconds),
            session_id=session_id,
            language=language,
            preset=preset,
            max_speakers=int(max_speakers),
            sample_rate=int(pcm_descriptor.get("sample_rate") or 16000),
            frame_duration_seconds=float(pcm_descriptor.get("frame_duration_seconds") or 0.08),
            client_cls=client_cls,
            ingest_client_cls=self._ingest_client_cls,
            client_init_kwargs=client_init_kwargs,
        )
        result = runner.run()
        result["pcm_descriptor"] = pcm_descriptor
        return {
            "transcription_session": result,
            "pcm_descriptor": pcm_descriptor,
            "session_id": result["session_id"],
            "stream_id": result["stream_id"],
            "status": result["status"],
            "source_mode": result["source_mode"],
            "source": result["source"],
            "sent_audio_seconds": result["sent_audio_seconds"],
            "sent_frame_count": result["sent_frame_count"],
            "text": result["text"],
            "speaker_count": result["speaker_count"],
            "speakers": result["speakers"],
            "transcript_events": result["transcript_events"],
            "transcript_event_count": result["transcript_event_count"],
            "events_jsonl_path": result["events_jsonl_path"],
            "result_json_path": result["result_json_path"],
            "transcript_text_path": result["transcript_text_path"],
            "publisher_log_path": result["publisher_log_path"],
            "ffmpeg_log_path": result["ffmpeg_log_path"],
        }


class LivepeerTranscriptOutputManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Livepeer Transcript Output",
            "version": "v1",
            "short_description": "Normalize a transcript session into workflow output fields.",
            "long_description": TRANSCRIPT_OUTPUT_LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "formatter",
            "search_keywords": [
                "livepeer",
                "transcript",
                "output",
                "formatter",
                "diarization",
                "composable",
            ],
            "ui_manifest": {
                "section": "formatters",
                "icon": "fal fa-file-lines",
                "blockPriority": 13,
            },
        }
    )

    type: Literal[
        "roboflow_livepeer_blocks/livepeer_transcript_output@v1",
        "LivepeerTranscriptOutput",
    ]
    transcription_session: Union[Selector(kind=[DICTIONARY_KIND]), Dict[str, Any]] = Field(
        description="Transcription session result emitted by the runner block.",
        examples=["$steps.true_streaming_transcription.transcription_session"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="session_id", kind=[STRING_KIND]),
            OutputDefinition(name="stream_id", kind=[STRING_KIND]),
            OutputDefinition(name="status", kind=[STRING_KIND]),
            OutputDefinition(name="source_mode", kind=[STRING_KIND]),
            OutputDefinition(name="source", kind=[STRING_KIND]),
            OutputDefinition(name="sent_audio_seconds", kind=[FLOAT_KIND]),
            OutputDefinition(name="sent_frame_count", kind=[INTEGER_KIND]),
            OutputDefinition(name="text", kind=[STRING_KIND]),
            OutputDefinition(name="speaker_count", kind=[INTEGER_KIND]),
            OutputDefinition(name="speakers", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="transcript_events", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="transcript_event_count", kind=[INTEGER_KIND]),
            OutputDefinition(name="event_counts", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="events_jsonl_path", kind=[STRING_KIND]),
            OutputDefinition(name="result_json_path", kind=[STRING_KIND]),
            OutputDefinition(name="transcript_text_path", kind=[STRING_KIND]),
            OutputDefinition(name="publisher_log_path", kind=[STRING_KIND]),
            OutputDefinition(name="ffmpeg_log_path", kind=[STRING_KIND]),
            OutputDefinition(name="result", kind=[DICTIONARY_KIND]),
        ]

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=True, reason="pure_formatter")

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class LivepeerTranscriptOutputV1(WorkflowBlock):
    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return []

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LivepeerTranscriptOutputManifest

    def run(self, transcription_session: Dict[str, Any]) -> BlockResult:
        if not isinstance(transcription_session, dict):
            raise ValueError("transcription_session must be a dictionary")
        session = transcription_session.get("transcription_session", transcription_session)
        if not isinstance(session, dict):
            raise ValueError("transcription_session payload must be a dictionary")
        events = session.get("events") or []
        event_counts: Dict[str, int] = {}
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, dict):
                    continue
                event_type = str(event.get("event_type") or "")
                if event_type:
                    event_counts[event_type] = event_counts.get(event_type, 0) + 1
        return {
            "session_id": str(session.get("session_id") or ""),
            "stream_id": str(session.get("stream_id") or ""),
            "status": str(session.get("status") or ""),
            "source_mode": str(session.get("source_mode") or ""),
            "source": str(session.get("source") or ""),
            "sent_audio_seconds": float(session.get("sent_audio_seconds") or 0.0),
            "sent_frame_count": int(session.get("sent_frame_count") or 0),
            "text": str(session.get("text") or ""),
            "speaker_count": int(session.get("speaker_count") or 0),
            "speakers": session.get("speakers") or [],
            "transcript_events": session.get("transcript_events") or [],
            "transcript_event_count": int(session.get("transcript_event_count") or 0),
            "event_counts": event_counts,
            "events_jsonl_path": str(session.get("events_jsonl_path") or ""),
            "result_json_path": str(session.get("result_json_path") or ""),
            "transcript_text_path": str(session.get("transcript_text_path") or ""),
            "publisher_log_path": str(session.get("publisher_log_path") or ""),
            "ffmpeg_log_path": str(session.get("ffmpeg_log_path") or ""),
            "result": session,
        }


class LivepeerScreenSlideCaptureManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Livepeer Screen Slide Capture",
            "version": "v1",
            "short_description": "Declare bounded screen and slide capture settings for a live video descriptor.",
            "long_description": SCREEN_SLIDE_CAPTURE_LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "search_keywords": [
                "livepeer",
                "video",
                "screen",
                "slides",
                "capture",
                "composable",
            ],
            "ui_manifest": {
                "section": "transformations",
                "icon": "fal fa-presentation-screen",
                "blockPriority": 14,
            },
        }
    )

    type: Literal[
        "roboflow_livepeer_blocks/livepeer_screen_slide_capture@v1",
        "LivepeerScreenSlideCapture",
    ]
    video_source_descriptor: Union[Selector(kind=[DICTIONARY_KIND]), Dict[str, Any]] = Field(
        description="Live video source descriptor emitted by the media source block.",
        examples=["$steps.media_source.video_source_descriptor"],
    )
    duration_seconds: Union[Selector(kind=[FLOAT_KIND, INTEGER_KIND]), float] = Field(
        default=0.0,
        description="Seconds of live video to capture. Use 0 to run until the source stream ends.",
        examples=[0.0, 60.0],
    )
    startup_seconds: Union[Selector(kind=[FLOAT_KIND, INTEGER_KIND]), float] = Field(
        default=8.0,
        examples=[8.0],
    )
    output_dir: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=str(DEFAULT_INGEST_OUTPUT_DIR / "screen-slide-capture"),
        examples=[str(DEFAULT_INGEST_OUTPUT_DIR / "screen-slide-capture")],
    )
    frame_interval_seconds: Union[Selector(kind=[FLOAT_KIND, INTEGER_KIND]), float] = Field(
        default=DEFAULT_FRAME_INTERVAL_SECONDS,
        examples=[DEFAULT_FRAME_INTERVAL_SECONDS],
    )
    max_frames: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        default=DEFAULT_MAX_FRAMES,
        examples=[DEFAULT_MAX_FRAMES],
    )
    min_slide_gap_seconds: Union[Selector(kind=[FLOAT_KIND, INTEGER_KIND]), float] = Field(
        default=DEFAULT_MIN_SLIDE_GAP_SECONDS,
        examples=[DEFAULT_MIN_SLIDE_GAP_SECONDS],
    )
    slide_change_threshold: Union[Selector(kind=[FLOAT_KIND, INTEGER_KIND]), float] = Field(
        default=DEFAULT_SLIDE_CHANGE_THRESHOLD,
        examples=[DEFAULT_SLIDE_CHANGE_THRESHOLD],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="capture_descriptor", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="video_source_descriptor", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="source", kind=[STRING_KIND]),
            OutputDefinition(name="stream_id", kind=[STRING_KIND]),
            OutputDefinition(name="output_dir", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=False, reason="inherits_live_video_source_requirements")

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class LivepeerScreenSlideCaptureV1(WorkflowBlock):
    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return []

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LivepeerScreenSlideCaptureManifest

    def run(
        self,
        video_source_descriptor: Dict[str, Any],
        duration_seconds: float = 0.0,
        startup_seconds: float = 8.0,
        output_dir: str = str(DEFAULT_INGEST_OUTPUT_DIR / "screen-slide-capture"),
        frame_interval_seconds: float = DEFAULT_FRAME_INTERVAL_SECONDS,
        max_frames: int = DEFAULT_MAX_FRAMES,
        min_slide_gap_seconds: float = DEFAULT_MIN_SLIDE_GAP_SECONDS,
        slide_change_threshold: float = DEFAULT_SLIDE_CHANGE_THRESHOLD,
    ) -> BlockResult:
        if not isinstance(video_source_descriptor, dict):
            raise ValueError("video_source_descriptor must be a dictionary")
        if video_source_descriptor.get("kind") != "live_video_source":
            raise ValueError("video_source_descriptor must describe a live_video_source")

        capture_descriptor = {
            "schema_version": "livepeer.screen_slide_capture.v1",
            "kind": "screen_slide_capture",
            "video_source_descriptor": video_source_descriptor,
            "source_type": str(video_source_descriptor.get("source_type") or ""),
            "source": str(video_source_descriptor.get("source") or ""),
            "stream_id": str(video_source_descriptor.get("stream_id") or ""),
            "duration_seconds": float(duration_seconds),
            "startup_seconds": float(startup_seconds),
            "output_dir": output_dir,
            "frame_interval_seconds": float(frame_interval_seconds),
            "max_frames": int(max_frames),
            "min_slide_gap_seconds": float(min_slide_gap_seconds),
            "slide_change_threshold": float(slide_change_threshold),
            "analysis_model": {
                "family": "florence-2",
                "model_id": DEFAULT_FLORENCE2_MODEL_ID,
                "tasks": ["<CAPTION>", "<DETAILED_CAPTION>", "<OCR>"],
                "vision_backend": DEFAULT_VISION_BACKEND,
                "supported_vision_backends": ["local", "remote"],
                "remote_primary_endpoint": "/v1/vision/analyze",
                "remote_compatibility_endpoint": "/infer/lmm",
                "meeting_context_prompt": DEFAULT_MEETING_CONTEXT_PROMPT,
                "meeting_context_prompt_supported": "remote_optional",
            },
            "execution": {
                "starts_capture": False,
                "capture_owned_by_consumer_block": True,
            },
        }
        return {
            "capture_descriptor": capture_descriptor,
            "video_source_descriptor": video_source_descriptor,
            "source": capture_descriptor["source"],
            "stream_id": capture_descriptor["stream_id"],
            "output_dir": output_dir,
        }


class LivepeerFlorence2ScreenSlideAnalysisManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Livepeer Florence-2 Screen Slide Analysis",
            "version": "v1",
            "short_description": "Capture live video frames and analyze them with Florence-2 for screen and slide events.",
            "long_description": FLORENCE2_SCREEN_SLIDE_ANALYSIS_LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "livepeer",
                "florence-2",
                "screen",
                "slides",
                "ocr",
                "vision",
                "composable",
            ],
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-screen-users",
                "blockPriority": 15,
            },
        }
    )

    type: Literal[
        "roboflow_livepeer_blocks/livepeer_florence2_screen_slide_analysis@v1",
        "LivepeerFlorence2ScreenSlideAnalysis",
    ]
    capture_descriptor: Union[Selector(kind=[DICTIONARY_KIND]), Dict[str, Any]] = Field(
        description="Capture descriptor emitted by the screen slide capture block.",
        examples=["$steps.screen_capture.capture_descriptor"],
    )
    output_dir: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=str(DEFAULT_INGEST_OUTPUT_DIR / "florence-screen-slide"),
        examples=[str(DEFAULT_INGEST_OUTPUT_DIR / "florence-screen-slide")],
    )
    model_id: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=DEFAULT_FLORENCE2_MODEL_ID,
        examples=[DEFAULT_FLORENCE2_MODEL_ID],
    )
    meeting_context_prompt: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default="",
        description=(
            "Optional free-form meeting context prompt for remote Florence-2 runners. "
            "The local fallback still separates text with heuristics when this is empty or unsupported."
        ),
        examples=[DEFAULT_MEETING_CONTEXT_PROMPT],
    )
    vision_backend: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=DEFAULT_VISION_BACKEND,
        description="Vision backend: local Florence runtime or remote Florence runner.",
        examples=["local", "remote"],
    )
    florence2_runner_url: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default="",
        description=(
            "Remote Florence runner base URL when vision_backend is remote. "
            "Defaults to FLORENCE2_RUNNER_URL."
        ),
        examples=[DEFAULT_FLORENCE2_RUNNER_URL],
    )
    livepeer_capability: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=DEFAULT_VISION_CAPABILITY,
        description="Livepeer capability used when the remote backend is brokered through the clearinghouse.",
        examples=[DEFAULT_VISION_CAPABILITY],
    )
    livepeer_offering: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=DEFAULT_VISION_OFFERING,
        description="Livepeer offering used when the remote backend is brokered through the clearinghouse.",
        examples=[DEFAULT_VISION_OFFERING],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="analysis_session", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="analysis_id", kind=[STRING_KIND]),
            OutputDefinition(name="stream_id", kind=[STRING_KIND]),
            OutputDefinition(name="source", kind=[STRING_KIND]),
            OutputDefinition(name="status", kind=[STRING_KIND]),
            OutputDefinition(name="recording_path", kind=[STRING_KIND]),
            OutputDefinition(name="frame_count", kind=[INTEGER_KIND]),
            OutputDefinition(name="slide_count", kind=[INTEGER_KIND]),
            OutputDefinition(name="sampled_frames", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="visual_events", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="meeting_visual_events", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="meeting_visual_summary", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="slide_text", kind=[STRING_KIND]),
            OutputDefinition(name="screen_share_text", kind=[STRING_KIND]),
            OutputDefinition(name="chat_text", kind=[STRING_KIND]),
            OutputDefinition(name="call_ui_text", kind=[STRING_KIND]),
            OutputDefinition(name="browser_or_player_chrome_text", kind=[STRING_KIND]),
            OutputDefinition(name="slides", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="slides_manifest_path", kind=[STRING_KIND]),
            OutputDefinition(name="events_jsonl_path", kind=[STRING_KIND]),
            OutputDefinition(name="meeting_events_jsonl_path", kind=[STRING_KIND]),
            OutputDefinition(name="result_json_path", kind=[STRING_KIND]),
            OutputDefinition(name="result", kind=[DICTIONARY_KIND]),
        ]

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=False, reason="requires_live_video_capture_and_visual_model")

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class LivepeerFlorence2ScreenSlideAnalysisV1(WorkflowBlock):
    def __init__(
        self,
        roboflow_api_key: Optional[str] = None,
        roboflow_inference_url: str = "",
        vision_backend: Optional[str] = None,
        florence2_runner_url: Optional[str] = None,
        livepeer_api_key: Optional[str] = None,
        livepeer_base_url: str = DEFAULT_OPEN_CLEARINGHOUSE_URL,
        vision_capability: Optional[str] = None,
        vision_offering: Optional[str] = None,
        analyzer_cls: Type[Florence2InferenceAnalyzer] = Florence2InferenceAnalyzer,
    ) -> None:
        self._roboflow_api_key = roboflow_api_key
        self._roboflow_inference_url = roboflow_inference_url
        self._vision_backend = vision_backend or init_vision_backend()
        self._florence2_runner_url = florence2_runner_url or init_florence2_runner_url()
        self._livepeer_api_key = livepeer_api_key or init_open_clearinghouse_api_key()
        self._livepeer_base_url = livepeer_base_url or init_open_clearinghouse_url()
        self._vision_capability = vision_capability or init_vision_capability()
        self._vision_offering = vision_offering or init_vision_offering()
        self._analyzer_cls = analyzer_cls

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return [
            "roboflow_api_key",
            "roboflow_inference_url",
            "vision_backend",
            "florence2_runner_url",
            "livepeer_api_key",
            "livepeer_base_url",
            "vision_capability",
            "vision_offering",
        ]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LivepeerFlorence2ScreenSlideAnalysisManifest

    def run(
        self,
        capture_descriptor: Dict[str, Any],
        output_dir: str = str(DEFAULT_INGEST_OUTPUT_DIR / "florence-screen-slide"),
        model_id: str = DEFAULT_FLORENCE2_MODEL_ID,
        meeting_context_prompt: str = "",
        vision_backend: str = "",
        florence2_runner_url: str = "",
        livepeer_capability: str = DEFAULT_VISION_CAPABILITY,
        livepeer_offering: str = DEFAULT_VISION_OFFERING,
    ) -> BlockResult:
        if not isinstance(capture_descriptor, dict):
            raise ValueError("capture_descriptor must be a dictionary")
        if capture_descriptor.get("kind") != "screen_slide_capture":
            raise ValueError("capture_descriptor must describe a screen_slide_capture")

        video_source_descriptor = capture_descriptor.get("video_source_descriptor")
        if not isinstance(video_source_descriptor, dict):
            raise ValueError("capture_descriptor.video_source_descriptor must be a dictionary")
        if video_source_descriptor.get("source_type") != "vdo_ninja":
            raise ValueError("only vdo_ninja live video sources are supported by this analysis block")

        source = str(video_source_descriptor.get("source") or capture_descriptor.get("source") or "")
        stream_id = str(video_source_descriptor.get("stream_id") or capture_descriptor.get("stream_id") or "")
        artifact_dir = _prepare_visual_artifact_dir(output_dir=output_dir, source=source or stream_id)

        recording_segment = record_vdo_segment(
            source=source,
            output_dir=artifact_dir,
            duration_seconds=float(capture_descriptor.get("duration_seconds") or 0.0),
            startup_seconds=float(capture_descriptor.get("startup_seconds") or 0.0),
            password=str(video_source_descriptor.get("password") or ""),
            signaling_server=str(video_source_descriptor.get("signaling_server") or ""),
            buffer_ms=int(video_source_descriptor.get("buffer_ms") or 300),
            audio_only=False,
            allow_missing_audio=True,
            segment_index=0,
        )
        frame_interval_seconds = float(capture_descriptor.get("frame_interval_seconds") or DEFAULT_FRAME_INTERVAL_SECONDS)
        frame_samples = extract_video_frames(
            recording_path=recording_segment.recording_path,
            output_dir=artifact_dir / "frames",
            frame_interval_seconds=frame_interval_seconds,
            max_frames=int(capture_descriptor.get("max_frames") or DEFAULT_MAX_FRAMES),
        )

        resolved_vision_backend = vision_backend or self._vision_backend
        resolved_florence2_runner_url = florence2_runner_url or self._florence2_runner_url
        resolved_livepeer_capability = livepeer_capability or self._vision_capability
        resolved_livepeer_offering = livepeer_offering or self._vision_offering
        normalized_vision_backend = str(resolved_vision_backend or "").strip().lower().replace("-", "_")
        use_livepeer_gateway = normalized_vision_backend in {
            "livepeer_remote",
            "livepeer",
            "livepeer_gateway",
        }
        use_direct_runner = normalized_vision_backend == "remote"
        if use_livepeer_gateway and not self._livepeer_api_key:
            raise ValueError(
                "vision_backend=livepeer_remote requires LIVEPEER_OPEN_CLEARINGHOUSE_API_KEY "
                "or an explicit livepeer_api_key block initializer"
            )
        try:
            analyzer = self._analyzer_cls(
                model_id=model_id,
                api_key=self._roboflow_api_key,
                api_url=self._roboflow_inference_url,
                vision_backend=resolved_vision_backend,
                runner_url=resolved_florence2_runner_url,
                livepeer_api_key=self._livepeer_api_key if use_livepeer_gateway else None,
                livepeer_base_url=self._livepeer_base_url,
                livepeer_capability=resolved_livepeer_capability,
                livepeer_offering=resolved_livepeer_offering,
            )
        except TypeError:
            analyzer = self._analyzer_cls(
                model_id=model_id,
                api_key=self._roboflow_api_key,
                api_url=self._roboflow_inference_url,
            )
        visual_events: List[Dict[str, Any]] = []
        meeting_visual_events: List[Dict[str, Any]] = []
        slides: List[Dict[str, Any]] = []
        prior_slide_signal = ""
        prior_slide_timestamp = -10_000.0
        min_slide_gap_seconds = float(
            capture_descriptor.get("min_slide_gap_seconds") or DEFAULT_MIN_SLIDE_GAP_SECONDS
        )
        slide_change_threshold = float(
            capture_descriptor.get("slide_change_threshold") or DEFAULT_SLIDE_CHANGE_THRESHOLD
        )

        for frame in frame_samples:
            try:
                analysis = analyzer.analyze_image(
                    frame["image_path"],
                    meeting_context_prompt=meeting_context_prompt,
                )
            except TypeError:
                analysis = analyzer.analyze_image(frame["image_path"])
            caption = str(analysis.get("caption") or "")
            detailed_caption = str(analysis.get("detailed_caption") or "")
            ocr_text = str(analysis.get("ocr_text") or "")
            meeting_context = analysis.get("meeting_context") or {}
            meeting_text = separate_meeting_visual_text(
                caption=caption,
                detailed_caption=detailed_caption,
                ocr_text=ocr_text,
                meeting_context=meeting_context.get("text") if isinstance(meeting_context, dict) else "",
            )
            is_presentation_frame = presentation_likelihood(
                caption=caption,
                detailed_caption=detailed_caption,
                ocr_text="\n".join(
                    part
                    for part in (
                        meeting_text["slide_text"],
                        meeting_text["screen_share_text"],
                    )
                    if part
                )
                or ocr_text,
            )
            content_text = "\n".join(
                part
                for part in (
                    meeting_text["slide_text"],
                    meeting_text["screen_share_text"],
                )
                if part
            )
            slide_signal = make_slide_signal_text(
                caption=caption,
                detailed_caption=detailed_caption,
                ocr_text=content_text or ocr_text,
            )
            similarity = similarity_ratio(prior_slide_signal, slide_signal)
            timestamp_seconds = float(frame["timestamp_seconds"])
            is_content_frame = bool(content_text) or any(
                role in meeting_text["content_roles"] for role in ("slide", "screen_share")
            )
            slide_changed = bool(
                is_presentation_frame
                and is_content_frame
                and (
                    not prior_slide_signal
                    or (
                        timestamp_seconds - prior_slide_timestamp >= min_slide_gap_seconds
                        and similarity < slide_change_threshold
                    )
                )
            )
            event = {
                **frame,
                "caption": caption,
                "detailed_caption": detailed_caption,
                "ocr_text": ocr_text,
                "meeting_context": meeting_context,
                "meeting_visual_text": meeting_text,
                "slide_text": meeting_text["slide_text"],
                "screen_share_text": meeting_text["screen_share_text"],
                "chat_text": meeting_text["chat_text"],
                "call_ui_text": meeting_text["call_ui_text"],
                "browser_or_player_chrome_text": meeting_text["browser_or_player_chrome_text"],
                "other_page_chrome_text": meeting_text["other_page_chrome_text"],
                "content_roles": meeting_text["content_roles"],
                "primary_content_role": meeting_text["primary_content_role"],
                "separation_confidence": meeting_text["separation_confidence"],
                "is_presentation_frame": is_presentation_frame,
                "similarity_to_previous_slide": round(similarity, 4),
                "slide_changed": slide_changed,
            }
            visual_events.append(event)
            meeting_visual_events.append(
                {
                    "schema_version": "livepeer.meeting_visual_event.v1",
                    "event_type": "meeting.visual.frame",
                    "index": frame["index"],
                    "timestamp_seconds": timestamp_seconds,
                    "image_path": frame["image_path"],
                    "caption": caption,
                    "detailed_caption": detailed_caption,
                    "content_roles": meeting_text["content_roles"],
                    "primary_content_role": meeting_text["primary_content_role"],
                    "slide_text": meeting_text["slide_text"],
                    "screen_share_text": meeting_text["screen_share_text"],
                    "chat_text": meeting_text["chat_text"],
                    "call_ui_text": meeting_text["call_ui_text"],
                    "browser_or_player_chrome_text": meeting_text["browser_or_player_chrome_text"],
                    "other_page_chrome_text": meeting_text["other_page_chrome_text"],
                    "raw_ocr_text": meeting_text["raw_ocr_text"],
                    "separation_confidence": meeting_text["separation_confidence"],
                    "is_presentation_frame": is_presentation_frame,
                    "slide_changed": slide_changed,
                }
            )
            if slide_changed:
                slide_index = len(slides) + 1
                slide_image_path = copy_slide_frame(
                    source_path=frame["image_path"],
                    slides_dir=artifact_dir / "slides",
                    slide_index=slide_index,
                )
                slide_entry = {
                    "index": slide_index,
                    "timestamp_seconds": timestamp_seconds,
                    "image_path": slide_image_path,
                    "source_frame_path": frame["image_path"],
                    "caption": caption,
                    "detailed_caption": detailed_caption,
                    "ocr_text": ocr_text,
                    "slide_text": meeting_text["slide_text"],
                    "screen_share_text": meeting_text["screen_share_text"],
                    "chat_text": meeting_text["chat_text"],
                    "call_ui_text": meeting_text["call_ui_text"],
                    "browser_or_player_chrome_text": meeting_text["browser_or_player_chrome_text"],
                    "meeting_visual_text": meeting_text,
                    "similarity_to_previous_slide": round(similarity, 4),
                }
                slides.append(slide_entry)
                prior_slide_signal = slide_signal
                prior_slide_timestamp = timestamp_seconds

        analysis_id = f"{stream_id or parse_vdo_stream_id(source)}-visual-{int(time.time())}"
        meeting_visual_summary = summarize_meeting_visual_events(meeting_visual_events)
        remote_transport = ""
        if use_livepeer_gateway:
            remote_transport = "livepeer_clearinghouse"
        elif use_direct_runner:
            remote_transport = "direct_runner"
        remote_primary_endpoint = (
            "/v1/chat/completions"
            if remote_transport == "livepeer_clearinghouse"
            else "/v1/vision/analyze"
        )
        slides_manifest_payload = {
            "schema_version": "livepeer.screen_slide_manifest.v1",
            "analysis_id": analysis_id,
            "stream_id": stream_id,
            "source": source,
            "model_id": model_id,
            "vision_backend": resolved_vision_backend,
            "remote_transport": remote_transport,
            "livepeer_capability": (
                resolved_livepeer_capability if remote_transport == "livepeer_clearinghouse" else ""
            ),
            "livepeer_offering": (
                resolved_livepeer_offering if remote_transport == "livepeer_clearinghouse" else ""
            ),
            "remote_primary_endpoint": remote_primary_endpoint,
            "remote_compatibility_endpoint": "/infer/lmm",
            "frame_interval_seconds": frame_interval_seconds,
            "meeting_visual_summary": meeting_visual_summary,
            "slides": slides,
        }
        analysis_session = {
            "schema_version": "livepeer.florence2_screen_slide_analysis.v1",
            "analysis_id": analysis_id,
            "stream_id": stream_id,
            "source": source,
            "status": "completed" if frame_samples else "no_frames",
            "model_id": model_id,
            "vision_backend": resolved_vision_backend,
            "remote_transport": remote_transport,
            "florence2_runner_url": resolved_florence2_runner_url
            if remote_transport == "direct_runner"
            else "",
            "livepeer_capability": (
                resolved_livepeer_capability if remote_transport == "livepeer_clearinghouse" else ""
            ),
            "livepeer_offering": (
                resolved_livepeer_offering if remote_transport == "livepeer_clearinghouse" else ""
            ),
            "remote_primary_endpoint": remote_primary_endpoint,
            "remote_compatibility_endpoint": "/infer/lmm",
            "capture_descriptor": capture_descriptor,
            "recording_path": str(recording_segment.recording_path),
            "audio_path": str(recording_segment.audio_path),
            "publisher_log_path": str(recording_segment.log_path),
            "frame_count": len(frame_samples),
            "slide_count": len(slides),
            "sampled_frames": frame_samples,
            "visual_events": visual_events,
            "meeting_visual_events": meeting_visual_events,
            "meeting_visual_summary": meeting_visual_summary,
            "slide_text": meeting_visual_summary["slide_text"],
            "screen_share_text": meeting_visual_summary["screen_share_text"],
            "chat_text": meeting_visual_summary["chat_text"],
            "call_ui_text": meeting_visual_summary["call_ui_text"],
            "browser_or_player_chrome_text": meeting_visual_summary["browser_or_player_chrome_text"],
            "slides": slides,
        }
        slides_manifest_path = write_json(artifact_dir / "slides" / "manifest.json", slides_manifest_payload)
        events_jsonl_path = write_jsonl(artifact_dir / "visual-events.jsonl", visual_events)
        meeting_events_jsonl_path = write_jsonl(
            artifact_dir / "meeting-visual-events.jsonl",
            meeting_visual_events,
        )
        analysis_session["slides_manifest_path"] = slides_manifest_path
        analysis_session["events_jsonl_path"] = events_jsonl_path
        analysis_session["meeting_events_jsonl_path"] = meeting_events_jsonl_path
        result_json_path = str((artifact_dir / "result.json").resolve())
        analysis_session["result_json_path"] = result_json_path
        write_json(result_json_path, analysis_session)
        return {
            "analysis_session": analysis_session,
            "analysis_id": analysis_id,
            "stream_id": stream_id,
            "source": source,
            "status": analysis_session["status"],
            "recording_path": str(recording_segment.recording_path),
            "frame_count": len(frame_samples),
            "slide_count": len(slides),
            "sampled_frames": frame_samples,
            "visual_events": visual_events,
            "meeting_visual_events": meeting_visual_events,
            "meeting_visual_summary": meeting_visual_summary,
            "slide_text": meeting_visual_summary["slide_text"],
            "screen_share_text": meeting_visual_summary["screen_share_text"],
            "chat_text": meeting_visual_summary["chat_text"],
            "call_ui_text": meeting_visual_summary["call_ui_text"],
            "browser_or_player_chrome_text": meeting_visual_summary["browser_or_player_chrome_text"],
            "slides": slides,
            "slides_manifest_path": slides_manifest_path,
            "events_jsonl_path": events_jsonl_path,
            "meeting_events_jsonl_path": meeting_events_jsonl_path,
            "result_json_path": result_json_path,
            "result": analysis_session,
        }


class LivepeerVDONinjaDirectTrueStreamingSessionManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Livepeer VDO.Ninja Direct True Streaming Session",
            "version": "v1",
            "short_description": "Pipe live VDO.Ninja audio directly into the runner WebSocket.",
            "long_description": DIRECT_TRUE_STREAMING_SESSION_LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "livepeer",
                "vdo.ninja",
                "audio",
                "transcription",
                "true-streaming",
                "websocket",
                "direct",
                "nemo",
            ],
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-tower-broadcast",
                "blockPriority": 9,
            },
        }
    )

    type: Literal[
        "roboflow_livepeer_blocks/livepeer_vdo_ninja_direct_true_streaming_session@v1",
        "LivepeerVDONinjaDirectTrueStreamingSession",
    ]
    source: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default="auto",
        description='Raw VDO.Ninja stream ID, viewer URL containing ?view=..., bridge URL, or "auto" to select the newest live bridge stream.',
        examples=["auto", "wss://vdo-signaling-bridge:9443", "stream_9xc43b5s6"],
    )
    signaling_server: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default="",
        description='Optional custom VDO-compatible signaling server, for example "wss://localhost:9443".',
        examples=["wss://localhost:9443", ""],
    )
    duration_seconds: Union[Selector(kind=[FLOAT_KIND, INTEGER_KIND]), float] = Field(
        default=60.0,
        description="Seconds of live audio to capture. Use 0 to run until the source stream ends.",
        examples=[60.0],
    )
    startup_timeout_seconds: Union[Selector(kind=[FLOAT_KIND, INTEGER_KIND]), float] = Field(
        default=20.0,
        examples=[20.0],
    )
    output_dir: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=str(DEFAULT_INGEST_OUTPUT_DIR / "direct-true-streaming"),
        examples=[str(DEFAULT_INGEST_OUTPUT_DIR / "direct-true-streaming")],
    )
    session_id: Union[Selector(kind=[STRING_KIND]), str] = Field(default="", examples=[""])
    password: Union[Selector(kind=[STRING_KIND]), str] = Field(default="", examples=[""])
    buffer_ms: Union[Selector(kind=[INTEGER_KIND]), int] = Field(default=300, examples=[300])
    language: Union[Selector(kind=[STRING_KIND]), str] = Field(default="en", examples=["en"])
    preset: Union[Selector(kind=[STRING_KIND]), str] = Field(default="meeting", examples=["meeting"])
    max_speakers: Union[Selector(kind=[INTEGER_KIND]), int] = Field(default=4, examples=[4])
    sample_rate: Union[Selector(kind=[INTEGER_KIND]), int] = Field(default=16000, examples=[16000])
    frame_duration_seconds: Union[Selector(kind=[FLOAT_KIND, INTEGER_KIND]), float] = Field(
        default=0.08,
        examples=[0.08],
    )
    transcription_backend: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=DEFAULT_TRUE_STREAMING_TRANSCRIPTION_BACKEND,
        examples=["local", "livepeer_remote", "livepeer_remote_http"],
    )
    livepeer_capability: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=DEFAULT_TRUE_STREAMING_CAPABILITY,
        examples=[DEFAULT_TRUE_STREAMING_CAPABILITY],
    )
    livepeer_offering: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=DEFAULT_TRUE_STREAMING_OFFERING,
        examples=[DEFAULT_TRUE_STREAMING_OFFERING],
    )
    livepeer_estimated_runway_units: Optional[int] = Field(default=None, examples=[105])
    livepeer_max_total_units: Optional[int] = Field(default=None, examples=[135])

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="session_id", kind=[STRING_KIND]),
            OutputDefinition(name="stream_id", kind=[STRING_KIND]),
            OutputDefinition(name="status", kind=[STRING_KIND]),
            OutputDefinition(name="source_mode", kind=[STRING_KIND]),
            OutputDefinition(name="source", kind=[STRING_KIND]),
            OutputDefinition(name="sent_audio_seconds", kind=[FLOAT_KIND]),
            OutputDefinition(name="sent_frame_count", kind=[INTEGER_KIND]),
            OutputDefinition(name="text", kind=[STRING_KIND]),
            OutputDefinition(name="speaker_count", kind=[INTEGER_KIND]),
            OutputDefinition(name="speakers", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="transcript_events", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="transcript_event_count", kind=[INTEGER_KIND]),
            OutputDefinition(name="events_jsonl_path", kind=[STRING_KIND]),
            OutputDefinition(name="result_json_path", kind=[STRING_KIND]),
            OutputDefinition(name="transcript_text_path", kind=[STRING_KIND]),
            OutputDefinition(name="publisher_log_path", kind=[STRING_KIND]),
            OutputDefinition(name="ffmpeg_log_path", kind=[STRING_KIND]),
            OutputDefinition(name="result", kind=[DICTIONARY_KIND]),
        ]

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=False, reason="requires_vdo_ninja_and_runner_websocket")

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class LivepeerVDONinjaDirectTrueStreamingSessionV1(WorkflowBlock):
    def __init__(
        self,
        runner_url: Optional[str] = None,
        vdo_signaling_server_url: str = "",
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_OPEN_CLEARINGHOUSE_URL,
        transcription_backend: str = DEFAULT_TRUE_STREAMING_TRANSCRIPTION_BACKEND,
        livepeer_capability: str = DEFAULT_TRUE_STREAMING_CAPABILITY,
        livepeer_offering: str = DEFAULT_TRUE_STREAMING_OFFERING,
        client_cls: Type[NemoTrueStreamingWebSocketClient] = NemoTrueStreamingWebSocketClient,
    ) -> None:
        self._runner_url = runner_url or init_nemo_diarized_runner_url()
        self._default_signaling_server = vdo_signaling_server_url or init_vdo_signaling_server_url()
        self._api_key = api_key or init_open_clearinghouse_api_key()
        self._base_url = base_url or init_open_clearinghouse_url()
        self._transcription_backend = (
            transcription_backend or init_true_streaming_transcription_backend()
        )
        self._livepeer_capability = livepeer_capability or init_true_streaming_capability()
        self._livepeer_offering = livepeer_offering or init_true_streaming_offering()
        self._client_cls = client_cls

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return [
            "runner_url",
            "vdo_signaling_server_url",
            "api_key",
            "base_url",
            "transcription_backend",
            "livepeer_capability",
            "livepeer_offering",
        ]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LivepeerVDONinjaDirectTrueStreamingSessionManifest

    def run(
        self,
        source: str = "auto",
        signaling_server: str = "",
        duration_seconds: float = 60.0,
        startup_timeout_seconds: float = 20.0,
        output_dir: str = str(DEFAULT_INGEST_OUTPUT_DIR / "direct-true-streaming"),
        session_id: str = "",
        password: str = "",
        buffer_ms: int = 300,
        language: str = "en",
        preset: str = "meeting",
        max_speakers: int = 4,
        sample_rate: int = 16000,
        frame_duration_seconds: float = 0.08,
        transcription_backend: str = "",
        livepeer_capability: str = "",
        livepeer_offering: str = "",
        livepeer_estimated_runway_units: Optional[int] = None,
        livepeer_max_total_units: Optional[int] = None,
    ) -> BlockResult:
        client_cls, client_base_url, client_init_kwargs = _resolve_true_streaming_client(
            backend=transcription_backend or self._transcription_backend,
            local_client_cls=self._client_cls,
            runner_url=self._runner_url,
            api_key=self._api_key,
            base_url=self._base_url,
            livepeer_capability=livepeer_capability or self._livepeer_capability,
            livepeer_offering=livepeer_offering or self._livepeer_offering,
            duration_seconds=float(duration_seconds),
            livepeer_estimated_runway_units=livepeer_estimated_runway_units,
            livepeer_max_total_units=livepeer_max_total_units,
        )
        if client_cls in {
            LivepeerRemoteFallbackTransportClient,
            LivepeerRemoteHttpChunkingClient,
        } and float(duration_seconds) > 0:
            client_init_kwargs = {
                **client_init_kwargs,
                "chunk_size_seconds": _livepeer_remote_http_chunk_size_seconds(
                    window_seconds=float(duration_seconds),
                    frame_duration_seconds=float(frame_duration_seconds),
                ),
            }
        runner = build_vdo_direct_true_streaming_runner(
            source=source,
            runner_url=client_base_url,
            output_dir=output_dir,
            duration_seconds=float(duration_seconds),
            startup_timeout_seconds=float(startup_timeout_seconds),
            session_id=session_id,
            password=password,
            signaling_server=str(signaling_server or self._default_signaling_server or ""),
            buffer_ms=int(buffer_ms),
            language=language,
            preset=preset,
            max_speakers=int(max_speakers),
            sample_rate=int(sample_rate),
            frame_duration_seconds=float(frame_duration_seconds),
            client_cls=client_cls,
            client_init_kwargs=client_init_kwargs,
        )
        result = runner.run()
        return {
            "session_id": result["session_id"],
            "stream_id": result["stream_id"],
            "status": result["status"],
            "source_mode": result["source_mode"],
            "source": result["source"],
            "sent_audio_seconds": result["sent_audio_seconds"],
            "sent_frame_count": result["sent_frame_count"],
            "text": result["text"],
            "speaker_count": result["speaker_count"],
            "speakers": result["speakers"],
            "transcript_events": result["transcript_events"],
            "transcript_event_count": result["transcript_event_count"],
            "events_jsonl_path": result["events_jsonl_path"],
            "result_json_path": result["result_json_path"],
            "transcript_text_path": result["transcript_text_path"],
            "publisher_log_path": result["publisher_log_path"],
            "ffmpeg_log_path": result["ffmpeg_log_path"],
            "result": result,
        }


class LivepeerVDONinjaRollingAudioCaptureManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Livepeer VDO.Ninja Rolling Audio Capture",
            "version": "v1",
            "short_description": "Capture rolling audio segments from VDO.Ninja via Raspberry.Ninja.",
            "long_description": INGEST_LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "source",
            "search_keywords": [
                "livepeer",
                "vdo.ninja",
                "raspberry.ninja",
                "gstreamer",
                "audio",
                "ingest",
            ],
            "ui_manifest": {
                "section": "sources",
                "icon": "fal fa-satellite-dish",
                "blockPriority": 4,
            },
        }
    )

    type: Literal[
        "roboflow_livepeer_blocks/livepeer_vdo_ninja_rolling_audio_capture@v1",
        "LivepeerVDONinjaRollingAudioCapture",
    ]
    source: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Raw VDO.Ninja stream ID or viewer URL containing ?view=...",
        examples=["stream_av53zc79i", "https://vdo.ninja/?view=stream_av53zc79i"],
    )
    segment_count: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        default=1,
        description="Number of rolling segments to capture in this workflow invocation.",
        examples=[1, 3],
    )
    segment_duration_seconds: Union[Selector(kind=[FLOAT_KIND, INTEGER_KIND]), float] = Field(
        default=30.0,
        description="Duration of each captured segment.",
        examples=[30.0],
    )
    startup_seconds: Union[Selector(kind=[FLOAT_KIND, INTEGER_KIND]), float] = Field(
        default=8.0,
        description="Warmup seconds before the first segment is finalized.",
        examples=[8.0],
    )
    output_dir: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=str(DEFAULT_INGEST_OUTPUT_DIR),
        description="Directory where recordings, logs, and extracted WAV files are stored.",
        examples=[str(DEFAULT_INGEST_OUTPUT_DIR)],
    )
    password: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default="",
        description="Optional VDO.Ninja stream password. Empty disables the password flag.",
        examples=[""],
    )
    buffer_ms: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        default=300,
        description="Raspberry.Ninja/GStreamer jitter buffer in milliseconds.",
        examples=[300],
    )
    audio_only: Union[Selector(kind=[BOOLEAN_KIND]), bool] = Field(
        default=True,
        description="Pass --novideo to Raspberry.Ninja and only capture audio.",
        examples=[True],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="stream_id", kind=[STRING_KIND]),
            OutputDefinition(name="segments", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="audio_paths", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="first_audio_path", kind=[STRING_KIND]),
            OutputDefinition(name="latest_audio_path", kind=[STRING_KIND]),
            OutputDefinition(name="segment_count", kind=[INTEGER_KIND]),
            OutputDefinition(name="output_dir", kind=[STRING_KIND]),
            OutputDefinition(name="result", kind=[DICTIONARY_KIND]),
        ]

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=False, reason="requires_vdo_ninja_network")

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class LivepeerVDONinjaRollingAudioCaptureV1(WorkflowBlock):
    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return []

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LivepeerVDONinjaRollingAudioCaptureManifest

    def run(
        self,
        source: str,
        segment_count: int = 1,
        segment_duration_seconds: float = 30.0,
        startup_seconds: float = 8.0,
        output_dir: str = str(DEFAULT_INGEST_OUTPUT_DIR),
        password: str = "",
        buffer_ms: int = 300,
        audio_only: bool = True,
    ) -> BlockResult:
        result = capture_rolling_audio_segments(
            source=source,
            output_dir=output_dir,
            segment_count=segment_count,
            segment_duration_seconds=float(segment_duration_seconds),
            startup_seconds=float(startup_seconds),
            password=password,
            buffer_ms=buffer_ms,
            audio_only=audio_only,
        )
        return {
            "stream_id": result["stream_id"],
            "segments": result["segments"],
            "audio_paths": result["audio_paths"],
            "first_audio_path": result["first_audio_path"],
            "latest_audio_path": result["latest_audio_path"],
            "segment_count": result["segment_count"],
            "output_dir": result["output_dir"],
            "result": result,
        }


def _prepare_live_artifact_dir(*, output_dir: str, source: str) -> Path:
    from .ingest import parse_vdo_stream_id

    stream_id = parse_vdo_stream_id(source)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    artifact_dir = Path(output_dir) / f"vdo-{stream_id.replace('stream_', '')}-live-{timestamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _prepare_visual_artifact_dir(*, output_dir: str, source: str) -> Path:
    stream_id = parse_vdo_stream_id(source)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    artifact_dir = Path(output_dir) / f"vdo-{stream_id.replace('stream_', '')}-visual-{timestamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _append_live_event(
    events: List[Dict[str, Any]],
    events_path: Path,
    event: Dict[str, Any],
) -> None:
    event_with_time = {"roboflow_recorded_at_epoch": time.time(), **event}
    events.append(event_with_time)
    with events_path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(event_with_time, sort_keys=True) + "\n")


def _append_live_transcript_events(
    events: List[Dict[str, Any]],
    transcript_path: Path,
    incoming_events: List[Dict[str, Any]],
) -> None:
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    with transcript_path.open("a", encoding="utf-8") as output:
        for event in incoming_events:
            event_with_time = {
                "roboflow_recorded_at_epoch": time.time(),
                "roboflow_transcript_event_index": len(events),
                **event,
            }
            events.append(event_with_time)
            output.write(json.dumps(event_with_time, sort_keys=True) + "\n")


def _transcript_events_from_live_response(
    response: Dict[str, Any],
    *,
    source_segment: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    runner_events = response.get("transcript_events")
    if isinstance(runner_events, list) and runner_events:
        events = [dict(event) for event in runner_events if isinstance(event, dict)]
        if source_segment:
            for event in events:
                event.setdefault("source_segment", source_segment)
        return events

    event_type = str(response.get("event_type") or "")
    session_id = str(response.get("session_id") or "")
    if event_type == "session.started":
        return [
            _transcript_lifecycle_event(
                session_id=session_id,
                event_type="transcript.session.started",
                status=str(response.get("status") or "active"),
                is_provisional=True,
                authority="online_diarization",
            )
        ]
    if event_type == "audio.ingested":
        return _derived_ingest_transcript_events(response, source_segment=source_segment)
    if event_type == "session.finished":
        events = _derived_final_transcript_events(response)
        events.append(
            _transcript_lifecycle_event(
                session_id=session_id,
                event_type="transcript.session.finished",
                status=str(response.get("status") or "closed"),
                is_provisional=False,
                authority="session_lifecycle",
            )
        )
        return events
    return []


def _derived_ingest_transcript_events(
    response: Dict[str, Any],
    *,
    source_segment: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    session_id = str(response.get("session_id") or "")
    chunk = response.get("chunk") or {}
    chunk_sequence_index = chunk.get("sequence_index")
    chunk_text = str(chunk.get("text") or "")
    chunk_text_status = str(
        chunk.get("text_status") or ("available" if chunk_text.strip() else "not_available_online")
    )
    events = [
        {
            "schema_version": "livepeer.diarized_transcript_event.v1",
            "event_id": f"{session_id}:roboflow:chunk:{chunk_sequence_index}",
            "event_type": "transcript.chunk_ingested",
            "session_id": session_id,
            "status": str(response.get("status") or "active"),
            "chunk_sequence_index": chunk_sequence_index,
            "start": _event_seconds(chunk.get("start", 0.0)),
            "end": _event_seconds(chunk.get("end", response.get("duration_seconds", 0.0))),
            "duration_seconds": _event_seconds(chunk.get("duration_seconds", 0.0)),
            "speaker": None,
            "text": chunk_text,
            "is_provisional": True,
            "is_final": False,
            "authority": str(chunk.get("authority") or "online_diarization_provisional_asr"),
            "text_status": chunk_text_status,
        }
    ]
    if chunk.get("asr_model"):
        events[0]["asr_model"] = chunk.get("asr_model")
    if not chunk_text.strip():
        events[0]["text_unavailable_reason"] = str(
            chunk.get("text_unavailable_reason")
            or (
                "Provisional ASR text was not included in this live ingest response; "
                "authoritative transcript text is produced by the final offline ASR pass."
            )
        )
    if source_segment:
        events[0]["source_segment"] = source_segment

    new_segments = response.get("new_segments")
    segments = new_segments if isinstance(new_segments, list) else response.get("segments") or []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "")
        text_status = "available" if text.strip() else "not_available_online"
        event = {
            "schema_version": "livepeer.diarized_transcript_event.v1",
            "event_id": (
                f"{session_id}:roboflow:provisional:{chunk_sequence_index}:"
                f"{index}:{segment.get('speaker')}"
            ),
            "event_type": "transcript.segment",
            "session_id": session_id,
            "status": str(response.get("status") or "active"),
            "chunk_sequence_index": chunk_sequence_index,
            "segment_id": _segment_event_id(session_id, "provisional", segment),
            "start": _event_seconds(segment.get("start", 0.0)),
            "end": _event_seconds(segment.get("end", 0.0)),
            "speaker": segment.get("speaker"),
            "text": text,
            "is_provisional": True,
            "is_final": False,
            "authority": str(segment.get("authority") or "online_diarization_provisional_asr"),
            "text_status": text_status,
        }
        if segment.get("asr_model") or chunk.get("asr_model"):
            event["asr_model"] = segment.get("asr_model") or chunk.get("asr_model")
        if not text.strip():
            event["text_unavailable_reason"] = str(
                segment.get("text_unavailable_reason")
                or chunk.get("text_unavailable_reason")
                or (
                    "Provisional ASR text was not included in this live ingest response; "
                    "authoritative transcript text is produced by the final offline ASR pass."
                )
            )
        if source_segment:
            event["source_segment"] = source_segment
        events.append(event)
    return events


def _derived_final_transcript_events(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    session_id = str(response.get("session_id") or "")
    final_transcription = response.get("final_transcription") or {}
    if not isinstance(final_transcription, dict):
        return []
    models = final_transcription.get("models") or {}
    asr_model = str(models.get("asr") or "")
    segments = final_transcription.get("segments") or []
    events: List[Dict[str, Any]] = []
    if isinstance(segments, list):
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                continue
            text = str(segment.get("text") or "").strip()
            events.append(
                {
                    "schema_version": "livepeer.diarized_transcript_event.v1",
                    "event_id": f"{session_id}:roboflow:final:{index}",
                    "event_type": "transcript.segment",
                    "session_id": session_id,
                    "status": str(response.get("status") or "closed"),
                    "segment_id": _segment_event_id(session_id, f"final:{index:06d}", segment),
                    "start": _event_seconds(segment.get("start", 0.0)),
                    "end": _event_seconds(segment.get("end", 0.0)),
                    "speaker": segment.get("speaker"),
                    "text": text,
                    "is_provisional": False,
                    "is_final": True,
                    "authority": "final_offline_diarized_transcription",
                    "text_status": "available" if text else "empty",
                    "asr_model": asr_model,
                }
            )
    if not events:
        text = str(final_transcription.get("text") or "").strip()
        if text:
            events.append(
                {
                    "schema_version": "livepeer.diarized_transcript_event.v1",
                    "event_id": f"{session_id}:roboflow:final:text",
                    "event_type": "transcript.segment",
                    "session_id": session_id,
                    "status": str(response.get("status") or "closed"),
                    "segment_id": f"{session_id}:final:text",
                    "start": 0.0,
                    "end": _event_seconds(response.get("duration_seconds", 0.0)),
                    "speaker": None,
                    "text": text,
                    "is_provisional": False,
                    "is_final": True,
                    "authority": "final_offline_diarized_transcription",
                    "text_status": "available",
                    "asr_model": asr_model,
                }
            )
    return events


def _transcript_lifecycle_event(
    *,
    session_id: str,
    event_type: str,
    status: str,
    is_provisional: bool,
    authority: str,
) -> Dict[str, Any]:
    return {
        "schema_version": "livepeer.diarized_transcript_event.v1",
        "event_id": f"{session_id}:roboflow:{event_type}",
        "event_type": event_type,
        "session_id": session_id,
        "status": status,
        "speaker": None,
        "text": "",
        "is_provisional": is_provisional,
        "is_final": not is_provisional,
        "authority": authority,
        "text_status": "not_applicable",
    }


def _segment_event_id(session_id: str, phase: str, segment: Dict[str, Any]) -> str:
    speaker = str(segment.get("speaker") or "unknown")
    start = _event_seconds(segment.get("start", 0.0))
    end = _event_seconds(segment.get("end", 0.0))
    return f"{session_id}:{phase}:{speaker}:{start:.3f}-{end:.3f}"


def _event_seconds(value: Any) -> float:
    return round(float(value or 0.0), 3)


def _write_transcript_text(
    *,
    artifact_dir: Path,
    final_transcription: Dict[str, Any],
) -> str:
    text = str(final_transcription.get("text") or "").strip()
    if not text:
        return ""
    transcript_path = artifact_dir / "final-transcript.txt"
    transcript_path.write_text(text + "\n", encoding="utf-8")
    return str(transcript_path)


def _live_session_result_payload(
    *,
    created: Dict[str, Any],
    finished: Dict[str, Any],
    events: List[Dict[str, Any]],
    transcript_events: List[Dict[str, Any]],
    captured_segments: List[Dict[str, Any]],
    artifact_dir: Path,
    events_path: Path,
    provisional_transcript_path: Path,
    result_path: Path,
    transcript_text_path: str,
) -> Dict[str, Any]:
    final_transcription = finished.get("final_transcription") or {}
    live_segments = finished.get("segments") or []
    audio_paths = [
        str(segment.get("audio_path"))
        for segment in captured_segments
        if segment.get("audio_path")
    ]
    return {
        "session_id": str(finished.get("session_id") or created.get("session_id") or ""),
        "stream_id": str(captured_segments[0].get("stream_id") if captured_segments else ""),
        "status": str(finished.get("status") or ""),
        "captured_segments": captured_segments,
        "audio_paths": audio_paths,
        "live_segments": live_segments,
        "text": str(final_transcription.get("text") or ""),
        "speaker_count": int(final_transcription.get("speaker_count") or 0),
        "speakers": final_transcription.get("speakers") or [],
        "words": final_transcription.get("words") or [],
        "final_audio_path": str(finished.get("final_audio_path") or ""),
        "final_transcription": final_transcription,
        "events": events,
        "event_count": len(events),
        "transcript_events": transcript_events,
        "transcript_event_count": len(transcript_events),
        "output_dir": str(artifact_dir),
        "events_jsonl_path": str(events_path),
        "provisional_transcript_jsonl_path": str(provisional_transcript_path),
        "result_json_path": str(result_path),
        "transcript_text_path": transcript_text_path,
        "created": created,
        "finished": finished,
    }
