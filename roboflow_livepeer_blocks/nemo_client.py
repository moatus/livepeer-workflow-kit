"""HTTP client for the standalone audio diarized transcription runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import init_nemo_diarized_runner_url

try:
    import httpx
except ImportError:  # pragma: no cover - exercised only in minimal environments
    httpx = None  # type: ignore[assignment]


WORK_UNITS_HEADERS = ("X-Livepeer-Work-Units", "Livepeer-Work-Units")


class NemoDiarizedTranscriptionClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_seconds: float = 1800.0,
        http_client: Any = None,
    ) -> None:
        self.base_url = (base_url or init_nemo_diarized_runner_url()).rstrip("/")
        if http_client is not None:
            self._client = http_client
            self._owns_client = False
        else:
            if httpx is None:
                raise RuntimeError("httpx is required to call the audio diarized runner")
            self._client = httpx.Client(timeout=timeout_seconds)
            self._owns_client = True

    def close(self) -> None:
        if self._owns_client and hasattr(self._client, "close"):
            self._client.close()

    def __enter__(self) -> "NemoDiarizedTranscriptionClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def diarized_transcribe_audio_file(
        self,
        audio_path: str,
        model: str = "nemo-diarized-transcription-meeting-v0",
        language: str = "en",
        preset: str = "meeting",
        num_speakers: Optional[int] = None,
        max_speakers: int = 8,
        response_format: str = "json",
        include_words: bool = True,
        include_artifacts: bool = True,
    ) -> Dict[str, Any]:
        return self.openai_transcribe_audio_file(
            audio_path=audio_path,
            model=model,
            language=language,
            preset=preset,
            num_speakers=num_speakers,
            max_speakers=max_speakers,
            response_format=response_format,
            include_words=include_words,
            include_artifacts=include_artifacts,
        )

    def openai_transcribe_audio_file(
        self,
        audio_path: str,
        model: str = "nemo-diarized-transcription-meeting-v0",
        language: str = "en",
        preset: str = "meeting",
        num_speakers: Optional[int] = None,
        max_speakers: int = 8,
        response_format: str = "json",
        include_words: bool = True,
        include_artifacts: bool = True,
    ) -> Dict[str, Any]:
        path = Path(audio_path)
        runner_response_format = (
            "verbose_json" if response_format in {"json", "verbose_json"} else response_format
        )
        data: Dict[str, Any] = {
            "model": model,
            "language": language,
            "preset": preset,
            "max_speakers": str(max_speakers),
            "response_format": runner_response_format,
            "include_words": str(include_words).lower(),
            "include_artifacts": str(include_artifacts).lower(),
            "diarization": "true",
        }
        if include_words:
            data["timestamp_granularities[]"] = ["segment", "word"]
        else:
            data["timestamp_granularities[]"] = ["segment"]
        if num_speakers is not None:
            data["num_speakers"] = str(num_speakers)

        with path.open("rb") as audio_file:
            response = self._client.post(
                f"{self.base_url}/v1/audio/transcriptions",
                data=data,
                files={"file": (path.name, audio_file, "application/octet-stream")},
            )
        body = _json_or_text(response)
        if response.status_code >= 400:
            raise RuntimeError(
                f"OpenAI-compatible audio transcription runner failed with status {response.status_code}: {body}"
            )

        actual_units = _extract_actual_units(response)
        if runner_response_format == "verbose_json":
            normalized = _normalize_openai_verbose_response(body)
        elif isinstance(body, dict):
            normalized = {"text": str(body.get("text") or ""), "openai_response": body}
        else:
            normalized = {"text": str(body), "openai_response": body}
        normalized["actual_units"] = actual_units
        normalized["api_endpoint"] = "/v1/audio/transcriptions"
        normalized["raw_response"] = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
        }
        return normalized

    def create_live_session(
        self,
        *,
        session_id: Optional[str] = None,
        language: str = "en",
        preset: str = "meeting",
        num_speakers: Optional[int] = None,
        max_speakers: int = 8,
        vad_strategy: str = "energy",
        rolling_window_seconds: float = 60.0,
        energy_threshold: float = 0.012,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "language": language,
            "preset": preset,
            "max_speakers": max_speakers,
            "vad_strategy": vad_strategy,
            "rolling_window_seconds": rolling_window_seconds,
            "energy_threshold": energy_threshold,
        }
        if session_id:
            payload["session_id"] = session_id
        if num_speakers is not None:
            payload["num_speakers"] = num_speakers
        response = self._client.post(
            f"{self.base_url}/v1/audio/diarized-transcriptions/live/sessions",
            json=payload,
        )
        return _checked_json(response, "create live diarization session")

    def get_live_session(self, session_id: str) -> Dict[str, Any]:
        response = self._client.get(
            f"{self.base_url}/v1/audio/diarized-transcriptions/live/sessions/{session_id}"
        )
        return _checked_json(response, "get live diarization session")

    def ingest_live_audio_file(
        self,
        *,
        session_id: str,
        audio_path: str,
        sequence_index: Optional[int] = None,
        vad_segments: Optional[List[Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        path = Path(audio_path)
        data: Dict[str, Any] = {}
        if sequence_index is not None:
            data["sequence_index"] = str(sequence_index)
        if vad_segments is not None:
            import json

            data["vad_segments_json"] = json.dumps(vad_segments)
        with path.open("rb") as audio_file:
            response = self._client.post(
                f"{self.base_url}/v1/audio/diarized-transcriptions/live/sessions/{session_id}/audio",
                data=data,
                files={"file": (path.name, audio_file, "application/octet-stream")},
            )
        return _checked_json(response, "ingest live diarization audio")

    def finish_live_session(
        self,
        *,
        session_id: str,
        run_final_transcription: bool = False,
        include_words: bool = True,
        include_artifacts: bool = True,
    ) -> Dict[str, Any]:
        response = self._client.post(
            f"{self.base_url}/v1/audio/diarized-transcriptions/live/sessions/{session_id}/finish",
            json={
                "run_final_transcription": run_final_transcription,
                "include_words": include_words,
                "include_artifacts": include_artifacts,
            },
        )
        return _checked_json(response, "finish live diarization session")


def _json_or_text(response: Any) -> Any:
    try:
        return response.json()
    except Exception:
        return response.text


def _extract_actual_units(response: Any) -> int:
    for name in WORK_UNITS_HEADERS:
        value = response.headers.get(name) or response.headers.get(name.lower())
        if value is not None:
            try:
                return int(value)
            except ValueError as error:
                raise RuntimeError(f"Invalid work units header value: {value!r}") from error
    raise RuntimeError("Audio diarized runner response missing X-Livepeer-Work-Units header")


def _normalize_openai_verbose_response(body: Any) -> Dict[str, Any]:
    if not isinstance(body, dict):
        raise RuntimeError(f"Expected verbose_json response from audio diarized runner, got: {body!r}")
    legacy_livepeer = body.get("x_livepeer") if isinstance(body.get("x_livepeer"), dict) else {}
    generic_extension = {
        "id": body.get("transcription_id"),
        "capability": body.get("capability"),
        "mode": body.get("mode"),
        "models": body.get("models"),
        "usage": body.get("usage"),
        "speaker_labeled_text": body.get("speaker_labeled_text"),
        "diarization": body.get("diarization"),
        "artifacts": body.get("artifacts"),
    }
    livepeer = legacy_livepeer or generic_extension
    diarization = (
        body.get("x_livepeer_diarization")
        if isinstance(body.get("x_livepeer_diarization"), dict)
        else livepeer.get("diarization", {})
    )
    if not isinstance(diarization, dict):
        diarization = {}
    usage = livepeer.get("usage", {}) if isinstance(livepeer.get("usage"), dict) else {}
    speaker_labeled_text = str(livepeer.get("speaker_labeled_text") or "").strip()
    text = speaker_labeled_text or str(body.get("text") or "").strip()
    segments = diarization.get("segments") if isinstance(diarization.get("segments"), list) else []
    words = diarization.get("words") if isinstance(diarization.get("words"), list) else []
    speakers = diarization.get("speakers") if isinstance(diarization.get("speakers"), list) else []
    artifacts = livepeer.get("artifacts") if isinstance(livepeer.get("artifacts"), dict) else {}
    return {
        "id": livepeer.get("id"),
        "status": "success",
        "capability": livepeer.get("capability"),
        "mode": livepeer.get("mode"),
        "models": livepeer.get("models") if isinstance(livepeer.get("models"), dict) else {},
        "duration_seconds": float(body.get("duration") or 0.0),
        "text": text,
        "openai_text": str(body.get("text") or "").strip(),
        "speaker_count": int(diarization.get("speaker_count") or len(speakers)),
        "speakers": speakers,
        "segments": segments,
        "words": words,
        "artifacts": artifacts,
        "usage": usage,
        "openai_response": body,
    }


def _checked_json(response: Any, action: str) -> Dict[str, Any]:
    body = _json_or_text(response)
    if response.status_code >= 400:
        raise RuntimeError(
            f"Audio diarized runner failed to {action} with status {response.status_code}: {body}"
        )
    if not isinstance(body, dict):
        raise RuntimeError(f"Expected JSON response while trying to {action}, got: {body!r}")
    return body
