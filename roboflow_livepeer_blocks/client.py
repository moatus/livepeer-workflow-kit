"""Direct Livepeer Open Clearinghouse handoff client for audio transcription."""

from __future__ import annotations

import math
import json
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from .audio import AudioChunk, materialize_audio_chunks
from .config import DEFAULT_OPEN_CLEARINGHOUSE_URL

try:
    import httpx
except ImportError:  # pragma: no cover - exercised only in minimal environments
    httpx = None  # type: ignore[assignment]


DEFAULT_CAPABILITY = "openai:audio-transcriptions"
DEFAULT_OFFERING = "whisper-large-v3"
DEFAULT_RESPONSE_FORMAT = "json"
SDK_IDENTITY = "roboflow-livepeer-blocks-poc/0.1.0"
WORK_UNITS_HEADERS = ("Livepeer-Work-Units", "X-Livepeer-Work-Units")


@dataclass(frozen=True)
class ChunkTranscription:
    chunk: AudioChunk
    text: str
    actual_units: int
    job_id: Optional[str]
    work_id: Optional[str]
    raw_responses: Dict[str, Any]


def _json_or_text(response: Any) -> Any:
    try:
        return response.json()
    except Exception:
        return response.text


def _response_snapshot(response: Any) -> Dict[str, Any]:
    return {
        "status_code": response.status_code,
        "headers": dict(response.headers),
        "body": _json_or_text(response),
    }


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise RuntimeError(f"Invalid work units header value: {value!r}")


def _extract_text(payload: Any) -> str:
    if isinstance(payload, dict):
        text = payload.get("text")
        if isinstance(text, str):
            return text
        if isinstance(payload.get("transcript"), str):
            return payload["transcript"]
    if isinstance(payload, str):
        return payload
    return ""


def _get_header(headers: Any, names: tuple[str, ...]) -> Optional[str]:
    for name in names:
        value = headers.get(name)
        if value is not None:
            return value
        value = headers.get(name.lower())
        if value is not None:
            return value
    return None


def _extract_actual_units(response: Any) -> int:
    actual_units = _get_header(response.headers, WORK_UNITS_HEADERS)
    if not actual_units:
        trailing_headers = getattr(response, "trailing_headers", None)
        if trailing_headers is not None:
            actual_units = _get_header(trailing_headers, WORK_UNITS_HEADERS)
    if actual_units is None:
        raise RuntimeError("Broker response missing Livepeer-Work-Units header")
    return _coerce_int(actual_units)


def _append_path(base_url: str, path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return path_or_url
    return urljoin(f"{base_url.rstrip('/')}/", path_or_url.lstrip("/"))


class LivepeerOpenClearinghouseClient:
    def __init__(
        self,
        api_key: Optional[str],
        base_url: str = DEFAULT_OPEN_CLEARINGHOUSE_URL,
        timeout_seconds: float = 120.0,
        http_client: Any = None,
    ) -> None:
        if not api_key:
            raise ValueError("LIVEPEER_OPEN_CLEARINGHOUSE_API_KEY is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        if http_client is not None:
            self._client = http_client
            self._owns_client = False
        else:
            if httpx is None:
                raise RuntimeError("httpx is required to call Livepeer Open Clearinghouse")
            self._client = httpx.Client(timeout=timeout_seconds)
            self._owns_client = True

    def close(self) -> None:
        if self._owns_client and hasattr(self._client, "close"):
            self._client.close()

    def __enter__(self) -> "LivepeerOpenClearinghouseClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def transcribe_chunk(
        self,
        chunk: AudioChunk,
        capability: str = DEFAULT_CAPABILITY,
        offering: str = DEFAULT_OFFERING,
        response_format: str = DEFAULT_RESPONSE_FORMAT,
        max_total_units: Optional[int] = None,
    ) -> ChunkTranscription:
        estimated_units = max(1, math.ceil(chunk.duration_seconds))
        open_payload = {
            "capability": capability,
            "offering": offering,
            "estimated_units": estimated_units,
            "max_total_units": max_total_units or max(estimated_units, estimated_units + 10),
        }
        open_response = self._client.post(
            f"{self.base_url}/v1/jobs",
            headers={
                "X-API-Key": self.api_key,
                "Livepeer-Open-Clearinghouse-SDK": SDK_IDENTITY,
            },
            json=open_payload,
        )
        open_response.raise_for_status()
        open_body = open_response.json()

        broker_url = open_body["broker_url"].rstrip("/")
        payment_envelope = open_body["payment_envelope"]
        payment_header = (
            json.dumps(payment_envelope)
            if isinstance(payment_envelope, (dict, list))
            else str(payment_envelope)
        )
        mode = str(open_body.get("mode", ""))
        request_id = str(uuid.uuid4())
        broker_headers = {
            "Livepeer-Capability": capability,
            "Livepeer-Offering": offering,
            "Livepeer-Payment": payment_header,
            "Livepeer-Mode": mode,
            "Livepeer-Spec-Version": "0.1",
            "Livepeer-Request-Id": request_id,
        }
        with Path(chunk.path).open("rb") as audio_file:
            broker_response = self._client.post(
                f"{broker_url}/v1/cap",
                headers=broker_headers,
                data={"model": offering, "response_format": response_format},
                files={"file": (Path(chunk.path).name, audio_file, "application/octet-stream")},
            )
        broker_body = _json_or_text(broker_response)
        settle_actual_units = 0
        settle_error: Optional[Exception] = None
        try:
            settle_actual_units = _extract_actual_units(broker_response)
        except RuntimeError as error:
            settle_error = error
            if broker_response.status_code < 400:
                # Successful transcriptions must report billable work units.
                settle_actual_units = 0

        settle_endpoint = open_body["settle_endpoint"]
        settle_response = self._client.post(
            _append_path(self.base_url, settle_endpoint),
            headers={
                "X-API-Key": self.api_key,
                "Livepeer-Open-Clearinghouse-SDK": SDK_IDENTITY,
            },
            json={"actual_units": settle_actual_units},
        )
        settle_response.raise_for_status()
        if broker_response.status_code >= 400:
            raise RuntimeError(
                f"Broker request failed with status {broker_response.status_code}: {broker_body}"
            )
        if settle_error is not None:
            raise settle_error

        return ChunkTranscription(
            chunk=chunk,
            text=_extract_text(broker_body),
            actual_units=settle_actual_units,
            job_id=open_body.get("job_id") or open_body.get("id"),
            work_id=open_body.get("work_id"),
            raw_responses={
                "open_job": _response_snapshot(open_response),
                "broker": _response_snapshot(broker_response),
                "settle": _response_snapshot(settle_response),
                "request_id": request_id,
            },
        )

    def transcribe_audio_file(
        self,
        audio_path: str,
        chunk_size_seconds: float = 10.0,
        capability: str = DEFAULT_CAPABILITY,
        offering: str = DEFAULT_OFFERING,
        response_format: str = DEFAULT_RESPONSE_FORMAT,
        max_total_units_per_chunk: Optional[int] = None,
    ) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="livepeer-audio-chunks-") as chunk_dir:
            chunks = materialize_audio_chunks(
                audio_path=audio_path,
                output_dir=chunk_dir,
                chunk_size_seconds=chunk_size_seconds,
            )
            chunk_results = [
                self.transcribe_chunk(
                    chunk=chunk,
                    capability=capability,
                    offering=offering,
                    response_format=response_format,
                    max_total_units=max_total_units_per_chunk,
                )
                for chunk in chunks
            ]
        return aggregate_transcriptions(
            results=chunk_results,
            source_audio_path=str(Path(audio_path)),
        )


def aggregate_transcriptions(
    results: List[ChunkTranscription],
    source_audio_path: Optional[str] = None,
) -> Dict[str, Any]:
    ordered = sorted(results, key=lambda result: result.chunk.index)
    return {
        "text": " ".join(result.text.strip() for result in ordered if result.text.strip()),
        "chunks": [
            {
                "index": result.chunk.index,
                "start_seconds": result.chunk.start_seconds,
                "end_seconds": result.chunk.end_seconds,
                "duration_seconds": result.chunk.duration_seconds,
                "temporary": result.chunk.temporary,
                "audio_path": source_audio_path or str(result.chunk.path),
                "chunk_file_path": (
                    None if result.chunk.temporary else str(result.chunk.path)
                ),
                "text": result.text,
                "actual_units": result.actual_units,
                "job_id": result.job_id,
                "work_id": result.work_id,
            }
            for result in ordered
        ],
        "actual_units": sum(result.actual_units for result in ordered),
        "job_ids": [result.job_id for result in ordered if result.job_id],
        "work_ids": [result.work_id for result in ordered if result.work_id],
        "raw_responses": [result.raw_responses for result in ordered],
    }
