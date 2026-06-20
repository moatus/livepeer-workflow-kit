"""Thin HTTP client for Florence-compatible vision runners."""

from __future__ import annotations

import base64
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .config import (
    DEFAULT_OPEN_CLEARINGHOUSE_URL,
    DEFAULT_VISION_CAPABILITY,
    DEFAULT_VISION_OFFERING,
    init_florence2_runner_url,
)

try:
    import httpx
except ImportError:  # pragma: no cover - exercised only in minimal environments
    httpx = None  # type: ignore[assignment]


PRIMARY_VISION_ANALYZE_ENDPOINT = "/v1/vision/analyze"
LIVEPEER_CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"
COMPAT_LMM_INFER_ENDPOINT = "/infer/lmm"
DEFAULT_FLORENCE2_TASKS = ("<CAPTION>", "<DETAILED_CAPTION>", "<OCR>")
COMPATIBILITY_STATUS_CODES = {404, 405}
SDK_IDENTITY = "roboflow-livepeer-blocks-poc/0.1.0"
WORK_UNITS_HEADERS = ("Livepeer-Work-Units", "X-Livepeer-Work-Units")


class _BrokeredResponse:
    def __init__(self, *, status_code: int, body: Any, headers: Optional[Dict[str, Any]] = None) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self) -> Any:
        if isinstance(self._body, str):
            raise ValueError("response body is not JSON")
        return self._body


class Florence2VisionRunnerClient:
    """Client for the runner-facing Florence vision endpoint.

    The primary contract is `POST /v1/vision/analyze`. `/infer/lmm` is tried
    only when the primary endpoint is unavailable on an older runner.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_seconds: float = 600.0,
        http_client: Any = None,
        livepeer_api_key: Optional[str] = None,
        livepeer_base_url: str = DEFAULT_OPEN_CLEARINGHOUSE_URL,
        capability: str = DEFAULT_VISION_CAPABILITY,
        offering: str = DEFAULT_VISION_OFFERING,
        use_livepeer_gateway: bool = False,
    ) -> None:
        self.base_url = (base_url or init_florence2_runner_url()).rstrip("/")
        self.livepeer_api_key = livepeer_api_key
        self.livepeer_base_url = (livepeer_base_url or DEFAULT_OPEN_CLEARINGHOUSE_URL).rstrip("/")
        self.capability = capability or DEFAULT_VISION_CAPABILITY
        self.offering = offering or DEFAULT_VISION_OFFERING
        self.use_livepeer_gateway = use_livepeer_gateway or bool(livepeer_api_key)
        if http_client is not None:
            self._client = http_client
            self._owns_client = False
        else:
            if httpx is None:
                raise RuntimeError("httpx is required to call the Florence vision runner")
            self._client = httpx.Client(timeout=timeout_seconds)
            self._owns_client = True
        self.last_endpoint = ""

    def close(self) -> None:
        if self._owns_client and hasattr(self._client, "close"):
            self._client.close()

    def __enter__(self) -> "Florence2VisionRunnerClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def analyze_image(
        self,
        *,
        image_path: str,
        model_id: str,
        meeting_context_prompt: str = "",
        tasks: Iterable[str] = DEFAULT_FLORENCE2_TASKS,
    ) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {
            "caption": "",
            "detailed_caption": "",
            "ocr_text": "",
            "meeting_context": {
                "supported": False,
                "prompt": meeting_context_prompt,
                "text": "",
                "error": "",
            },
        }
        for task_prompt in tasks:
            response = self._post_primary(
                image_path=image_path,
                model_id=model_id,
                task_prompt=task_prompt,
            )
            if not self.use_livepeer_gateway and response.status_code in COMPATIBILITY_STATUS_CODES:
                response = self._post_compat(
                    image_path=image_path,
                    model_id=model_id,
                    task_prompt=task_prompt,
                )
            body = _json_or_text(response)
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Florence vision runner failed with status {response.status_code}: {body}"
                )
            _merge_partial_analysis(normalized, _normalize_response_for_task(body, task_prompt))

        if meeting_context_prompt:
            response = self._post_primary(
                image_path=image_path,
                model_id=model_id,
                task_prompt=meeting_context_prompt,
                task="custom",
                text=meeting_context_prompt,
            )
            body = _json_or_text(response)
            if response.status_code >= 400:
                normalized["meeting_context"]["error"] = str(body)
            else:
                context_text = _best_text_from_partial(
                    _normalize_response_for_task(body, meeting_context_prompt)
                )
                normalized["meeting_context"] = {
                    "supported": bool(context_text),
                    "prompt": meeting_context_prompt,
                    "text": context_text,
                    "error": "",
                }

        normalized["api_endpoint"] = self.last_endpoint
        return normalized

    def _post_primary(
        self,
        *,
        image_path: str,
        model_id: str,
        task_prompt: str,
        task: Optional[str] = None,
        text: str = "",
    ) -> Any:
        payload: Dict[str, Any] = {
            "model": model_id,
            "model_id": model_id,
            "input": _image_input_payload(image_path),
            "task": task or _prompt_to_task(task_prompt),
            "prompt": task_prompt,
        }
        if text:
            payload["text"] = text
        if self.use_livepeer_gateway:
            response = self._post_livepeer(
                _chat_completions_payload(
                    image_path=image_path,
                    model_id=model_id,
                    prompt=text or task_prompt,
                )
            )
            self.last_endpoint = LIVEPEER_CHAT_COMPLETIONS_ENDPOINT
        else:
            response = self._client.post(
                f"{self.base_url}{PRIMARY_VISION_ANALYZE_ENDPOINT}",
                json=payload,
            )
            self.last_endpoint = PRIMARY_VISION_ANALYZE_ENDPOINT
        return response

    def _post_compat(
        self,
        *,
        image_path: str,
        model_id: str,
        task_prompt: str,
    ) -> Any:
        payload = {
            "image": {"type": "base64", "value": _image_base64(image_path)},
            "model_id": model_id,
            "prompt": task_prompt,
        }
        response = self._client.post(f"{self.base_url}{COMPAT_LMM_INFER_ENDPOINT}", json=payload)
        self.last_endpoint = COMPAT_LMM_INFER_ENDPOINT
        return response

    def _post_livepeer(self, payload: Dict[str, Any]) -> Any:
        if not self.livepeer_api_key:
            raise ValueError(
                "vision_backend=livepeer_remote requires LIVEPEER_OPEN_CLEARINGHOUSE_API_KEY "
                "or an explicit livepeer_api_key block initializer"
            )
        open_response = self._client.post(
            f"{self.livepeer_base_url}/v1/jobs",
            headers={
                "X-API-Key": self.livepeer_api_key,
                "Livepeer-Open-Clearinghouse-SDK": SDK_IDENTITY,
            },
            json={
                "capability": self.capability,
                "offering": self.offering,
                "estimated_units": 1,
                "max_total_units": 4,
            },
        )
        open_body = _json_or_text(open_response)
        if open_response.status_code >= 400:
            return _BrokeredResponse(
                status_code=open_response.status_code,
                body=open_body,
                headers=dict(getattr(open_response, "headers", {}) or {}),
            )

        broker_url = str(open_body["broker_url"]).rstrip("/")
        payment_envelope = open_body["payment_envelope"]
        payment_header = (
            json.dumps(payment_envelope)
            if isinstance(payment_envelope, (dict, list))
            else str(payment_envelope)
        )
        broker_response = self._client.post(
            f"{broker_url}/v1/cap",
            headers={
                "Livepeer-Capability": self.capability,
                "Livepeer-Offering": self.offering,
                "Livepeer-Payment": payment_header,
                "Livepeer-Mode": str(open_body.get("mode", "http-reqresp@v0")),
                "Livepeer-Spec-Version": "0.1",
                "Livepeer-Request-Id": str(uuid.uuid4()),
            },
            json=payload,
        )
        broker_body = _json_or_text(broker_response)
        actual_units = 0
        if broker_response.status_code < 400:
            actual_units = _extract_actual_units(broker_response)
        settle_endpoint = str(open_body["settle_endpoint"])
        self._client.post(
            _append_path(self.livepeer_base_url, settle_endpoint),
            headers={
                "X-API-Key": self.livepeer_api_key,
                "Livepeer-Open-Clearinghouse-SDK": SDK_IDENTITY,
            },
            json={"actual_units": actual_units},
        )
        return _BrokeredResponse(
            status_code=broker_response.status_code,
            body=broker_body,
            headers=dict(getattr(broker_response, "headers", {}) or {}),
        )


def normalize_florence2_analysis_response(payload: Any) -> Dict[str, Any]:
    body = _unwrap_payload(payload)
    if isinstance(body, dict) and isinstance(body.get("results"), list) and body["results"]:
        first = body["results"][0]
        if isinstance(first, dict):
            body = first
    task_outputs = _task_outputs(body)

    caption = _first_text(
        body,
        task_outputs,
        keys=("caption", "short_caption", "<CAPTION>", "CAPTION", "response"),
    )
    detailed_caption = _first_text(
        body,
        task_outputs,
        keys=("detailed_caption", "description", "<DETAILED_CAPTION>", "DETAILED_CAPTION"),
    )
    ocr_text = _first_text(
        body,
        task_outputs,
        keys=("ocr_text", "ocr", "text", "<OCR>", "OCR"),
    )

    meeting_context = _coerce_meeting_context(body)
    return {
        "caption": caption,
        "detailed_caption": detailed_caption,
        "ocr_text": ocr_text,
        "meeting_context": meeting_context,
    }


def normalize_vision_backend(backend: str, *, fallback: str = "local") -> str:
    resolved = str(backend or fallback).strip().lower().replace("-", "_")
    if resolved in {"", "default"}:
        resolved = fallback
    if resolved in {"runner", "remote_runner", "local_runner"}:
        resolved = "remote"
    if resolved in {"livepeer", "livepeer_gateway", "clearinghouse"}:
        resolved = "livepeer_remote"
    if resolved not in {"local", "remote", "livepeer_remote"}:
        raise ValueError("vision_backend must be 'local', 'remote', or 'livepeer_remote'")
    return resolved


def _json_or_text(response: Any) -> Any:
    try:
        return response.json()
    except Exception:
        return response.text


def _append_path(base_url: str, path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return path_or_url
    return f"{base_url.rstrip('/')}/{path_or_url.lstrip('/')}"


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
    if actual_units is None:
        raise RuntimeError("Broker response missing Livepeer-Work-Units header")
    try:
        return int(actual_units)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid Livepeer-Work-Units header value: {actual_units!r}") from exc


def _unwrap_payload(payload: Any) -> Any:
    current = payload
    if isinstance(current, list) and current:
        current = current[0]
    if isinstance(current, dict):
        chat_text = _chat_completion_text(current)
        if chat_text and not any(
            key in current
            for key in (
                "analysis",
                "result",
                "results",
                "output",
                "data",
                "caption",
                "detailed_caption",
                "ocr_text",
                "ocr",
                "text",
            )
        ):
            return {"response": chat_text}
    while isinstance(current, dict):
        for key in ("analysis", "result", "results", "output", "data"):
            nested = current.get(key)
            if isinstance(nested, dict):
                current = nested
                break
            if isinstance(nested, list) and nested:
                current = nested[0]
                break
        else:
            return current
    return current


def _task_outputs(body: Any) -> Dict[str, Any]:
    if not isinstance(body, dict):
        return {}
    for key in ("tasks", "task_outputs", "outputs", "predictions"):
        value = body.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            outputs: Dict[str, Any] = {}
            for item in value:
                if not isinstance(item, dict):
                    continue
                task = item.get("task") or item.get("prompt") or item.get("name")
                if task:
                    outputs[str(task)] = item.get("text", item.get("response", item))
            if outputs:
                return outputs
    return {}


def _first_text(*payloads: Any, keys: tuple[str, ...]) -> str:
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for key in keys:
            if key in payload:
                return _normalize_text(payload[key])
    return ""


def _coerce_meeting_context(body: Any) -> Dict[str, Any]:
    context: Any = {}
    if isinstance(body, dict):
        context = (
            body.get("meeting_context")
            or body.get("meeting_context_text")
            or body.get("screen_context")
            or body.get("visual_context")
            or {}
        )
    if isinstance(context, dict):
        text = _normalize_text(context.get("text") or context.get("response") or context.get("raw_output"))
        return {
            "supported": bool(context.get("supported", bool(text))),
            "prompt": _normalize_text(context.get("prompt")),
            "text": text,
            "error": _normalize_text(context.get("error")),
        }
    text = _normalize_text(context)
    return {"supported": bool(text), "prompt": "", "text": text, "error": ""}


def _normalize_text(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, dict):
        parts = [_normalize_text(item) for item in value.values()]
        return " ".join(part for part in parts if part).strip()
    if isinstance(value, list):
        parts = [_normalize_text(item) for item in value]
        return " ".join(part for part in parts if part).strip()
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _image_base64(image_path: str) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")


def _image_input_payload(image_path: str) -> Dict[str, Any]:
    return {"type": "image_base64", "data": _image_base64(image_path)}


def _prompt_to_task(prompt: str) -> str:
    normalized = str(prompt or "").strip()
    mapping = {
        "<CAPTION>": "caption",
        "<DETAILED_CAPTION>": "detailed_caption",
        "<MORE_DETAILED_CAPTION>": "more_detailed_caption",
        "<OCR>": "ocr",
        "<OCR_WITH_REGION>": "ocr_with_region",
        "<OD>": "object_detection",
        "<DENSE_REGION_CAPTION>": "dense_region_caption",
        "<REGION_PROPOSAL>": "region_proposal",
        "<CAPTION_TO_PHRASE_GROUNDING>": "phrase_grounding",
        "<OPEN_VOCABULARY_DETECTION>": "open_vocabulary_detection",
        "<REGION_TO_DESCRIPTION>": "region_to_description",
        "<REGION_TO_OCR>": "region_to_ocr",
    }
    return mapping.get(normalized, "custom")


def _merge_partial_analysis(target: Dict[str, Any], partial: Dict[str, Any]) -> None:
    for key in ("caption", "detailed_caption", "ocr_text"):
        if partial.get(key):
            target[key] = partial[key]


def _best_text_from_partial(partial: Dict[str, Any]) -> str:
    for key in ("caption", "detailed_caption", "ocr_text"):
        if partial.get(key):
            return str(partial[key])
    return ""


def _normalize_response_for_task(payload: Any, task_prompt: str) -> Dict[str, Any]:
    partial = normalize_florence2_analysis_response(payload)
    generic_text = _generic_response_text(payload)
    if not generic_text:
        return partial

    task = _prompt_to_task(task_prompt)
    normalized = {
        "caption": "",
        "detailed_caption": "",
        "ocr_text": "",
        "meeting_context": partial.get("meeting_context") or {
            "supported": False,
            "prompt": "",
            "text": "",
            "error": "",
        },
    }
    if task == "caption":
        normalized["caption"] = partial.get("caption") or generic_text
        return normalized
    if task == "detailed_caption":
        normalized["detailed_caption"] = partial.get("detailed_caption") or generic_text
        return normalized
    if task == "ocr":
        normalized["ocr_text"] = partial.get("ocr_text") or generic_text
        return normalized
    normalized["caption"] = _best_text_from_partial(partial) or generic_text
    return normalized


def _generic_response_text(payload: Any) -> str:
    chat_text = _chat_completion_text(payload)
    if chat_text:
        return chat_text
    body = _unwrap_payload(payload)
    if isinstance(body, dict):
        for key in ("response", "text", "raw_output", "content"):
            if key in body:
                return _normalize_text(body[key])
    return ""


def _chat_completion_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message") or {}
    if not isinstance(message, dict):
        return ""
    return _normalize_chat_content(message.get("content"))


def _normalize_chat_content(content: Any) -> str:
    if isinstance(content, str):
        return _normalize_text(content)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type in {None, "text", "output_text"}:
                    parts.append(_normalize_text(item.get("text") or item.get("content")))
            else:
                parts.append(_normalize_text(item))
        return " ".join(part for part in parts if part).strip()
    return _normalize_text(content)


def _chat_completions_payload(*, image_path: str, model_id: str, prompt: str) -> Dict[str, Any]:
    return {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": str(prompt or "").strip()},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _image_data_url(image_path),
                        },
                    },
                ],
            }
        ],
        "temperature": 0,
    }


def _image_data_url(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower()
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/jpeg")
    return f"data:{mime_type};base64,{_image_base64(image_path)}"
