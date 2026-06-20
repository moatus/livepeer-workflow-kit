"""Vision helpers for screen and slide capture blocks."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from .config import (
    init_florence2_runner_url,
    init_open_clearinghouse_api_key,
    init_open_clearinghouse_url,
    init_vision_backend,
    init_vision_capability,
    init_vision_offering,
)
from .florence_client import Florence2VisionRunnerClient, normalize_vision_backend
from .ingest import _run_checked

DEFAULT_FLORENCE2_MODEL_ID = "florence-2-large"
DEFAULT_FLORENCE2_HF_MODEL_ID = "microsoft/Florence-2-large"
DEFAULT_FRAME_INTERVAL_SECONDS = 5.0
DEFAULT_MAX_FRAMES = 24
DEFAULT_MIN_SLIDE_GAP_SECONDS = 4.0
DEFAULT_SLIDE_CHANGE_THRESHOLD = 0.72
DEFAULT_MEETING_CONTEXT_PROMPT = (
    "Analyze this meeting screen capture. Separate presentation or screen-share "
    "content from chat messages, call controls, browser chrome, player controls, "
    "sidebars, and other page UI. Return concise text grouped by category."
)

PRESENTATION_HINT_KEYWORDS = (
    "slide",
    "slides",
    "presentation",
    "browser",
    "webpage",
    "dashboard",
    "spreadsheet",
    "document",
    "code editor",
    "terminal",
    "chart",
    "table",
    "whiteboard",
    "screen",
    "screen share",
    "ui",
)

SLIDE_HINT_KEYWORDS = (
    "slide",
    "slides",
    "presentation",
    "deck",
    "agenda",
    "roadmap",
    "overview",
    "appendix",
)

SCREEN_SHARE_HINT_KEYWORDS = (
    "screen share",
    "screen",
    "dashboard",
    "spreadsheet",
    "document",
    "code editor",
    "terminal",
    "whiteboard",
    "table",
    "chart",
    "browser window",
    "webpage",
)

CHAT_TEXT_HINTS = (
    "chat",
    "message",
    "messages",
    "send",
    "reply",
    "type a message",
    "to everyone",
    "everyone:",
    "privately",
    "direct message",
    "comments",
)

CALL_UI_TEXT_HINTS = (
    "mute",
    "unmute",
    "microphone",
    "camera",
    "participants",
    "raise hand",
    "reactions",
    "leave",
    "join",
    "meeting",
    "stop sharing",
    "share screen",
    "recording",
    "captions",
    "breakout",
)

BROWSER_OR_PLAYER_CHROME_HINTS = (
    "http://",
    "https://",
    "www.",
    ".com",
    "address bar",
    "search google",
    "reload",
    "bookmark",
    "tab",
    "chrome",
    "safari",
    "firefox",
    "edge",
    "play",
    "pause",
    "volume",
    "fullscreen",
    "settings",
    "quality",
    "captions",
    "youtube",
    "vdo.ninja",
)

LONG_SCREEN_SHARE_PATTERNS = (
    r"\bQuick Recap\s*[-:]\s*[^.]{3,80}",
    r"\bHere is [^.]{12,180}",
    r"\bBy steadily [^.]{12,240}",
    r"\bToday, [^.]{12,220}",
    r"\bIMMEDIATE INCREASE\b[^.]{0,220}",
    r"\bDELAYED INCREASES?\b[^.]{0,220}",
    r"\bFake ?Boost Amount\s*:\s*[^.]{1,80}",
    r"\bCost per week\s*:\s*[^.]{1,100}",
    r"\bReturn ?on rewards\s*:\s*[^.]{1,100}",
    r"\b\d+\s*Week(?:ly)? (?:cost|profit)\s*:\s*[^.]{1,100}",
)

LONG_CHAT_PATTERNS = (
    r"\bCommunity Calls\b",
    r"\bActivity\b",
    r"\b\w[\w.-]{1,40}\s+Today\s+at\s+\d{1,2}[:.]\d{2}\s*(?:AM|PM)?",
    r"\b\d+\s+Messages?\b",
    r"\b(?:Pokt News|Relays Rewards)\b",
)

LONG_CALL_UI_PATTERNS = (
    r"\bjoined (?:the )?recording\b",
    r"\bStop recording\b",
    r"\bAdd (?:a )?note\b",
    r"\brecording panel\b",
)


@dataclass(frozen=True)
class FrameSample:
    index: int
    timestamp_seconds: float
    image_path: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp_seconds": self.timestamp_seconds,
            "image_path": self.image_path,
        }


def extract_video_frames(
    *,
    recording_path: str | Path,
    output_dir: str | Path,
    frame_interval_seconds: float = DEFAULT_FRAME_INTERVAL_SECONDS,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> List[Dict[str, Any]]:
    if frame_interval_seconds <= 0:
        raise ValueError("frame_interval_seconds must be positive")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")

    frames_dir = Path(output_dir).resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)
    pattern = frames_dir / "frame_%06d.jpg"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(recording_path),
        "-vf",
        f"fps=1/{frame_interval_seconds:g}",
        "-frames:v",
        str(max_frames),
        str(pattern),
    ]
    _run_checked(command)

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    samples = [
        FrameSample(
            index=index,
            timestamp_seconds=round(index * float(frame_interval_seconds), 3),
            image_path=str(path),
        ).as_dict()
        for index, path in enumerate(frame_paths)
    ]
    return samples


def normalize_visual_text(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, dict):
        parts = [normalize_visual_text(item) for item in value.values()]
        return " ".join(part for part in parts if part).strip()
    if isinstance(value, list):
        parts = [normalize_visual_text(item) for item in value]
        return " ".join(part for part in parts if part).strip()
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def similarity_ratio(left: str, right: str) -> float:
    left_norm = normalize_visual_text(left).lower()
    right_norm = normalize_visual_text(right).lower()
    if not left_norm and not right_norm:
        return 1.0
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def presentation_likelihood(*, caption: str, detailed_caption: str, ocr_text: str) -> bool:
    normalized = " ".join(
        normalize_visual_text(part).lower()
        for part in (caption, detailed_caption, ocr_text)
        if normalize_visual_text(part)
    )
    if len(normalize_visual_text(ocr_text)) >= 24:
        return True
    return any(keyword in normalized for keyword in PRESENTATION_HINT_KEYWORDS)


def make_slide_signal_text(*, caption: str, detailed_caption: str, ocr_text: str) -> str:
    text = normalize_visual_text(ocr_text)
    if len(text) >= 24:
        return text.lower()
    combined = " ".join(
        normalize_visual_text(part)
        for part in (caption, detailed_caption, ocr_text)
        if normalize_visual_text(part)
    )
    return combined.lower()


def split_visual_text_lines(value: Any) -> List[str]:
    text = normalize_visual_text(value)
    if not text:
        return []
    raw_lines = re.split(r"[\n\r]+| {2,}|[•\u2022]\s*", str(value))
    lines = [normalize_visual_text(line) for line in raw_lines]
    if len(lines) <= 1:
        lines = [part.strip() for part in re.split(r"\s+[|]\s+", text)]
    return [line for line in lines if line]


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    normalized = normalize_visual_text(text).lower()
    return any(keyword in normalized for keyword in keywords)


def _append_unique(bucket: List[str], value: str) -> None:
    normalized = normalize_visual_text(value)
    if normalized and normalized not in bucket:
        bucket.append(normalized)


def _append_pattern_matches(bucket: List[str], text: str, patterns: tuple[str, ...]) -> bool:
    matched = False
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            _append_unique(bucket, match.group(0))
            matched = True
    return matched


def _classify_ocr_line(*, line: str, caption_context: str) -> str:
    normalized = normalize_visual_text(line).lower()
    context = normalize_visual_text(caption_context).lower()

    if _contains_any(normalized, CHAT_TEXT_HINTS):
        return "chat_text"
    if _contains_any(normalized, BROWSER_OR_PLAYER_CHROME_HINTS):
        return "browser_or_player_chrome_text"
    if _contains_any(normalized, CALL_UI_TEXT_HINTS):
        return "call_ui_text"
    if normalized in {"copy link", "share", "more", "close", "back", "next", "previous"}:
        return "other_page_chrome_text"

    if _contains_any(context, SLIDE_HINT_KEYWORDS):
        return "slide_text"
    if _contains_any(context, SCREEN_SHARE_HINT_KEYWORDS):
        return "screen_share_text"
    if len(normalized) >= 24:
        return "screen_share_text"
    return "other_page_chrome_text" if len(normalized) <= 14 else "screen_share_text"


def separate_meeting_visual_text(
    *,
    caption: str,
    detailed_caption: str,
    ocr_text: str,
    meeting_context: Any = None,
) -> Dict[str, Any]:
    """Best-effort local splitter for meeting screen text without OCR boxes.

    The schema is intentionally additive and runner-agnostic: future remote
    visual runners can replace the heuristic while preserving the same buckets.
    """

    caption_context = " ".join(
        part for part in (normalize_visual_text(caption), normalize_visual_text(detailed_caption)) if part
    )
    buckets: Dict[str, List[str]] = {
        "slide_text": [],
        "screen_share_text": [],
        "chat_text": [],
        "call_ui_text": [],
        "browser_or_player_chrome_text": [],
        "other_page_chrome_text": [],
    }

    for line in split_visual_text_lines(ocr_text):
        if len(line) > 180:
            found_content = _append_pattern_matches(
                buckets["screen_share_text"],
                line,
                LONG_SCREEN_SHARE_PATTERNS,
            )
            found_chat = _append_pattern_matches(buckets["chat_text"], line, LONG_CHAT_PATTERNS)
            found_call_ui = _append_pattern_matches(
                buckets["call_ui_text"],
                line,
                LONG_CALL_UI_PATTERNS,
            )
            found_browser = _append_pattern_matches(
                buckets["browser_or_player_chrome_text"],
                line,
                BROWSER_OR_PLAYER_CHROME_HINTS,
            )
            if found_content or found_chat or found_call_ui or found_browser:
                continue
        bucket = _classify_ocr_line(line=line, caption_context=caption_context)
        _append_unique(buckets[bucket], line)

    meeting_context_text = normalize_visual_text(meeting_context)
    if meeting_context_text:
        context_lower = meeting_context_text.lower()
        for bucket_name in buckets:
            if bucket_name.replace("_", " ") in context_lower:
                _append_unique(buckets[bucket_name], meeting_context_text)
                break

    content_roles: List[str] = []
    if buckets["slide_text"] or _contains_any(caption_context, SLIDE_HINT_KEYWORDS):
        content_roles.append("slide")
    if buckets["screen_share_text"] or _contains_any(caption_context, SCREEN_SHARE_HINT_KEYWORDS):
        content_roles.append("screen_share")
    if buckets["chat_text"] or _contains_any(caption_context, CHAT_TEXT_HINTS):
        content_roles.append("chat")
    if buckets["call_ui_text"] or _contains_any(caption_context, CALL_UI_TEXT_HINTS):
        content_roles.append("call_ui")
    if buckets["browser_or_player_chrome_text"] or _contains_any(
        caption_context, BROWSER_OR_PLAYER_CHROME_HINTS
    ):
        content_roles.append("browser_or_player_chrome")
    if buckets["other_page_chrome_text"]:
        content_roles.append("other_page_chrome")

    content_text = buckets["slide_text"] + buckets["screen_share_text"]
    chrome_text = (
        buckets["chat_text"]
        + buckets["call_ui_text"]
        + buckets["browser_or_player_chrome_text"]
        + buckets["other_page_chrome_text"]
    )
    if content_text and chrome_text:
        confidence = 0.72
    elif content_text or chrome_text:
        confidence = 0.64
    else:
        confidence = 0.0

    return {
        "schema_version": "livepeer.meeting_visual_text.v1",
        "classification_method": "local_heuristic_v1",
        "slide_text": "\n".join(buckets["slide_text"]),
        "screen_share_text": "\n".join(buckets["screen_share_text"]),
        "chat_text": "\n".join(buckets["chat_text"]),
        "call_ui_text": "\n".join(buckets["call_ui_text"]),
        "browser_or_player_chrome_text": "\n".join(buckets["browser_or_player_chrome_text"]),
        "other_page_chrome_text": "\n".join(buckets["other_page_chrome_text"]),
        "text_buckets": buckets,
        "content_roles": content_roles,
        "primary_content_role": content_roles[0] if content_roles else "unknown",
        "separation_confidence": round(confidence, 2),
        "raw_ocr_text": normalize_visual_text(ocr_text),
        "meeting_context_text": meeting_context_text,
    }


def summarize_meeting_visual_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    bucket_names = (
        "slide_text",
        "screen_share_text",
        "chat_text",
        "call_ui_text",
        "browser_or_player_chrome_text",
        "other_page_chrome_text",
    )
    aggregate: Dict[str, List[str]] = {name: [] for name in bucket_names}
    for event in events:
        for name in bucket_names:
            for line in split_visual_text_lines(event.get(name, "")):
                _append_unique(aggregate[name], line)

    return {
        "schema_version": "livepeer.meeting_visual_summary.v1",
        "frame_count": len(events),
        "slide_text": "\n".join(aggregate["slide_text"]),
        "screen_share_text": "\n".join(aggregate["screen_share_text"]),
        "chat_text": "\n".join(aggregate["chat_text"]),
        "call_ui_text": "\n".join(aggregate["call_ui_text"]),
        "browser_or_player_chrome_text": "\n".join(aggregate["browser_or_player_chrome_text"]),
        "other_page_chrome_text": "\n".join(aggregate["other_page_chrome_text"]),
        "text_buckets": aggregate,
        "chat_or_ui_text_separated": bool(
            aggregate["chat_text"]
            or aggregate["call_ui_text"]
            or aggregate["browser_or_player_chrome_text"]
            or aggregate["other_page_chrome_text"]
        ),
    }


def write_json(path: str | Path, payload: Any) -> str:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(target)


def write_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> str:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")
    return str(target)


def copy_slide_frame(*, source_path: str | Path, slides_dir: str | Path, slide_index: int) -> str:
    target_dir = Path(slides_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / f"slide_{slide_index:04d}.jpg"
    shutil.copy2(str(source_path), str(destination))
    return str(destination)


class Florence2InferenceAnalyzer:
    """Thin adapter around local Florence-2 and remote vision runner backends."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_FLORENCE2_MODEL_ID,
        api_key: Optional[str] = None,
        api_url: str = "",
        vision_backend: Optional[str] = None,
        runner_url: Optional[str] = None,
        livepeer_api_key: Optional[str] = None,
        livepeer_base_url: Optional[str] = None,
        livepeer_capability: Optional[str] = None,
        livepeer_offering: Optional[str] = None,
        remote_client_cls: Any = Florence2VisionRunnerClient,
    ) -> None:
        self._model_id = model_id
        self._api_key = api_key
        self._api_url = api_url
        self._vision_backend = normalize_vision_backend(vision_backend or init_vision_backend())
        self._runner_url = runner_url or init_florence2_runner_url()
        self._livepeer_api_key = (
            livepeer_api_key
            if livepeer_api_key is not None
            else init_open_clearinghouse_api_key()
        )
        self._livepeer_base_url = (
            livepeer_base_url
            if livepeer_base_url is not None
            else init_open_clearinghouse_url()
        )
        self._livepeer_capability = (
            livepeer_capability
            if livepeer_capability is not None
            else init_vision_capability()
        )
        self._livepeer_offering = (
            livepeer_offering
            if livepeer_offering is not None
            else init_vision_offering()
        )
        self._remote_client_cls = remote_client_cls
        self._remote_client: Any = None
        self._rf_model: Any = None
        self._hf_bundle: Any = None

    def analyze_image(self, image_path: str, meeting_context_prompt: str = "") -> Dict[str, Any]:
        if self._vision_backend in {"remote", "livepeer_remote"}:
            return self._analyze_image_via_remote_runner(
                image_path=image_path,
                meeting_context_prompt=meeting_context_prompt,
            )

        caption = self._run_prompt(image_path=image_path, prompt="<CAPTION>")
        detailed_caption = self._run_prompt(image_path=image_path, prompt="<DETAILED_CAPTION>")
        ocr_text = self._run_prompt(image_path=image_path, prompt="<OCR>")
        meeting_context: Dict[str, Any] = {
            "supported": False,
            "prompt": meeting_context_prompt,
            "text": "",
            "error": "",
        }
        if meeting_context_prompt and (self._api_key or self._api_url):
            try:
                meeting_context["text"] = self._run_prompt(
                    image_path=image_path,
                    prompt=meeting_context_prompt,
                )
                meeting_context["supported"] = bool(meeting_context["text"])
            except BaseException as exc:
                meeting_context["error"] = str(exc)
        return {
            "caption": caption,
            "detailed_caption": detailed_caption,
            "ocr_text": ocr_text,
            "meeting_context": meeting_context,
        }

    def _analyze_image_via_remote_runner(
        self,
        *,
        image_path: str,
        meeting_context_prompt: str,
    ) -> Dict[str, Any]:
        client = self._ensure_remote_client()
        return client.analyze_image(
            image_path=image_path,
            model_id=self._model_id,
            meeting_context_prompt=meeting_context_prompt,
        )

    def _ensure_remote_client(self) -> Any:
        if self._remote_client is None:
            try:
                self._remote_client = self._remote_client_cls(
                    base_url=self._runner_url,
                    livepeer_api_key=self._livepeer_api_key,
                    livepeer_base_url=self._livepeer_base_url,
                    capability=self._livepeer_capability,
                    offering=self._livepeer_offering,
                    use_livepeer_gateway=self._vision_backend == "livepeer_remote",
                )
            except TypeError:
                self._remote_client = self._remote_client_cls(base_url=self._runner_url)
        return self._remote_client

    def _run_prompt(self, *, image_path: str, prompt: str) -> str:
        if self._api_key or self._api_url:
            model = self._ensure_rf_model()
            result = model.infer(image_path, prompt=prompt)
            return self._extract_text(result)
        return self._run_prompt_via_transformers(image_path=image_path, prompt=prompt)

    def _ensure_rf_model(self) -> Any:
        if self._rf_model is not None:
            return self._rf_model
        try:
            from inference import get_model
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Roboflow-backed Florence-2 analysis requires the `inference` package. "
                "Install `inference[transformers]` or use the direct transformers path without ROBOFLOW_API_KEY."
            ) from exc

        kwargs: Dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_url:
            kwargs["api_url"] = self._api_url
        try:
            self._rf_model = get_model(self._model_id, **kwargs)
        except TypeError:
            kwargs.pop("api_url", None)
            self._rf_model = get_model(self._model_id, **kwargs)
        return self._rf_model

    def _ensure_hf_bundle(self) -> Dict[str, Any]:
        if self._hf_bundle is not None:
            return self._hf_bundle
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Direct Florence-2 analysis requires `torch` and `transformers`. "
                "Install the Florence runtime dependencies in the container first."
            ) from exc

        hf_model_id = self._resolve_hf_model_id(self._model_id)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
            trust_remote_code=True,
        ).to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
        )
        self._hf_bundle = {
            "torch": torch,
            "device": device,
            "torch_dtype": torch_dtype,
            "model": model,
            "processor": processor,
        }
        return self._hf_bundle

    def _run_prompt_via_transformers(self, *, image_path: str, prompt: str) -> str:
        bundle = self._ensure_hf_bundle()
        torch = bundle["torch"]
        device = bundle["device"]
        torch_dtype = bundle["torch_dtype"]
        model = bundle["model"]
        processor = bundle["processor"]

        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        pixel_values = inputs["pixel_values"].to(device=device, dtype=torch_dtype)
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=512,
                num_beams=1,
                do_sample=False,
                use_cache=False,
            )
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]
        parsed = processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image.width, image.height),
        )
        return self._extract_text(parsed)

    @staticmethod
    def _resolve_hf_model_id(model_id: str) -> str:
        normalized = (model_id or "").strip()
        if normalized.startswith("microsoft/") or normalized.startswith("florence-community/"):
            return normalized
        if normalized == "florence-2-large":
            return DEFAULT_FLORENCE2_HF_MODEL_ID
        if normalized == "florence-2-base":
            return "microsoft/Florence-2-base"
        return normalized or DEFAULT_FLORENCE2_HF_MODEL_ID

    @staticmethod
    def _extract_text(result: Any) -> str:
        payload = result
        if isinstance(payload, list) and payload:
            payload = payload[0]

        if hasattr(payload, "response"):
            return normalize_visual_text(getattr(payload, "response"))
        if hasattr(payload, "text"):
            return normalize_visual_text(getattr(payload, "text"))

        if isinstance(payload, dict):
            for key in ("response", "text", "raw_output"):
                if key in payload:
                    return normalize_visual_text(payload[key])
            if "parsed_output" in payload:
                return normalize_visual_text(payload["parsed_output"])

        return normalize_visual_text(payload)
