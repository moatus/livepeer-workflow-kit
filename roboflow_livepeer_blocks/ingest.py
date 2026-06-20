"""Raspberry.Ninja-backed VDO.Ninja ingest helpers for Roboflow blocks."""

from __future__ import annotations

import json
import hashlib
import os
import re
import signal
import shutil
import ssl
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse, urlunparse
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RASPBERRY_NINJA = REPO_ROOT / "references" / "raspberry_ninja" / "publish.py"
DEFAULT_INGEST_OUTPUT_DIR = REPO_ROOT / "artifacts" / "rolling-ingest"
MEDIA_SUFFIXES = {".webm", ".mkv", ".ts", ".mp4"}
AUTO_VDO_SOURCE_TOKENS = {"", "*", "any", "auto"}
DEFAULT_RASPBERRY_NINJA_PASSWORD = "someEncryptionKey123"


@dataclass(frozen=True)
class RollingSegment:
    index: int
    stream_id: str
    recording_path: Path
    audio_path: Path
    log_path: Path
    started_at_epoch: float
    completed_at_epoch: float
    requested_duration_seconds: float
    audio_duration_seconds: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "stream_id": self.stream_id,
            "recording_path": str(self.recording_path),
            "audio_path": str(self.audio_path),
            "log_path": str(self.log_path),
            "started_at_epoch": self.started_at_epoch,
            "completed_at_epoch": self.completed_at_epoch,
            "requested_duration_seconds": self.requested_duration_seconds,
            "audio_duration_seconds": self.audio_duration_seconds,
        }


def parse_vdo_stream_id(source: str) -> str:
    """Accept a raw VDO.Ninja stream ID or a URL containing a view parameter."""
    source = source.strip()
    if not source:
        raise ValueError("source must contain a VDO.Ninja stream id or viewer URL")
    parsed = urlparse(source)
    query = parse_qs(parsed.query)
    values = query.get("view") or query.get("streamid") or query.get("streamID")
    if values and values[0].strip():
        return values[0].strip()
    if re.fullmatch(r"[A-Za-z0-9_.:-]+", source):
        return source
    raise ValueError(f"Could not find a VDO.Ninja stream id in {source!r}")


def is_auto_vdo_source(source: str) -> bool:
    """Return whether a source means "select a live bridge stream automatically"."""
    source = (source or "").strip()
    if source.lower() in AUTO_VDO_SOURCE_TOKENS:
        return True
    parsed = urlparse(source)
    if parsed.scheme.lower() not in {"ws", "wss", "http", "https"}:
        return False
    query = parse_qs(parsed.query)
    stream_values = query.get("view") or query.get("streamid") or query.get("streamID")
    return not any(value.strip() for value in stream_values or [])


def resolve_vdo_stream_source(
    *,
    source: str,
    signaling_server: str = "",
    password: str = "",
    timeout_seconds: float = 30.0,
    poll_interval_seconds: float = 1.0,
    selection: str = "latest",
) -> Dict[str, Any]:
    """Resolve either an explicit VDO stream ID or an automatic bridge stream source."""
    requested_source = (source or "").strip()
    resolved_signaling_server = (signaling_server or "").strip()
    if _looks_like_bridge_url(requested_source):
        resolved_signaling_server = requested_source
        requested_source = "auto"

    if not is_auto_vdo_source(requested_source):
        stream_id = parse_vdo_stream_id(requested_source)
        resolved_password = infer_explicit_vdo_password(
            source=requested_source,
            stream_id=stream_id,
            signaling_server=resolved_signaling_server,
            password=password,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        return {
            "source": requested_source,
            "stream_id": stream_id,
            "signaling_server": resolved_signaling_server,
            "password": resolved_password,
            "auto_resolved": False,
            "requested_source": requested_source,
        }

    if not resolved_signaling_server:
        raise ValueError("signaling_server is required when source is auto")
    status_url = vdo_bridge_status_url(resolved_signaling_server)
    bridge_stream_id = wait_for_vdo_bridge_stream(
        status_url=status_url,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        selection=selection,
    )
    stream_id = strip_raspberry_ninja_hash_suffix(
        bridge_stream_id,
        signaling_server=resolved_signaling_server,
        password=password,
    )
    resolved_password = password
    if not password and stream_id == bridge_stream_id:
        resolved_password = "false"
    return {
        "source": stream_id,
        "stream_id": stream_id,
        "bridge_stream_id": bridge_stream_id,
        "signaling_server": resolved_signaling_server,
        "password": resolved_password,
        "auto_resolved": True,
        "requested_source": requested_source or "auto",
        "status_url": status_url,
        "selection": selection,
    }


def infer_explicit_vdo_password(
    *,
    source: str,
    stream_id: str,
    signaling_server: str,
    password: str = "",
    timeout_seconds: float = 30.0,
    poll_interval_seconds: float = 1.0,
) -> str:
    """Infer unencrypted self-hosted bridge streams for explicit VDO sources."""
    if password:
        return password
    parsed = urlparse((source or "").strip())
    if parsed.scheme.lower() not in {"ws", "wss", "http", "https"}:
        return password
    query = parse_qs(parsed.query)
    password_values = query.get("password") or query.get("pass")
    if password_values:
        return password_values[0].strip()
    if not signaling_server:
        return password
    signaling_parsed = urlparse(signaling_server.strip())
    if parsed.netloc and signaling_parsed.netloc and parsed.netloc == signaling_parsed.netloc:
        return "false"

    try:
        status_url = vdo_bridge_status_url(signaling_server)
    except ValueError:
        return password
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    suffix = raspberry_ninja_hash_suffix(signaling_server=signaling_server, password=password)
    suffixed_stream_id = f"{stream_id}{suffix}" if suffix else stream_id
    last_error: Optional[BaseException] = None
    while True:
        try:
            payload = _read_vdo_bridge_status(status_url)
            streams = payload.get("streams") or {}
            if isinstance(streams, dict):
                stream_ids = {str(candidate) for candidate in streams.keys()}
                if stream_id in stream_ids:
                    return "false"
                if suffixed_stream_id in stream_ids:
                    return password
        except BaseException as exc:
            last_error = exc
        if time.monotonic() >= deadline:
            if last_error:
                return password
            return password
        time.sleep(min(poll_interval_seconds, max(0.0, deadline - time.monotonic())))


def vdo_bridge_status_url(signaling_server: str) -> str:
    parsed = urlparse(signaling_server.strip())
    if parsed.scheme.lower() not in {"ws", "wss", "http", "https"}:
        raise ValueError(f"Unsupported VDO signaling server URL: {signaling_server!r}")
    scheme = "https" if parsed.scheme.lower() in {"wss", "https"} else "http"
    return urlunparse((scheme, parsed.netloc, "/statusz", "", "", ""))


def strip_raspberry_ninja_hash_suffix(
    stream_id: str,
    *,
    signaling_server: str,
    password: str = DEFAULT_RASPBERRY_NINJA_PASSWORD,
) -> str:
    suffix = raspberry_ninja_hash_suffix(signaling_server=signaling_server, password=password)
    if suffix and stream_id.endswith(suffix) and len(stream_id) > len(suffix):
        return stream_id[: -len(suffix)]
    return stream_id


def raspberry_ninja_hash_suffix(
    *,
    signaling_server: str,
    password: str = DEFAULT_RASPBERRY_NINJA_PASSWORD,
) -> str:
    effective_password = password or DEFAULT_RASPBERRY_NINJA_PASSWORD
    return hashlib.sha256(f"{effective_password}vdo.ninja".encode("utf-8")).digest()[:3].hex()


def wait_for_vdo_bridge_stream(
    *,
    status_url: str,
    timeout_seconds: float = 30.0,
    poll_interval_seconds: float = 1.0,
    selection: str = "latest",
) -> str:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be positive")
    deadline = time.monotonic() + timeout_seconds
    last_error: Optional[BaseException] = None
    while True:
        try:
            payload = _read_vdo_bridge_status(status_url)
            stream_id = select_vdo_bridge_stream(payload, selection=selection)
            if stream_id:
                return stream_id
        except BaseException as exc:
            last_error = exc
        if time.monotonic() >= deadline:
            detail = f": {last_error}" if last_error else ""
            raise TimeoutError(
                f"timed out after {timeout_seconds:.1f}s waiting for a VDO bridge stream{detail}"
            )
        time.sleep(min(poll_interval_seconds, max(0.0, deadline - time.monotonic())))


def select_vdo_bridge_stream(payload: Dict[str, Any], *, selection: str = "latest") -> Optional[str]:
    stream_ids = _vdo_bridge_stream_ids(payload)
    if not stream_ids:
        return None
    candidates: List[tuple[float, str]] = []
    clients = payload.get("clients") or []
    if isinstance(clients, list):
        for client in clients:
            if not isinstance(client, dict):
                continue
            stream_id = (
                client.get("stream_id")
                or client.get("streamID")
                or client.get("sid")
            )
            if not stream_id or str(stream_id) not in stream_ids:
                continue
            connected_at = client.get("connected_at_epoch") or 0.0
            try:
                connected_at_value = float(connected_at)
            except (TypeError, ValueError):
                connected_at_value = 0.0
            candidates.append((connected_at_value, str(stream_id)))
    seen = {stream_id for _, stream_id in candidates}
    for stream_id in sorted(stream_ids - seen):
        candidates.append((0.0, stream_id))
    if not candidates:
        return None
    normalized_selection = selection.strip().lower()
    if normalized_selection in {"oldest", "first"}:
        return min(candidates, key=lambda item: (item[0], item[1]))[1]
    if normalized_selection in {"latest", "newest", "last", "any", "auto"}:
        return max(candidates, key=lambda item: (item[0], item[1]))[1]
    raise ValueError(f"Unsupported VDO stream selection: {selection!r}")


def _vdo_bridge_stream_ids(payload: Dict[str, Any]) -> set[str]:
    stream_ids: set[str] = set()

    streams = payload.get("streams") or {}
    if isinstance(streams, dict):
        stream_ids.update(str(stream_id) for stream_id in streams.keys() if str(stream_id))
    elif isinstance(streams, list):
        for entry in streams:
            if isinstance(entry, str):
                stream_ids.add(entry)
            elif isinstance(entry, dict):
                stream_id = entry.get("stream_id") or entry.get("streamID") or entry.get("sid")
                if stream_id:
                    stream_ids.add(str(stream_id))

    stream_ids_by_uuid = payload.get("stream_ids_by_uuid") or payload.get("streamIDs") or {}
    if isinstance(stream_ids_by_uuid, dict):
        stream_ids.update(str(stream_id) for stream_id in stream_ids_by_uuid.values() if str(stream_id))

    clients = payload.get("clients") or []
    if isinstance(clients, list):
        for client in clients:
            if not isinstance(client, dict):
                continue
            stream_id = client.get("stream_id") or client.get("streamID") or client.get("sid")
            if stream_id:
                stream_ids.add(str(stream_id))

    return stream_ids


def _read_vdo_bridge_status(status_url: str) -> Dict[str, Any]:
    context = ssl._create_unverified_context() if urlparse(status_url).scheme == "https" else None
    with urlopen(status_url, timeout=5, context=context) as response:
        return json.loads(response.read().decode("utf-8"))


def _looks_like_bridge_url(source: str) -> bool:
    parsed = urlparse((source or "").strip())
    if parsed.scheme.lower() not in {"ws", "wss", "http", "https"}:
        return False
    query = parse_qs(parsed.query)
    stream_values = query.get("view") or query.get("streamid") or query.get("streamID")
    return not any(value.strip() for value in stream_values or [])


def safe_file_prefix(value: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return prefix or "vdo_stream"


def _run_checked(command: List[str], *, cwd: Optional[Path] = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        capture_output=True,
    )


def probe_media_duration_seconds(path: str | Path) -> float:
    result = _run_checked(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(result.stdout)
    duration = float(payload["format"]["duration"])
    if duration <= 0:
        raise ValueError(f"Media duration must be positive for {path}")
    return duration


def extract_audio_to_wav(recording_path: str | Path, audio_path: str | Path) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(recording_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(audio_path),
    ]
    _run_checked(command)


def _raspberry_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("GI_TYPELIB_PATH", "/usr/lib/x86_64-linux-gnu/girepository-1.0")
    env.setdefault("RN_FORCE_SINK", "fakesink sync=true async=false")
    env.setdefault("RN_FORCE_AUDIO_SINK", "fakesink sync=true async=false")
    env.setdefault("XDG_RUNTIME_DIR", "/tmp/xdg-runtime")
    Path(env["XDG_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)
    return env


def _python_can_import_gi(executable: str) -> bool:
    try:
        result = subprocess.run(
            [executable, "-c", "import gi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def raspberry_ninja_python_executable() -> str:
    override = os.environ.get("RASPBERRY_NINJA_PYTHON")
    if override:
        return override
    candidates = [sys.executable, shutil.which("python3"), "/usr/bin/python3"]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if _python_can_import_gi(candidate):
            return candidate
    return sys.executable


def record_vdo_segment(
    *,
    source: str,
    output_dir: str | Path = DEFAULT_INGEST_OUTPUT_DIR,
    duration_seconds: float = 30.0,
    startup_seconds: float = 8.0,
    password: str = "",
    signaling_server: str = "",
    buffer_ms: int = 300,
    audio_only: bool = True,
    allow_missing_audio: bool = False,
    segment_index: int = 0,
    raspberry_ninja_path: str | Path = DEFAULT_RASPBERRY_NINJA,
) -> RollingSegment:
    if duration_seconds < 0:
        raise ValueError("duration_seconds must not be negative")
    if startup_seconds < 0:
        raise ValueError("startup_seconds must not be negative")
    if buffer_ms < 10:
        raise ValueError("buffer_ms must be at least 10")

    stream_id = parse_vdo_stream_id(source)
    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    before = {path.resolve() for path in target_dir.glob("*") if path.is_file()}
    timestamp = int(time.time())
    record_prefix = f"{safe_file_prefix(stream_id)}_seg{segment_index:04d}_{timestamp}"
    log_path = target_dir / f"raspberry_ninja_{record_prefix}.log"
    command = [
        raspberry_ninja_python_executable(),
        str(raspberry_ninja_path),
        "--view",
        stream_id,
        "--record",
        record_prefix,
        "--buffer",
        str(buffer_ms),
    ]
    if password:
        command.extend(["--password", password])
    if signaling_server:
        command.extend(["--server", signaling_server])
    if audio_only:
        command.append("--novideo")

    started_at = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(target_dir),
            env=_raspberry_env(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            time.sleep(max(0.0, startup_seconds))
            if duration_seconds == 0:
                process.wait()
            else:
                time.sleep(max(0.0, duration_seconds))
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=15)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    completed_at = time.time()

    recording_path = newest_recording_path(
        target_dir=target_dir,
        before=before,
        log_path=log_path,
        prefer_audio=audio_only,
    )
    audio_path = target_dir / f"{recording_path.stem}.transcribe.wav"
    try:
        extract_audio_to_wav(recording_path=recording_path, audio_path=audio_path)
        audio_duration = probe_media_duration_seconds(audio_path)
    except subprocess.CalledProcessError:
        if not allow_missing_audio:
            raise
        audio_duration = 0.0
    return RollingSegment(
        index=segment_index,
        stream_id=stream_id,
        recording_path=recording_path,
        audio_path=audio_path,
        log_path=log_path,
        started_at_epoch=started_at,
        completed_at_epoch=completed_at,
        requested_duration_seconds=duration_seconds,
        audio_duration_seconds=audio_duration,
    )


def newest_recording_path(
    *,
    target_dir: Path,
    before: set[Path],
    log_path: Path,
    prefer_audio: bool = True,
) -> Path:
    candidates = [
        path
        for path in target_dir.glob("*")
        if path.is_file()
        and path.resolve() not in before
        and path.suffix.lower() in MEDIA_SUFFIXES
        and path.stat().st_size > 0
    ]
    if not candidates:
        log_tail = log_path.read_text(errors="replace")[-4000:] if log_path.exists() else ""
        raise RuntimeError(
            "Raspberry.Ninja did not produce a media recording. "
            f"Log: {log_path}\n--- log tail ---\n{log_tail}"
        )
    audio_candidates = [path for path in candidates if "_audio." in path.name]
    video_candidates = [path for path in candidates if "_audio." not in path.name]
    preferred_candidates = audio_candidates if prefer_audio else video_candidates
    return max(preferred_candidates or candidates, key=lambda path: path.stat().st_mtime)


def capture_rolling_audio_segments(
    *,
    source: str,
    output_dir: str | Path = DEFAULT_INGEST_OUTPUT_DIR,
    segment_count: int = 1,
    segment_duration_seconds: float = 30.0,
    startup_seconds: float = 8.0,
    password: str = "",
    signaling_server: str = "",
    buffer_ms: int = 300,
    audio_only: bool = True,
    raspberry_ninja_path: str | Path = DEFAULT_RASPBERRY_NINJA,
) -> Dict[str, Any]:
    if segment_count <= 0:
        raise ValueError("segment_count must be positive")
    segments = [
        record_vdo_segment(
            source=source,
            output_dir=output_dir,
            duration_seconds=segment_duration_seconds,
            startup_seconds=startup_seconds if index == 0 else 0.0,
            password=password,
            signaling_server=signaling_server,
            buffer_ms=buffer_ms,
            audio_only=audio_only,
            allow_missing_audio=False,
            segment_index=index,
            raspberry_ninja_path=raspberry_ninja_path,
        )
        for index in range(segment_count)
    ]
    segment_dicts = [segment.as_dict() for segment in segments]
    audio_paths = [segment["audio_path"] for segment in segment_dicts]
    return {
        "stream_id": segments[0].stream_id,
        "output_dir": str(Path(output_dir).resolve()),
        "segment_count": len(segments),
        "segments": segment_dicts,
        "audio_paths": audio_paths,
        "first_audio_path": audio_paths[0],
        "latest_audio_path": audio_paths[-1],
    }
