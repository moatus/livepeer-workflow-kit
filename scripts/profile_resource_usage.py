#!/usr/bin/env python3
"""Sample host, GPU, and Docker resource usage for a bounded profiling window."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


GPU_QUERY_FIELDS = [
    "index",
    "name",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "power.draw",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--duration-seconds", type=float, default=600.0)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument(
        "--container-substring",
        action="append",
        default=[],
        help="Collect docker stats for running containers whose names contain this substring.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "resource-samples.csv"
    containers_path = output_dir / "container-stats.jsonl"
    summary_path = output_dir / "resource-summary.json"

    container_substrings = args.container_substring or [
        "audio-diarized-transcription-runner",
        "livepeer-poc",
    ]

    cpu_prev = _read_cpu_times()
    started = time.monotonic()
    rows: List[Dict[str, Any]] = []
    container_rows: List[Dict[str, Any]] = []

    fieldnames = [
        "timestamp_iso",
        "elapsed_seconds",
        "host_cpu_pct",
        "host_mem_used_mib",
        "host_mem_total_mib",
        "gpu_index",
        "gpu_name",
        "gpu_util_pct",
        "gpu_mem_used_mib",
        "gpu_mem_total_mib",
        "gpu_power_w",
        "container_count",
        "container_cpu_pct_sum",
        "container_mem_used_mib_sum",
    ]

    with samples_path.open("w", newline="", encoding="utf-8") as samples_file, containers_path.open(
        "w", encoding="utf-8"
    ) as containers_file:
        writer = csv.DictWriter(samples_file, fieldnames=fieldnames)
        writer.writeheader()
        while True:
            now_monotonic = time.monotonic()
            elapsed = now_monotonic - started
            if elapsed > args.duration_seconds:
                break

            cpu_now = _read_cpu_times()
            host_cpu_pct = _cpu_pct(cpu_prev, cpu_now)
            cpu_prev = cpu_now
            mem = _read_meminfo()
            gpus = _read_gpu_stats()
            containers = _read_matching_container_stats(container_substrings)
            for container in containers:
                containers_file.write(
                    json.dumps(
                        {
                            "timestamp_iso": _utc_now(),
                            "elapsed_seconds": round(elapsed, 3),
                            "container": container,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                container_rows.append(container)

            container_cpu_sum = round(sum(_pct(container.get("CPUPerc")) for container in containers), 3)
            container_mem_sum = round(
                sum(_memory_usage_mib(str(container.get("MemUsage", ""))) for container in containers),
                3,
            )
            base_row = {
                "timestamp_iso": _utc_now(),
                "elapsed_seconds": round(elapsed, 3),
                "host_cpu_pct": round(host_cpu_pct, 3) if host_cpu_pct is not None else "",
                "host_mem_used_mib": round(mem["used_mib"], 3),
                "host_mem_total_mib": round(mem["total_mib"], 3),
                "container_count": len(containers),
                "container_cpu_pct_sum": container_cpu_sum,
                "container_mem_used_mib_sum": container_mem_sum,
            }
            if not gpus:
                row = {**base_row, **_empty_gpu_row()}
                writer.writerow(row)
                rows.append(row)
            for gpu in gpus:
                row = {
                    **base_row,
                    "gpu_index": gpu.get("index", ""),
                    "gpu_name": gpu.get("name", ""),
                    "gpu_util_pct": gpu.get("utilization.gpu [%]", ""),
                    "gpu_mem_used_mib": gpu.get("memory.used [MiB]", ""),
                    "gpu_mem_total_mib": gpu.get("memory.total [MiB]", ""),
                    "gpu_power_w": gpu.get("power.draw [W]", ""),
                }
                writer.writerow(row)
                rows.append(row)
            samples_file.flush()
            containers_file.flush()
            sleep_for = args.interval_seconds - (time.monotonic() - now_monotonic)
            if sleep_for > 0:
                time.sleep(sleep_for)

    summary = _summarize(rows, container_rows, container_substrings, args.duration_seconds)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(command: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _read_gpu_stats() -> List[Dict[str, Any]]:
    result = _run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
            "--format=csv",
        ]
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    reader = csv.DictReader(lines)
    return [_numeric_gpu_row(row) for row in reader]


def _numeric_gpu_row(row: Dict[str, str]) -> Dict[str, Any]:
    converted: Dict[str, Any] = {}
    for key, value in row.items():
        key = key.strip()
        clean = value.strip()
        if key == "name":
            converted[key] = clean
        elif key == "index":
            converted[key] = clean
        else:
            converted[key] = _float_prefix(clean)
    return converted


def _read_matching_container_stats(substrings: Iterable[str]) -> List[Dict[str, Any]]:
    names = _running_container_names()
    selected = [name for name in names if any(part in name for part in substrings)]
    rows = []
    for name in selected:
        result = _run(["docker", "stats", "--no-stream", "--format", "{{json .}}", name])
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _running_container_names() -> List[str]:
    result = _run(["docker", "ps", "--format", "{{.Names}}"])
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _read_cpu_times() -> Optional[List[int]]:
    try:
        first = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
    except OSError:
        return None
    parts = first.split()
    if not parts or parts[0] != "cpu":
        return None
    return [int(part) for part in parts[1:]]


def _cpu_pct(prev: Optional[List[int]], cur: Optional[List[int]]) -> Optional[float]:
    if prev is None or cur is None:
        return None
    total_prev = sum(prev)
    total_cur = sum(cur)
    idle_prev = prev[3] + (prev[4] if len(prev) > 4 else 0)
    idle_cur = cur[3] + (cur[4] if len(cur) > 4 else 0)
    total_delta = total_cur - total_prev
    idle_delta = idle_cur - idle_prev
    if total_delta <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * (1.0 - (idle_delta / total_delta))))


def _read_meminfo() -> Dict[str, float]:
    values: Dict[str, float] = {}
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return {"total_mib": 0.0, "used_mib": 0.0}
    for line in lines:
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        amount = rest.strip().split()[0]
        try:
            values[key] = float(amount) / 1024.0
        except ValueError:
            continue
    total = values.get("MemTotal", 0.0)
    available = values.get("MemAvailable", 0.0)
    return {"total_mib": total, "used_mib": max(0.0, total - available)}


def _empty_gpu_row() -> Dict[str, str]:
    return {
        "gpu_index": "",
        "gpu_name": "",
        "gpu_util_pct": "",
        "gpu_mem_used_mib": "",
        "gpu_mem_total_mib": "",
        "gpu_power_w": "",
    }


def _float_prefix(value: str) -> Any:
    token = value.strip().split()[0] if value.strip() else ""
    if token.upper() in {"N/A", "[N/A]"}:
        return ""
    try:
        return float(token)
    except ValueError:
        return value


def _pct(value: Any) -> float:
    try:
        return float(str(value).strip().rstrip("%"))
    except ValueError:
        return 0.0


def _memory_usage_mib(value: str) -> float:
    first = value.split("/", 1)[0].strip()
    if not first:
        return 0.0
    return _memory_to_mib(first)


def _memory_to_mib(value: str) -> float:
    units = [
        ("GiB", 1024.0),
        ("MiB", 1.0),
        ("KiB", 1.0 / 1024.0),
        ("GB", 1000.0),
        ("MB", 1000.0 / 1024.0),
        ("KB", 1000.0 / (1024.0 * 1024.0)),
        ("B", 1.0 / (1024.0 * 1024.0)),
    ]
    for suffix, multiplier in units:
        if value.endswith(suffix):
            return _safe_float(value[: -len(suffix)].strip()) * multiplier
    return _safe_float(value)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _summarize(
    rows: List[Dict[str, Any]],
    container_rows: List[Dict[str, Any]],
    container_substrings: List[str],
    requested_duration_seconds: float,
) -> Dict[str, Any]:
    gpu_indices = sorted({str(row.get("gpu_index")) for row in rows if str(row.get("gpu_index"))})
    gpu_summary = {
        index: {
            "name": next(
                str(row.get("gpu_name"))
                for row in rows
                if str(row.get("gpu_index")) == index and row.get("gpu_name")
            ),
            "util_pct": _stats(_numbers(row.get("gpu_util_pct") for row in rows if str(row.get("gpu_index")) == index)),
            "mem_used_mib": _stats(
                _numbers(row.get("gpu_mem_used_mib") for row in rows if str(row.get("gpu_index")) == index)
            ),
            "power_w": _stats(_numbers(row.get("gpu_power_w") for row in rows if str(row.get("gpu_index")) == index)),
        }
        for index in gpu_indices
    }
    return {
        "created_at_utc": _utc_now(),
        "requested_duration_seconds": requested_duration_seconds,
        "sample_count": len(rows),
        "observed_elapsed_seconds": max(_numbers(row.get("elapsed_seconds") for row in rows), default=0.0),
        "container_name_substrings": container_substrings,
        "host_cpu_pct": _stats(_numbers(row.get("host_cpu_pct") for row in rows)),
        "host_mem_used_mib": _stats(_numbers(row.get("host_mem_used_mib") for row in rows)),
        "container_cpu_pct_sum": _stats(_numbers(row.get("container_cpu_pct_sum") for row in rows)),
        "container_mem_used_mib_sum": _stats(_numbers(row.get("container_mem_used_mib_sum") for row in rows)),
        "container_samples": len(container_rows),
        "container_names_seen": sorted({str(row.get("Name") or row.get("Name", "")) for row in container_rows}),
        "gpu": gpu_summary,
    }


def _numbers(values: Iterable[Any]) -> List[float]:
    numbers = []
    for value in values:
        if value == "" or value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            numbers.append(number)
    return numbers


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "avg": round(sum(ordered) / len(ordered), 3),
        "p50": round(_percentile(ordered, 50), 3),
        "p95": round(_percentile(ordered, 95), 3),
        "max": round(max(ordered), 3),
    }


def _percentile(ordered: List[float], percentile: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


if __name__ == "__main__":
    raise SystemExit(main())
