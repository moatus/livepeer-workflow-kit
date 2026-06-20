"""Provider adapters for dynamic Livepeer module discovery.

Provider profiles are gateway adapters, not hardcoded model catalogs. The
Livepeer Modules adapter discovers capabilities from Cloudspe/Open
Clearinghouse at runtime and returns workflow parameters for selected
capabilities.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .config import (
    DEFAULT_OPEN_CLEARINGHOUSE_URL,
    OPEN_CLEARINGHOUSE_API_KEY_ENV,
    OPEN_CLEARINGHOUSE_URL_ENV,
)


class ProviderConfigurationError(RuntimeError):
    """Raised when a provider cannot be used due to missing configuration."""


class ProviderDiscoveryError(RuntimeError):
    """Raised when provider discovery fails."""


@dataclass(frozen=True)
class CapabilityOffering:
    capability: str
    offering: str
    work_unit: str
    price_per_work_unit_wei: str
    extra: Mapping[str, Any]

    @property
    def interaction_mode(self) -> str:
        return str(self.extra.get("interaction_mode") or "")


class CapabilityCatalog:
    """Thin normalized view over provider-specific capability payloads."""

    def __init__(self, *, provider_name: str, raw: Mapping[str, Any]) -> None:
        self.provider_name = provider_name
        self.raw = raw
        self._offerings = self._parse_offerings(raw)

    @staticmethod
    def _parse_offerings(raw: Mapping[str, Any]) -> List[CapabilityOffering]:
        offerings: List[CapabilityOffering] = []
        for item in raw.get("items", []):
            capability = str(item.get("name") or "")
            for offering in item.get("offerings", []):
                offerings.append(
                    CapabilityOffering(
                        capability=capability,
                        offering=str(offering.get("id") or ""),
                        work_unit=str(offering.get("work_unit") or item.get("work_unit") or ""),
                        price_per_work_unit_wei=str(
                            offering.get("price_per_work_unit_wei") or ""
                        ),
                        extra=offering.get("extra") or {},
                    )
                )
        return offerings

    @property
    def offerings(self) -> List[CapabilityOffering]:
        return list(self._offerings)

    def find(
        self,
        *,
        capability: str,
        offering: Optional[str] = None,
        interaction_mode: Optional[str] = None,
        task: Optional[str] = None,
    ) -> CapabilityOffering:
        matches = [
            item
            for item in self._offerings
            if item.capability == capability
            and (offering is None or item.offering == offering)
            and (interaction_mode is None or item.interaction_mode == interaction_mode)
            and (task is None or _contains_task(item.extra, task))
        ]
        if not matches:
            available = ", ".join(
                f"{item.capability}/{item.offering}({item.interaction_mode})"
                for item in self._offerings
                if item.capability == capability
            )
            raise ProviderDiscoveryError(
                f"No provider offering matched capability={capability!r}, "
                f"offering={offering!r}, interaction_mode={interaction_mode!r}, "
                f"task={task!r}. Available for capability: {available or 'none'}"
            )
        return matches[0]


def _contains_task(extra: Mapping[str, Any], task: str) -> bool:
    wanted = task.lower()
    stack: List[Any] = [extra]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
        elif isinstance(value, str) and value.lower() == wanted:
            return True
    return False


class LivepeerModulesProvider:
    """Cloudspe-backed Livepeer Modules provider adapter."""

    name = "livepeer-modules"

    def __init__(self, *, base_url: str, api_key: str, timeout_seconds: float = 20.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> "LivepeerModulesProvider":
        base_url = os.getenv(OPEN_CLEARINGHOUSE_URL_ENV, DEFAULT_OPEN_CLEARINGHOUSE_URL)
        api_key = os.getenv(OPEN_CLEARINGHOUSE_API_KEY_ENV)
        if not api_key:
            raise ProviderConfigurationError(
                f"Livepeer Modules provider requires {OPEN_CLEARINGHOUSE_API_KEY_ENV}. "
                "Set it to use Cloudspe/Livepeer, or explicitly choose a self-hosted/local "
                "provider mode for development."
            )
        return cls(base_url=base_url, api_key=api_key)

    def _get_json(self, path: str, query: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
        url = self.base_url + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        request = urllib.request.Request(
            url,
            headers={"X-API-Key": self.api_key, "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ProviderDiscoveryError(
                f"Livepeer Modules API request failed: GET {path} returned "
                f"HTTP {exc.code}: {body[:500]}"
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ProviderDiscoveryError(
                f"Livepeer Modules API request failed: GET {path}: {exc}"
            ) from exc

    def discover(self) -> CapabilityCatalog:
        return CapabilityCatalog(provider_name=self.name, raw=self._get_json("/v1/capabilities"))

    def route(self, offering: CapabilityOffering) -> Dict[str, Any]:
        return self._get_json(
            "/v1/routes",
            {"capability": offering.capability, "offering": offering.offering},
        )

    def require_route(self, offering: CapabilityOffering) -> Dict[str, Any]:
        route = self.route(offering)
        if not route.get("worker_url"):
            raise ProviderDiscoveryError(
                f"Livepeer Modules route for {offering.capability}/{offering.offering} "
                "did not include a worker_url."
            )
        return route

    def choose_audio_transcription(self, *, streaming: bool = False) -> CapabilityOffering:
        catalog = self.discover()
        if streaming:
            return catalog.find(
                capability="openai:audio-transcriptions",
                offering="nemo-meeting-stream",
                interaction_mode="ws-realtime@v0",
                task="transcription",
            )
        return catalog.find(
            capability="openai:audio-transcriptions",
            offering="nemo-meeting",
            interaction_mode="http-multipart@v0",
            task="transcription",
        )

    def choose_screen_vision(self) -> CapabilityOffering:
        return self.discover().find(
            capability="openai:vision",
            offering="florence-2-large",
            interaction_mode="http-reqresp@v0",
            task="analyze",
        )

    def transcription_block_params(self, offering: CapabilityOffering) -> Dict[str, Any]:
        return {
            "transcription_backend": "livepeer_remote_http",
            "livepeer_capability": offering.capability,
            "livepeer_offering": offering.offering,
        }

    def vision_block_params(self, offering: CapabilityOffering) -> Dict[str, Any]:
        return {
            "vision_backend": "livepeer_remote",
            "florence2_runner_url": "",
            "livepeer_capability": offering.capability,
            "livepeer_offering": offering.offering,
            "model_id": offering.offering,
        }


def require_livepeer_modules_provider() -> LivepeerModulesProvider:
    """Return the default Livepeer provider or fail with an actionable message."""
    return LivepeerModulesProvider.from_env()


__all__ = [
    "CapabilityCatalog",
    "CapabilityOffering",
    "LivepeerModulesProvider",
    "ProviderConfigurationError",
    "ProviderDiscoveryError",
    "require_livepeer_modules_provider",
]
