"""Stateful VDO.Ninja-compatible signaling bridge for stock extensions."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.websockets import WebSocketDisconnect


DEFAULT_VDO_BRIDGE_CERT_HOSTS = (
    "localhost",
    "127.0.0.1",
    "host.docker.internal",
    "vdo-signaling-bridge",
)


@dataclass
class _BridgeClient:
    websocket: WebSocket
    connection_id: str
    connected_at_epoch: float
    uuid: str
    uses_from_identity: bool = False
    stream_id: Optional[str] = None
    room_id: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "connected_at_epoch": self.connected_at_epoch,
            "uuid": self.uuid,
            "uses_from_identity": self.uses_from_identity,
            "stream_id": self.stream_id,
            "room_id": self.room_id,
        }


class VDOSignalingBridgeState:
    """Route messages using the upstream stateful VDO.Ninja handshake flow."""

    def __init__(self) -> None:
        self._clients_by_conn: Dict[str, _BridgeClient] = {}
        self._clients_by_uuid: Dict[str, _BridgeClient] = {}
        self._streams: Dict[str, str] = {}
        self._stream_ids_by_uuid: Dict[str, str] = {}
        self._callback_view: Dict[str, list[str]] = {}
        self._callback_cleanup: Dict[str, set[str]] = {}
        self._directors: Dict[str, str] = {}
        self._rooms: Dict[str, list[str]] = {}
        self._lock = asyncio.Lock()

    async def open_client(self, websocket: WebSocket) -> _BridgeClient:
        client = _BridgeClient(
            websocket=websocket,
            connection_id=f"conn_{uuid.uuid4().hex[:12]}",
            connected_at_epoch=time.time(),
            uuid=str(uuid.uuid4()),
        )
        async with self._lock:
            self._clients_by_conn[client.connection_id] = client
            self._clients_by_uuid[client.uuid] = client
        return client

    async def close_client(self, client: _BridgeClient) -> None:
        async with self._lock:
            self._cleanup_client_locked(client.uuid)

    async def process_message(self, client: _BridgeClient, payload: Dict[str, Any]) -> None:
        deliveries: list[tuple[WebSocket, Dict[str, Any]]] = []
        async with self._lock:
            self._adopt_from_identity_locked(client, payload)
            if not payload.get("request"):
                target_uuid = self._read_id(payload.get("UUID"))
                if not target_uuid:
                    await self._flush_deliveries(deliveries)
                    return
                target = self._clients_by_uuid.get(target_uuid)
                if target is not None:
                    deliveries.append(
                        (
                            target.websocket,
                            self._address_payload_for_target_locked(
                                payload,
                                sender_uuid=client.uuid,
                                target=target,
                            ),
                        )
                    )
                await self._flush_deliveries(deliveries)
                return

            request_type = str(payload.get("request") or "")
            requester = self._clients_by_uuid.get(client.uuid)
            if requester is None:
                await self._flush_deliveries(deliveries)
                return

            if request_type == "play":
                stream_id = self._read_id(payload.get("streamID"))
                if not stream_id:
                    await self._flush_deliveries(deliveries)
                    return
                if stream_id not in self._streams:
                    self._queue_for_stream_locked(client.uuid, stream_id)
                    await self._flush_deliveries(deliveries)
                    return
                seeder_uuid = self._streams.get(stream_id)
                if seeder_uuid == client.uuid:
                    await self._flush_deliveries(deliveries)
                    return
                seeder = self._clients_by_uuid.get(seeder_uuid or "")
                if seeder is None:
                    if seeder_uuid:
                        self._cleanup_client_locked(seeder_uuid)
                    self._queue_for_stream_locked(client.uuid, stream_id)
                    await self._flush_deliveries(deliveries)
                    return
                if seeder.room_id and client.room_id != seeder.room_id:
                    self._queue_for_stream_locked(client.uuid, stream_id)
                    await self._flush_deliveries(deliveries)
                    return
                deliveries.append(
                    (
                        seeder.websocket,
                        self._address_payload_for_target_locked(
                            {"request": "offerSDP"},
                            sender_uuid=client.uuid,
                            target=seeder,
                        ),
                    )
                )

            elif request_type == "seed":
                stream_id = self._read_id(payload.get("streamID"))
                if not stream_id:
                    await self._flush_deliveries(deliveries)
                    return
                existing_owner = self._streams.get(stream_id)
                if existing_owner and existing_owner != client.uuid:
                    existing_client = self._clients_by_uuid.get(existing_owner)
                    if existing_client is not None:
                        deliveries.append(
                            (
                                requester.websocket,
                                {
                                    "request": "alert",
                                    "message": "Stream ID is already in use.",
                                },
                            )
                        )
                        await self._flush_deliveries(deliveries)
                        return
                    self._cleanup_client_locked(existing_owner)
                self._assign_stream_locked(client, stream_id)
                if client.room_id:
                    self._notify_room_locked(
                        client.room_id,
                        {"request": "videoaddedtoroom", "streamID": stream_id},
                        deliveries,
                        sender_uuid=client.uuid,
                        skip={client.uuid},
                    )
                elif stream_id in self._callback_view:
                    waiting_viewers = list(self._callback_view.pop(stream_id))
                    for viewer_uuid in waiting_viewers:
                        cleanup = self._callback_cleanup.get(viewer_uuid)
                        if cleanup is not None:
                            cleanup.discard(stream_id)
                        if not cleanup:
                            self._callback_cleanup.pop(viewer_uuid, None)
                        deliveries.append(
                            (
                                requester.websocket,
                                self._address_payload_for_target_locked(
                                    {"request": "offerSDP"},
                                    sender_uuid=viewer_uuid,
                                    target=requester,
                                ),
                            )
                        )

            elif request_type == "joinroom":
                room_id = self._read_id(payload.get("roomid")).lower()
                if not room_id or client.room_id:
                    await self._flush_deliveries(deliveries)
                    return
                client.room_id = room_id
                stream_id = self._read_id(payload.get("streamID"))
                if stream_id:
                    existing_owner = self._streams.get(stream_id)
                    if existing_owner and existing_owner != client.uuid:
                        existing_client = self._clients_by_uuid.get(existing_owner)
                        if existing_client is not None:
                            deliveries.append(
                                (
                                    requester.websocket,
                                    {
                                        "request": "alert",
                                        "message": "Stream ID is already in use.",
                                    },
                                )
                            )
                            await self._flush_deliveries(deliveries)
                            return
                        self._cleanup_client_locked(existing_owner)
                    self._assign_stream_locked(client, stream_id)
                is_director = False
                response: Dict[str, Any] = {"request": "listing", "list": []}
                if payload.get("claim"):
                    current_director = self._directors.get(room_id)
                    if not current_director or current_director not in self._clients_by_uuid:
                        self._directors[room_id] = client.uuid
                        response["claim"] = True
                        is_director = True
                    else:
                        response["claim"] = current_director == client.uuid
                        if not response["claim"]:
                            response["director"] = current_director
                elif room_id in self._directors:
                    response["director"] = self._directors[room_id]
                members = self._rooms.setdefault(room_id, [])
                for member_uuid in members:
                    entry = {"UUID": member_uuid}
                    if member_uuid in self._stream_ids_by_uuid:
                        entry["streamID"] = self._stream_ids_by_uuid[member_uuid]
                    response["list"].append(entry)
                deliveries.append((requester.websocket, response))
                joined_notice: Dict[str, Any] = {"request": "someonejoined", "UUID": client.uuid}
                if is_director:
                    joined_notice["director"] = True
                if client.uuid in self._stream_ids_by_uuid:
                    joined_notice["streamID"] = self._stream_ids_by_uuid[client.uuid]
                self._notify_room_locked(
                    room_id,
                    joined_notice,
                    deliveries,
                    sender_uuid=client.uuid,
                    skip={client.uuid},
                )
                if client.uuid not in members:
                    members.append(client.uuid)

            elif request_type == "migrate":
                target_uuid = self._read_id(payload.get("target"))
                destination = self._read_id(payload.get("roomid")).lower()
                director_room = client.room_id
                if (
                    not target_uuid
                    or not destination
                    or not director_room
                    or self._directors.get(director_room) != client.uuid
                ):
                    await self._flush_deliveries(deliveries)
                    return
                target_client = self._clients_by_uuid.get(target_uuid)
                if target_client is None or target_client.room_id != director_room or target_uuid == client.uuid:
                    await self._flush_deliveries(deliveries)
                    return
                source_members = self._rooms.get(director_room)
                if not source_members or target_uuid not in source_members:
                    await self._flush_deliveries(deliveries)
                    return
                source_members.remove(target_uuid)
                if not source_members:
                    self._rooms.pop(director_room, None)
                target_client.room_id = destination
                dest_members = self._rooms.setdefault(destination, [])
                transferred: Dict[str, Any] = {"request": "transferred", "list": []}
                if destination in self._directors:
                    transferred["director"] = self._directors[destination]
                for member_uuid in dest_members:
                    entry = {"UUID": member_uuid}
                    if member_uuid in self._stream_ids_by_uuid:
                        entry["streamID"] = self._stream_ids_by_uuid[member_uuid]
                    transferred["list"].append(entry)
                deliveries.append((target_client.websocket, transferred))
                joined_notice = {"request": "someonejoined", "UUID": target_uuid}
                target_stream = self._stream_ids_by_uuid.get(target_uuid)
                if target_stream:
                    joined_notice["streamID"] = target_stream
                self._notify_room_locked(
                    destination,
                    joined_notice,
                    deliveries,
                    sender_uuid=target_uuid,
                    skip={target_uuid},
                )
                if target_uuid not in dest_members:
                    dest_members.append(target_uuid)

        await self._flush_deliveries(deliveries)

    async def snapshot(self) -> Dict[str, Any]:
        async with self._lock:
            clients = [client.as_dict() for client in self._clients_by_conn.values()]
            rooms = {room_id: list(members) for room_id, members in sorted(self._rooms.items())}
            streams = dict(sorted(self._streams.items()))
            waiting = {
                stream_id: list(viewers)
                for stream_id, viewers in sorted(self._callback_view.items())
            }
            directors = dict(sorted(self._directors.items()))
        return {
            "status": "ok",
            "client_count": len(clients),
            "stream_count": len(streams),
            "room_count": len(rooms),
            "clients": clients,
            "streams": streams,
            "rooms": rooms,
            "directors": directors,
            "waiting_viewers": waiting,
        }

    def _assign_stream_locked(self, client: _BridgeClient, stream_id: str) -> None:
        previous_stream = self._stream_ids_by_uuid.get(client.uuid)
        if previous_stream and previous_stream != stream_id:
            self._streams.pop(previous_stream, None)
        client.stream_id = stream_id
        self._streams[stream_id] = client.uuid
        self._stream_ids_by_uuid[client.uuid] = stream_id

    def _cleanup_client_locked(self, client_uuid: str) -> None:
        client = self._clients_by_uuid.pop(client_uuid, None)
        if client is None:
            return
        self._clients_by_conn.pop(client.connection_id, None)
        stream_id = self._stream_ids_by_uuid.pop(client_uuid, None)
        if stream_id and self._streams.get(stream_id) == client_uuid:
            self._streams.pop(stream_id, None)
        if client.room_id:
            room_id = client.room_id
            if self._directors.get(room_id) == client_uuid:
                self._directors.pop(room_id, None)
            members = self._rooms.get(room_id)
            if members and client_uuid in members:
                members.remove(client_uuid)
                if not members:
                    self._rooms.pop(room_id, None)
        self._remove_from_callback_locked(client_uuid)

    def _adopt_from_identity_locked(self, client: _BridgeClient, payload: Dict[str, Any]) -> None:
        requested_uuid = self._read_id(payload.get("from"))
        if not requested_uuid:
            return
        client.uses_from_identity = True
        if requested_uuid == client.uuid:
            return
        existing = self._clients_by_uuid.get(requested_uuid)
        if existing is not None and existing is not client:
            return
        old_uuid = client.uuid
        self._clients_by_uuid.pop(old_uuid, None)
        client.uuid = requested_uuid
        self._clients_by_uuid[client.uuid] = client

        stream_id = self._stream_ids_by_uuid.pop(old_uuid, None)
        if stream_id:
            self._stream_ids_by_uuid[client.uuid] = stream_id
            if self._streams.get(stream_id) == old_uuid:
                self._streams[stream_id] = client.uuid
        if client.room_id:
            members = self._rooms.get(client.room_id)
            if members:
                self._replace_member_locked(members, old_uuid, client.uuid)
            if self._directors.get(client.room_id) == old_uuid:
                self._directors[client.room_id] = client.uuid
        self._rename_callback_client_locked(old_uuid, client.uuid)

    def _replace_member_locked(self, members: list[str], old_uuid: str, new_uuid: str) -> None:
        try:
            index = members.index(old_uuid)
        except ValueError:
            if new_uuid not in members:
                members.append(new_uuid)
            return
        members[index] = new_uuid

    def _rename_callback_client_locked(self, old_uuid: str, new_uuid: str) -> None:
        pending = self._callback_cleanup.pop(old_uuid, None)
        if pending:
            cleanup = self._callback_cleanup.setdefault(new_uuid, set())
            cleanup.update(pending)
            for stream_id in pending:
                waiting = self._callback_view.get(stream_id)
                if waiting:
                    self._replace_member_locked(waiting, old_uuid, new_uuid)

    def _address_payload_for_target_locked(
        self,
        payload: Dict[str, Any],
        *,
        sender_uuid: str,
        target: _BridgeClient,
    ) -> Dict[str, Any]:
        forwarded = dict(payload)
        if target.uses_from_identity:
            forwarded["UUID"] = target.uuid
            forwarded["from"] = sender_uuid
        else:
            forwarded.pop("from", None)
            forwarded["UUID"] = sender_uuid
        return forwarded

    def _queue_for_stream_locked(self, viewer_uuid: str, stream_id: str) -> None:
        waiting = self._callback_view.setdefault(stream_id, [])
        if viewer_uuid not in waiting:
            waiting.append(viewer_uuid)
        cleanup = self._callback_cleanup.setdefault(viewer_uuid, set())
        cleanup.add(stream_id)

    def _remove_from_callback_locked(self, client_uuid: str) -> None:
        pending = self._callback_cleanup.pop(client_uuid, None)
        if not pending:
            return
        for stream_id in pending:
            waiting = self._callback_view.get(stream_id)
            if not waiting:
                continue
            try:
                waiting.remove(client_uuid)
            except ValueError:
                pass
            if not waiting:
                self._callback_view.pop(stream_id, None)

    def _notify_room_locked(
        self,
        room_id: str,
        payload: Dict[str, Any],
        deliveries: list[tuple[WebSocket, Dict[str, Any]]],
        *,
        sender_uuid: str,
        skip: Optional[set[str]] = None,
    ) -> None:
        members = self._rooms.get(room_id)
        if not members:
            return
        for member_uuid in members:
            if skip and member_uuid in skip:
                continue
            target = self._clients_by_uuid.get(member_uuid)
            if target is not None:
                deliveries.append(
                    (
                        target.websocket,
                        self._address_payload_for_target_locked(
                            payload,
                            sender_uuid=sender_uuid,
                            target=target,
                        ),
                    )
                )

    @staticmethod
    def _read_id(value: Any) -> str:
        return value.strip() if isinstance(value, str) else ""

    @staticmethod
    async def _send_json(websocket: WebSocket, payload: Dict[str, Any]) -> None:
        await websocket.send_text(json.dumps(payload))

    async def _flush_deliveries(
        self, deliveries: Iterable[tuple[WebSocket, Dict[str, Any]]]
    ) -> None:
        for websocket, payload in deliveries:
            await self._send_json(websocket, payload)


def ensure_vdo_bridge_certificate(
    *,
    cert_path: str | Path,
    key_path: str | Path,
    hosts: Iterable[str] = DEFAULT_VDO_BRIDGE_CERT_HOSTS,
) -> tuple[Path, Path]:
    cert_file = Path(cert_path)
    key_file = Path(key_path)
    if cert_file.exists() and key_file.exists():
        return cert_file, key_file
    cert_file.parent.mkdir(parents=True, exist_ok=True)
    key_file.parent.mkdir(parents=True, exist_ok=True)
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Livepeer Roboflow"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ]
    )
    san_entries = []
    for host in hosts:
        host_value = str(host).strip()
        if not host_value:
            continue
        try:
            san_entries.append(x509.IPAddress(ipaddress.ip_address(host_value)))
        except ValueError:
            san_entries.append(x509.DNSName(host_value))
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc) - timedelta(hours=1))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .sign(private_key, hashes.SHA256())
    )
    key_file.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    cert_file.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    return cert_file, key_file


def create_vdo_signaling_bridge_app(
    *, state: Optional[VDOSignalingBridgeState] = None
) -> FastAPI:
    bridge = state or VDOSignalingBridgeState()
    app = FastAPI(title="VDO Signaling Bridge")

    @app.get("/")
    async def root() -> HTMLResponse:
        return HTMLResponse(
            "<html><body><h1>VDO Signaling Bridge</h1>"
            "<p>Service is running. For local browser use, open this page once and accept the "
            "certificate warning if prompted, then use the same host in the extension custom server.</p>"
            "</body></html>"
        )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/statusz")
    async def statusz() -> Dict[str, Any]:
        return await bridge.snapshot()

    @app.websocket("/")
    async def bridge_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        client = await bridge.open_client(websocket)
        await websocket.send_text(json.dumps({"id": client.uuid}))
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                text_payload = message.get("text")
                if not text_payload:
                    continue
                try:
                    payload = json.loads(text_payload)
                except json.JSONDecodeError:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "request": "error",
                                "message": "invalid json",
                                "code": "INVALID_JSON",
                            }
                        )
                    )
                    continue
                if not isinstance(payload, dict):
                    continue
                await bridge.process_message(client, payload)
        except WebSocketDisconnect:
            pass
        finally:
            await bridge.close_client(client)
            try:
                await websocket.close()
            except RuntimeError:
                pass

    return app


app = create_vdo_signaling_bridge_app()
