import asyncio

from fastapi.testclient import TestClient

from roboflow_livepeer_blocks.local_ingest import (
    LOCAL_AUDIO_FRAME_SENTINEL,
    LocalAudioIngestSessionManager,
    create_local_audio_ingest_app,
    parse_local_audio_ingest_source,
)


def test_parse_local_audio_ingest_source_accepts_full_url():
    info = parse_local_audio_ingest_source(
        source="ws://127.0.0.1:8765/v1/ingest/audio/test-session"
    )

    assert info["session_id"] == "test-session"
    assert info["ingest_url"] == "ws://127.0.0.1:8765/v1/ingest/audio/test-session"
    assert info["consume_url"] == "ws://127.0.0.1:8765/v1/ingest/audio/test-session/consume"
    assert info["status_url"] == "http://127.0.0.1:8765/v1/ingest/audio/test-session"


def test_parse_local_audio_ingest_source_accepts_session_id_with_default_base():
    info = parse_local_audio_ingest_source(
        source="test-session",
        default_base_url="http://local-audio-ingest:8765",
    )

    assert info["session_id"] == "test-session"
    assert info["ingest_url"] == "ws://local-audio-ingest:8765/v1/ingest/audio/test-session"


def test_local_audio_ingest_session_manager_broadcasts_and_closes():
    async def scenario():
        manager = LocalAudioIngestSessionManager()
        opened = await manager.open_publisher(
            session_id="test-session",
            source="ws://127.0.0.1:8765/v1/ingest/audio/test-session",
            source_label="pytest",
            sample_rate=16000,
            channels=1,
            sample_format="s16le",
        )
        assert opened["status"] == "open"

        consumer_id, queue, snapshot = await manager.open_consumer(session_id="test-session")
        assert snapshot["consumer_count"] == 1

        await manager.publish_audio(session_id="test-session", frame=b"\x00\x01\x02\x03")
        assert await queue.get() == b"\x00\x01\x02\x03"

        closed = await manager.close_publisher(session_id="test-session")
        assert closed["status"] == "closed"
        assert await queue.get() is LOCAL_AUDIO_FRAME_SENTINEL

        await manager.close_consumer(session_id="test-session", consumer_id=consumer_id)
        final = await manager.get_session("test-session")
        assert final["consumer_count"] == 0

    asyncio.run(scenario())


def test_local_audio_ingest_websocket_disconnect_closes_session_without_server_error():
    app = create_local_audio_ingest_app()
    client = TestClient(app)

    with client.websocket_connect("/v1/ingest/audio/test-session") as websocket:
        connected = websocket.receive_json()
        assert connected["event_type"] == "source.connected"
        websocket.send_bytes(b"\x00\x01\x02\x03")

    response = client.get("/v1/ingest/audio/test-session")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "closed"
    assert payload["total_audio_bytes"] == 4
