from fastapi.testclient import TestClient

from roboflow_livepeer_blocks.vdo_signaling_bridge import create_vdo_signaling_bridge_app


def _receive_assigned_uuid(socket) -> str:
    payload = socket.receive_json()
    assert "id" in payload
    return payload["id"]


def test_vdo_signaling_bridge_routes_seed_play_and_peer_messages():
    client = TestClient(create_vdo_signaling_bridge_app())

    with client.websocket_connect("/") as publisher:
        publisher_uuid = _receive_assigned_uuid(publisher)
        publisher.send_json({"request": "seed", "streamID": "stream_123"})

        with client.websocket_connect("/") as viewer:
            viewer_uuid = _receive_assigned_uuid(viewer)
            viewer.send_json({"request": "play", "streamID": "stream_123"})
            offer_request = publisher.receive_json()
            assert offer_request == {"request": "offerSDP", "UUID": viewer_uuid}

            publisher.send_json({"UUID": viewer_uuid, "description": {"type": "offer"}})
            offer = viewer.receive_json()
            assert offer["UUID"] == publisher_uuid
            assert offer["description"]["type"] == "offer"


def test_vdo_signaling_bridge_lists_room_members_and_announces_new_seed():
    client = TestClient(create_vdo_signaling_bridge_app())

    with client.websocket_connect("/") as first:
        first_uuid = _receive_assigned_uuid(first)
        first.send_json({"request": "joinroom", "roomid": "room_a"})
        listing = first.receive_json()
        assert listing["request"] == "listing"
        assert listing["list"] == []

        first.send_json({"request": "seed", "streamID": "stream_123"})

        with client.websocket_connect("/") as second:
            second_uuid = _receive_assigned_uuid(second)
            second.send_json({"request": "joinroom", "roomid": "room_a"})
            second_listing = second.receive_json()
            assert second_listing["request"] == "listing"
            assert second_listing["list"] == [{"UUID": first_uuid, "streamID": "stream_123"}]
            joined = first.receive_json()
            assert joined["request"] == "someonejoined"
            assert joined["UUID"] == second_uuid

            second.send_json({"request": "seed", "streamID": "stream_456"})
            added = first.receive_json()
            assert added["request"] == "videoaddedtoroom"
            assert added["UUID"] == second_uuid
            assert added["streamID"] == "stream_456"


def test_vdo_signaling_bridge_tracks_joinroom_stream_id_for_browser_publishers():
    client = TestClient(create_vdo_signaling_bridge_app())

    with client.websocket_connect("/") as publisher:
        publisher_uuid = _receive_assigned_uuid(publisher)
        publisher.send_json(
            {"request": "joinroom", "roomid": "room_a", "streamID": "stream_room"}
        )
        listing = publisher.receive_json()
        assert listing["request"] == "listing"

        status = client.get("/statusz").json()
        assert status["streams"] == {"stream_room": publisher_uuid}
        assert status["clients"][0]["stream_id"] == "stream_room"

        with client.websocket_connect("/") as viewer:
            viewer_uuid = _receive_assigned_uuid(viewer)
            viewer.send_json({"request": "joinroom", "roomid": "room_a"})
            viewer_listing = viewer.receive_json()
            assert viewer_listing["request"] == "listing"
            assert viewer_listing["list"] == [
                {"UUID": publisher_uuid, "streamID": "stream_room"}
            ]
            joined = publisher.receive_json()
            assert joined["request"] == "someonejoined"
            assert joined["UUID"] == viewer_uuid
            viewer.send_json({"request": "play", "streamID": "stream_room"})
            assert publisher.receive_json() == {
                "request": "offerSDP",
                "UUID": viewer_uuid,
            }


def test_vdo_signaling_bridge_queues_viewers_until_stream_is_seeded():
    client = TestClient(create_vdo_signaling_bridge_app())

    with client.websocket_connect("/") as viewer:
        viewer_uuid = _receive_assigned_uuid(viewer)
        viewer.send_json({"request": "play", "streamID": "stream_queued"})

        with client.websocket_connect("/") as publisher:
            _receive_assigned_uuid(publisher)
            publisher.send_json({"request": "seed", "streamID": "stream_queued"})
            offer_request = publisher.receive_json()
            assert offer_request == {"request": "offerSDP", "UUID": viewer_uuid}


def test_vdo_signaling_bridge_preserves_from_identity_for_puuid_clients():
    client = TestClient(create_vdo_signaling_bridge_app())

    with client.websocket_connect("/") as publisher:
        publisher_uuid = _receive_assigned_uuid(publisher)
        publisher.send_json({"request": "seed", "streamID": "rf_local_test_2808d64"})

        with client.websocket_connect("/") as viewer:
            _receive_assigned_uuid(viewer)
            viewer.send_json(
                {
                    "request": "play",
                    "streamID": "rf_local_test_2808d64",
                    "from": "raspberry-puuid",
                }
            )
            offer_request = publisher.receive_json()
            assert offer_request == {"request": "offerSDP", "UUID": "raspberry-puuid"}

            publisher.send_json(
                {
                    "UUID": "raspberry-puuid",
                    "description": {"type": "offer"},
                    "session": "session-1",
                }
            )
            offer = viewer.receive_json()
            assert offer["UUID"] == "raspberry-puuid"
            assert offer["from"] == publisher_uuid
            assert offer["description"]["type"] == "offer"

            viewer.send_json(
                {
                    "UUID": publisher_uuid,
                    "from": "raspberry-puuid",
                    "description": {"type": "answer"},
                    "session": "session-1",
                }
            )
            answer = publisher.receive_json()
            assert answer["UUID"] == "raspberry-puuid"
            assert answer["description"]["type"] == "answer"
