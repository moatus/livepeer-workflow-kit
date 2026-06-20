from pathlib import Path

from roboflow_livepeer_blocks.audio import materialize_audio_chunks, plan_audio_chunks


def test_plan_audio_chunks_defaults_to_ten_second_windows():
    chunks = plan_audio_chunks(
        audio_path=Path("meeting.mp3"),
        chunk_size_seconds=10,
        duration_seconds=25.2,
    )

    assert [chunk.index for chunk in chunks] == [0, 1, 2]
    assert [chunk.start_seconds for chunk in chunks] == [0, 10, 20]
    assert [chunk.end_seconds for chunk in chunks] == [10, 20, 25.2]
    assert [chunk.duration_seconds for chunk in chunks] == [10, 10, 5.199999999999999]


def test_plan_audio_chunks_rejects_non_positive_chunk_size():
    try:
        plan_audio_chunks(
            audio_path=Path("meeting.mp3"),
            chunk_size_seconds=0,
            duration_seconds=25.2,
        )
    except ValueError as error:
        assert "chunk_size_seconds" in str(error)
    else:
        raise AssertionError("Expected ValueError")


def test_materialize_audio_chunks_uses_ffmpeg_for_multi_chunk_audio(tmp_path, monkeypatch):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"audio")
    commands = []

    def fake_probe(_audio_path):
        return 20.1

    def fake_run(command, check, capture_output):
        commands.append((command, check, capture_output))

    monkeypatch.setattr("roboflow_livepeer_blocks.audio.probe_audio_duration_seconds", fake_probe)
    monkeypatch.setattr("roboflow_livepeer_blocks.audio.subprocess.run", fake_run)

    chunks = materialize_audio_chunks(
        audio_path=source,
        output_dir=tmp_path / "chunks",
        chunk_size_seconds=10,
    )

    assert [chunk.path.name for chunk in chunks] == [
        "meeting.chunk-0000.mp3",
        "meeting.chunk-0001.mp3",
        "meeting.chunk-0002.mp3",
    ]
    assert [command[0][0] for command in commands] == ["ffmpeg", "ffmpeg", "ffmpeg"]
    assert commands[0][0][commands[0][0].index("-ss") + 1] == "0.000000"
    assert commands[1][0][commands[1][0].index("-ss") + 1] == "10.000000"
    assert commands[2][0][commands[2][0].index("-t") + 1] == "0.100000"
