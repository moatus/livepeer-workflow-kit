# Raw Block Catalog

This catalog gives block contracts for composing Livepeer Roboflow workflow JSON. It is a block-level reference, not a complete workflow recipe.

## Workflow Input

Use a workflow parameter named `source` when the media source should be supplied at runtime:

```json
{"type": "WorkflowParameter", "name": "source"}
```

Reference it inside steps as:

```json
"$inputs.source"
```

## VDO Media Source

Block type:

```text
roboflow_livepeer_blocks/livepeer_vdo_ninja_media_source@v1
```

Purpose: connect to a VDO/WebRTC publisher and produce audio/video source descriptors.

Common parameters:

- `source`: explicit stream ID/view URL or `$inputs.source`
- `signaling_server`: WSS/WS bridge URL visible from the runtime environment
- `password`: optional
- `audio_enabled`: boolean
- `video_enabled`: boolean
- `audio_sample_rate`: usually `48000`
- `audio_channels`: usually `1`
- `video_frame_rate`: sampling hint, for example `0.2`
- `buffer_ms`: small receive buffer, for example `300`

Outputs:

- `audio_source_descriptor`
- `video_source_descriptor`

## PCM16 Audio Transform

Block type:

```text
roboflow_livepeer_blocks/livepeer_pcm16_audio_transform@v1
```

Purpose: normalize media audio into PCM16 for transcription.

Common parameters:

- `source_descriptor`: usually `$steps.<media_source_step>.audio_source_descriptor`
- `sample_rate`: usually `16000`
- `channels`: usually `1`
- `frame_duration_seconds`: usually `0.08`

Output:

- `pcm_descriptor`

## True Streaming Transcription Session

Block type:

```text
roboflow_livepeer_blocks/livepeer_true_streaming_transcription_session@v1
```

Purpose: transcribe PCM16 audio and write streaming transcript artifacts.

Common parameters:

- `pcm_descriptor`: usually `$steps.<pcm_step>.pcm_descriptor`
- `duration_seconds`
- `startup_timeout_seconds`
- `output_dir`
- `session_id`
- `language`: usually `en`
- `preset`: often `meeting` for speech
- `max_speakers`
- `transcription_backend`: `livepeer_remote_http` for clearinghouse HTTP chunking or `local` for a local websocket runner
- `livepeer_capability`: commonly `openai:audio-transcriptions`
- `livepeer_offering`: commonly `nemo-meeting`
- `vdo_ingest_mode`: commonly `segmented_wav`
- `vdo_segment_duration_seconds`
- `vdo_segment_startup_seconds`

Output:

- `transcription_session`

## Transcript Output

Block type:

```text
roboflow_livepeer_blocks/livepeer_transcript_output@v1
```

Purpose: persist/normalize transcript output fields from a transcription session.

Common parameter:

- `transcription_session`: usually `$steps.<transcription_step>.transcription_session`

Common outputs include transcript text and transcript artifact paths.

## Screen/Slide Capture

Block type:

```text
roboflow_livepeer_blocks/livepeer_screen_slide_capture@v1
```

Purpose: record a visual window, extract sampled frames, and select slide/screen-change artifacts.

Common parameters:

- `video_source_descriptor`: usually `$steps.<media_source_step>.video_source_descriptor`
- `duration_seconds`
- `startup_seconds`
- `frame_interval_seconds`
- `max_frames`
- `output_dir`
- `min_slide_gap_seconds`
- `slide_change_threshold`

Output:

- `capture_descriptor`

## Florence Screen/Slide Analysis

Block type:

```text
roboflow_livepeer_blocks/livepeer_florence2_screen_slide_analysis@v1
```

Purpose: analyze captured visual frames/slides and produce screen/slide text plus visual summary artifacts.

Common parameters:

- `capture_descriptor`: usually `$steps.<screen_capture_step>.capture_descriptor`
- `output_dir`
- `model_id`: commonly `florence-2-large`
- `vision_backend`: `livepeer_remote` for clearinghouse or `remote` for direct runner URL
- `florence2_runner_url`: direct runner URL when using `remote`; empty when using clearinghouse
- `livepeer_capability`: commonly `openai:vision`
- `livepeer_offering`: commonly `florence-2-large`

Common outputs include:

- `visual_status`
- `frame_count`
- `slide_count`
- `visual_result_json_path`
- `meeting_visual_events_jsonl_path`
- `slide_text`
- `screen_share_text`
- `meeting_visual_summary`

## Composition Notes

Connect block outputs by referencing prior step outputs with `$steps.<step_name>.<output_name>`.

Choose blocks by requested behavior:

- Audio-only live transcription: VDO Media Source -> PCM16 Audio Transform -> True Streaming Transcription -> Transcript Output.
- Visual-only screen understanding: VDO Media Source -> Screen/Slide Capture -> Florence Screen/Slide Analysis.
- Audio+visual capture: combine the audio and visual branches from the same VDO Media Source.
