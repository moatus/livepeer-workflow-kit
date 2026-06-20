import pytest

from roboflow_livepeer_blocks.florence_client import (
    COMPAT_LMM_INFER_ENDPOINT,
    LIVEPEER_CHAT_COMPLETIONS_ENDPOINT,
    PRIMARY_VISION_ANALYZE_ENDPOINT,
    Florence2VisionRunnerClient,
)
from roboflow_livepeer_blocks.vision import Florence2InferenceAnalyzer


class FakeResponse:
    def __init__(self, status_code, body, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = str(body)

    def json(self):
        return self._body


class FakeHttpClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def post(self, url, data=None, files=None, json=None, headers=None):
        self.requests.append(
            {"url": url, "data": data, "files": files, "json": json, "headers": headers}
        )
        return self.responses.pop(0)


def test_florence_client_posts_primary_v1_vision_analyze_endpoint(tmp_path):
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"jpg")
    http_client = FakeHttpClient(
        [
            FakeResponse(200, {"results": [{"caption": "a slide"}]}),
            FakeResponse(200, {"results": [{"detailed_caption": "a slide with a roadmap"}]}),
            FakeResponse(200, {"results": [{"ocr_text": "Roadmap"}]}),
            FakeResponse(200, {"results": [{"response": "presentation content"}]}),
        ]
    )
    client = Florence2VisionRunnerClient(
        base_url="http://florence2-runner:8080",
        http_client=http_client,
    )

    result = client.analyze_image(
        image_path=str(image_path),
        model_id="florence-2-large",
        meeting_context_prompt="Separate meeting UI from content.",
    )

    assert http_client.requests[0]["url"] == "http://florence2-runner:8080/v1/vision/analyze"
    assert http_client.requests[0]["json"]["model"] == "florence-2-large"
    assert http_client.requests[0]["json"]["model_id"] == "florence-2-large"
    assert http_client.requests[0]["json"]["task"] == "caption"
    assert http_client.requests[0]["json"]["input"]["type"] == "image_base64"
    assert result["caption"] == "a slide"
    assert result["detailed_caption"] == "a slide with a roadmap"
    assert result["ocr_text"] == "Roadmap"
    assert result["meeting_context"]["text"] == "presentation content"
    assert result["api_endpoint"] == PRIMARY_VISION_ANALYZE_ENDPOINT


def test_florence_client_falls_back_to_infer_lmm_for_older_runner(tmp_path):
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"jpg")
    http_client = FakeHttpClient(
        [
            FakeResponse(404, {"detail": "not found"}),
            FakeResponse(200, {"caption": "a dashboard"}),
            FakeResponse(404, {"detail": "not found"}),
            FakeResponse(200, {"detailed_caption": "a dashboard with charts"}),
            FakeResponse(404, {"detail": "not found"}),
            FakeResponse(200, {"ocr_text": "Revenue"}),
        ]
    )
    client = Florence2VisionRunnerClient(base_url="http://runner", http_client=http_client)

    result = client.analyze_image(image_path=str(image_path), model_id="florence-2-large")

    assert [request["url"] for request in http_client.requests] == [
        "http://runner/v1/vision/analyze",
        "http://runner/infer/lmm",
        "http://runner/v1/vision/analyze",
        "http://runner/infer/lmm",
        "http://runner/v1/vision/analyze",
        "http://runner/infer/lmm",
    ]
    assert result["caption"] == "a dashboard"
    assert result["detailed_caption"] == "a dashboard with charts"
    assert result["ocr_text"] == "Revenue"
    assert result["api_endpoint"] == COMPAT_LMM_INFER_ENDPOINT


def test_florence_client_can_call_livepeer_clearinghouse_for_remote_runner(tmp_path):
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"jpg")
    http_client = FakeHttpClient(
        [
            FakeResponse(
                200,
                {
                    "broker_url": "https://broker.example",
                    "payment_envelope": {"test": True},
                    "mode": "http-reqresp@v0",
                    "settle_endpoint": "/v1/jobs/1/settle",
                },
            ),
            FakeResponse(
                200,
                {"choices": [{"message": {"content": "a slide"}}]},
                headers={"X-Livepeer-Work-Units": "1"},
            ),
            FakeResponse(200, {"ok": True}),
            FakeResponse(
                200,
                {
                    "broker_url": "https://broker.example",
                    "payment_envelope": {"test": True},
                    "mode": "http-reqresp@v0",
                    "settle_endpoint": "/v1/jobs/2/settle",
                },
            ),
            FakeResponse(
                200,
                    {"choices": [{"message": {"content": "a slide with quarterly metrics"}}]},
                headers={"X-Livepeer-Work-Units": "1"},
            ),
            FakeResponse(200, {"ok": True}),
            FakeResponse(
                200,
                {
                    "broker_url": "https://broker.example",
                    "payment_envelope": {"test": True},
                    "mode": "http-reqresp@v0",
                    "settle_endpoint": "/v1/jobs/3/settle",
                },
            ),
            FakeResponse(
                200,
                {"choices": [{"message": {"content": "Revenue"}}]},
                headers={"X-Livepeer-Work-Units": "1"},
            ),
            FakeResponse(200, {"ok": True}),
        ]
    )
    client = Florence2VisionRunnerClient(
        base_url="http://unused-runner",
        livepeer_api_key="lp-key",
        livepeer_base_url="https://loc.example",
        capability="openai:vision",
        offering="florence-2-large",
        http_client=http_client,
    )

    result = client.analyze_image(image_path=str(image_path), model_id="florence-2-large")

    assert http_client.requests[0]["url"] == "https://loc.example/v1/jobs"
    assert http_client.requests[0]["json"]["capability"] == "openai:vision"
    assert http_client.requests[1]["url"] == "https://broker.example/v1/cap"
    assert "task" not in http_client.requests[1]["json"]
    assert "input" not in http_client.requests[1]["json"]
    assert http_client.requests[1]["json"]["model"] == "florence-2-large"
    assert http_client.requests[1]["json"]["messages"][0]["role"] == "user"
    assert http_client.requests[1]["json"]["messages"][0]["content"][0] == {
        "type": "text",
        "text": "<CAPTION>",
    }
    assert http_client.requests[1]["json"]["messages"][0]["content"][1]["type"] == "image_url"
    assert http_client.requests[1]["json"]["messages"][0]["content"][1]["image_url"][
        "url"
    ].startswith("data:image/jpeg;base64,")
    assert http_client.requests[2]["url"] == "https://loc.example/v1/jobs/1/settle"
    assert http_client.requests[4]["json"]["messages"][0]["content"][0]["text"] == "<DETAILED_CAPTION>"
    assert http_client.requests[7]["json"]["messages"][0]["content"][0]["text"] == "<OCR>"
    assert result["caption"] == "a slide"
    assert result["detailed_caption"] == "a slide with quarterly metrics"
    assert result["ocr_text"] == "Revenue"
    assert result["api_endpoint"] == LIVEPEER_CHAT_COMPLETIONS_ENDPOINT


def test_florence_client_uses_openai_style_payload_for_forced_livepeer_gateway(tmp_path):
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"jpg")

    def open_response(index):
        return FakeResponse(
            200,
            {
                "broker_url": "https://broker.example",
                "payment_envelope": {"test": True},
                "mode": "http-reqresp@v0",
                "settle_endpoint": f"/v1/jobs/{index}/settle",
            },
        )

    http_client = FakeHttpClient(
        [
            open_response(1),
            FakeResponse(
                200,
                {"choices": [{"message": {"content": "a slide"}}]},
                headers={"X-Livepeer-Work-Units": "1"},
            ),
            FakeResponse(200, {"ok": True}),
            open_response(2),
            FakeResponse(
                200,
                {"choices": [{"message": {"content": "a slide with metrics"}}]},
                headers={"X-Livepeer-Work-Units": "1"},
            ),
            FakeResponse(200, {"ok": True}),
            open_response(3),
            FakeResponse(
                200,
                {"choices": [{"message": {"content": "Revenue"}}]},
                headers={"X-Livepeer-Work-Units": "1"},
            ),
            FakeResponse(200, {"ok": True}),
        ]
    )
    client = Florence2VisionRunnerClient(
        base_url="",
        livepeer_api_key="lp-key",
        livepeer_base_url="https://loc.example",
        capability="openai:vision",
        offering="florence-2-large",
        use_livepeer_gateway=True,
        http_client=http_client,
    )

    result = client.analyze_image(image_path=str(image_path), model_id="florence-2-large")

    urls = [request["url"] for request in http_client.requests]
    assert "" not in urls
    assert "https://broker.example/v1/cap" in urls
    assert all(not url.endswith("/infer/lmm") for url in urls)
    assert all(
        request["json"]["capability"] == "openai:vision"
        for request in http_client.requests
        if request["url"].endswith("/v1/jobs")
    )
    assert http_client.requests[1]["json"]["messages"][0]["content"][0]["text"] == "<CAPTION>"
    assert http_client.requests[4]["json"]["messages"][0]["content"][0]["text"] == "<DETAILED_CAPTION>"
    assert http_client.requests[7]["json"]["messages"][0]["content"][0]["text"] == "<OCR>"
    assert result["caption"] == "a slide"
    assert result["detailed_caption"] == "a slide with metrics"
    assert result["ocr_text"] == "Revenue"
    assert result["api_endpoint"] == LIVEPEER_CHAT_COMPLETIONS_ENDPOINT


def test_florence_client_does_not_fallback_to_direct_compat_after_broker_404(tmp_path):
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"jpg")
    http_client = FakeHttpClient(
        [
            FakeResponse(
                200,
                {
                    "broker_url": "https://broker.example",
                    "payment_envelope": {"test": True},
                    "mode": "http-reqresp@v0",
                    "settle_endpoint": "/v1/jobs/1/settle",
                },
            ),
            FakeResponse(404, {"error": "NO_ROUTE_AVAILABLE"}),
            FakeResponse(200, {"ok": True}),
        ]
    )
    client = Florence2VisionRunnerClient(
        base_url="",
        livepeer_api_key="lp-key",
        livepeer_base_url="https://loc.example",
        capability="openai:vision",
        offering="florence-2-large",
        use_livepeer_gateway=True,
        http_client=http_client,
    )

    with pytest.raises(RuntimeError, match="NO_ROUTE_AVAILABLE"):
        client.analyze_image(image_path=str(image_path), model_id="florence-2-large")

    assert [request["url"] for request in http_client.requests] == [
        "https://loc.example/v1/jobs",
        "https://broker.example/v1/cap",
        "https://loc.example/v1/jobs/1/settle",
    ]
    assert http_client.requests[0]["json"]["capability"] == "openai:vision"
    assert http_client.requests[1]["json"]["messages"][0]["content"][0]["text"] == "<CAPTION>"


def test_florence_client_forced_livepeer_gateway_never_uses_empty_direct_url(tmp_path):
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"jpg")
    http_client = FakeHttpClient([])
    client = Florence2VisionRunnerClient(
        base_url="",
        livepeer_api_key="",
        livepeer_base_url="https://loc.example",
        capability="openai:vision",
        offering="florence-2-large",
        use_livepeer_gateway=True,
        http_client=http_client,
    )

    with pytest.raises(ValueError, match="LIVEPEER_OPEN_CLEARINGHOUSE_API_KEY"):
        client.analyze_image(image_path=str(image_path), model_id="florence-2-large")

    assert http_client.requests == []


def test_florence_analyzer_delegates_remote_backend_to_runner_client():
    calls = []

    class FakeRemoteClient:
        def __init__(self, base_url, **kwargs):
            self.base_url = base_url
            self.kwargs = kwargs

        def analyze_image(self, **kwargs):
            calls.append({"base_url": self.base_url, "kwargs": self.kwargs, **kwargs})
            return {
                "caption": "remote caption",
                "detailed_caption": "remote details",
                "ocr_text": "remote text",
                "meeting_context": {"text": "remote context"},
            }

    analyzer = Florence2InferenceAnalyzer(
        model_id="florence-2-large",
        vision_backend="remote",
        runner_url="http://remote-runner",
        livepeer_api_key="",
        remote_client_cls=FakeRemoteClient,
    )

    result = analyzer.analyze_image("/tmp/frame.jpg", meeting_context_prompt="context prompt")

    assert calls == [
        {
            "base_url": "http://remote-runner",
            "kwargs": {
                "livepeer_api_key": "",
                "livepeer_base_url": "https://loc.cloudspe.com",
                "capability": "openai:vision",
                "offering": "florence-2-large",
                "use_livepeer_gateway": False,
            },
            "image_path": "/tmp/frame.jpg",
            "model_id": "florence-2-large",
            "meeting_context_prompt": "context prompt",
        }
    ]
    assert result["caption"] == "remote caption"


def test_florence_analyzer_forces_livepeer_gateway_for_livepeer_remote_backend():
    calls = []

    class FakeRemoteClient:
        def __init__(self, base_url, **kwargs):
            self.base_url = base_url
            self.kwargs = kwargs

        def analyze_image(self, **kwargs):
            calls.append({"base_url": self.base_url, "kwargs": self.kwargs, **kwargs})
            return {
                "caption": "brokered caption",
                "detailed_caption": "brokered details",
                "ocr_text": "brokered text",
                "meeting_context": {"text": ""},
            }

    analyzer = Florence2InferenceAnalyzer(
        model_id="florence-2-large",
        vision_backend="livepeer_remote",
        runner_url="",
        livepeer_api_key="lp-key",
        livepeer_base_url="https://loc.example",
        livepeer_capability="openai:vision",
        livepeer_offering="florence-2-large",
        remote_client_cls=FakeRemoteClient,
    )

    result = analyzer.analyze_image("/tmp/frame.jpg")

    assert len(calls) == 1
    assert calls[0]["kwargs"] == {
        "livepeer_api_key": "lp-key",
        "livepeer_base_url": "https://loc.example",
        "capability": "openai:vision",
        "offering": "florence-2-large",
        "use_livepeer_gateway": True,
    }
    assert calls[0]["image_path"] == "/tmp/frame.jpg"
    assert calls[0]["model_id"] == "florence-2-large"
    assert calls[0]["meeting_context_prompt"] == ""
    assert result["caption"] == "brokered caption"
