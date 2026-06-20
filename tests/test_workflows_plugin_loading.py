import importlib
import os
import sys
from pathlib import Path


def test_workflows_plugins_can_load_livepeer_block(monkeypatch):
    monkeypatch.setenv("WORKFLOWS_PLUGINS", "roboflow_livepeer_blocks")

    plugin_names = os.environ["WORKFLOWS_PLUGINS"].split(",")
    modules = [importlib.import_module(name) for name in plugin_names]
    blocks = [block for module in modules for block in module.load_blocks()]

    assert [block.__name__ for block in blocks] == [
        "LivepeerAudioTranscribeV1",
        "LivepeerVDONinjaRollingAudioCaptureV1",
        "LivepeerAudioDiarizedTranscribeLocalV1",
        "LivepeerVDONinjaLiveDiarizedSessionV1",
        "LivepeerVDONinjaTrueStreamingSessionV1",
        "LivepeerVDONinjaMediaSourceV1",
        "LivepeerVDONinjaLiveAudioSourceV1",
        "LivepeerLocalAudioIngressLiveAudioSourceV1",
        "LivepeerPCM16AudioTransformV1",
        "LivepeerTrueStreamingTranscriptionSessionV1",
        "LivepeerLocalAudioIngressTrueStreamingTranscriptionSessionV1",
        "LivepeerTranscriptOutputV1",
        "LivepeerScreenSlideCaptureV1",
        "LivepeerFlorence2ScreenSlideAnalysisV1",
        "LivepeerVDONinjaDirectTrueStreamingSessionV1",
    ]
    assert blocks[0].get_manifest().__name__ == "LivepeerAudioTranscribeManifest"


def test_reference_loader_can_discover_plugin(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    roboflow_reference = repo_root / "references" / "roboflow-inference"
    if str(roboflow_reference) not in sys.path:
        sys.path.insert(0, str(roboflow_reference))
    module_prefixes = (
        "inference",
        "roboflow_livepeer_blocks",
    )
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in module_prefixes)
    }
    for name in list(previous_modules):
        sys.modules.pop(name, None)

    try:
        try:
            from inference.core.workflows.execution_engine.introspection.blocks_loader import (
                load_blocks_from_plugin,
                load_initializers_from_plugin,
            )
        except ModuleNotFoundError:
            return
        monkeypatch.setenv("WORKFLOWS_PLUGINS", "roboflow_livepeer_blocks")

        blocks = load_blocks_from_plugin("roboflow_livepeer_blocks")
        initializers = load_initializers_from_plugin("roboflow_livepeer_blocks")

        assert len(blocks) == 15
        assert blocks[0].block_class.__name__ == "LivepeerAudioTranscribeV1"
        assert blocks[1].block_class.__name__ == "LivepeerVDONinjaRollingAudioCaptureV1"
        assert blocks[2].block_class.__name__ == "LivepeerAudioDiarizedTranscribeLocalV1"
        assert blocks[3].block_class.__name__ == "LivepeerVDONinjaLiveDiarizedSessionV1"
        assert blocks[4].block_class.__name__ == "LivepeerVDONinjaTrueStreamingSessionV1"
        assert blocks[5].block_class.__name__ == "LivepeerVDONinjaMediaSourceV1"
        assert blocks[6].block_class.__name__ == "LivepeerVDONinjaLiveAudioSourceV1"
        assert blocks[7].block_class.__name__ == "LivepeerLocalAudioIngressLiveAudioSourceV1"
        assert blocks[8].block_class.__name__ == "LivepeerPCM16AudioTransformV1"
        assert blocks[9].block_class.__name__ == "LivepeerTrueStreamingTranscriptionSessionV1"
        assert (
            blocks[10].block_class.__name__
            == "LivepeerLocalAudioIngressTrueStreamingTranscriptionSessionV1"
        )
        assert blocks[11].block_class.__name__ == "LivepeerTranscriptOutputV1"
        assert blocks[12].block_class.__name__ == "LivepeerScreenSlideCaptureV1"
        assert blocks[13].block_class.__name__ == "LivepeerFlorence2ScreenSlideAnalysisV1"
        assert blocks[14].block_class.__name__ == "LivepeerVDONinjaDirectTrueStreamingSessionV1"
        assert any(key.endswith(".api_key") for key in initializers.keys())
        assert any(key.endswith(".base_url") for key in initializers.keys())
        assert any(key.endswith(".runner_url") for key in initializers.keys())
        assert any(key.endswith(".local_audio_ingest_url") for key in initializers.keys())
        assert any(key.endswith(".vdo_signaling_server_url") for key in initializers.keys())
        assert any(key.endswith(".roboflow_api_key") for key in initializers.keys())
        assert any(key.endswith(".roboflow_inference_url") for key in initializers.keys())
        assert any(key.endswith(".vision_backend") for key in initializers.keys())
        assert any(key.endswith(".florence2_runner_url") for key in initializers.keys())
    finally:
        for name in list(sys.modules):
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in module_prefixes):
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)
