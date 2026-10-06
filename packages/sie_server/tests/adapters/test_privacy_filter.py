import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from sie_server.adapters.privacy_filter import adapter as privacy_module
from sie_server.adapters.privacy_filter.adapter import PRIVACY_LABELS, PrivacyFilterAdapter, _StrictDecoder
from sie_server.api.extract import _build_response
from sie_server.core.worker.handlers.extract import ExtractHandler
from sie_server.types.inputs import InvalidInputError, Item

BIAS_KEYS = (
    "transition_bias_background_stay",
    "transition_bias_background_to_start",
    "transition_bias_inside_to_continue",
    "transition_bias_inside_to_end",
    "transition_bias_end_to_background",
    "transition_bias_end_to_start",
)


class Encoding:
    def __init__(self) -> None:
        self.rows = {"short": [1, 2], "longer": [3, 4, 5, 6, 7], "": []}
        self.decoded = {tuple(tokens): text for text, tokens in self.rows.items()}
        self.calls = []
        self._mergeable_ranks = {b" " * 128: 100, b"word": 101}
        self._special_tokens = {"<|endoftext|>": 102}

    def encode(self, text, *, allowed_special):
        self.calls.append((text, allowed_special))
        return self.rows[text]

    def decode(self, tokens):
        return self.decoded[tuple(tokens)]


@pytest.fixture
def native_modules(monkeypatch, tmp_path):
    checkpoint = tmp_path / "original"
    checkpoint.mkdir()
    config = {
        "model_type": "privacy_filter",
        "max_position_embeddings": 131072,
        "encoding": "o200k_base",
        "param_dtype": "bfloat16",
    }
    (checkpoint / "config.json").write_text(json.dumps(config))
    (checkpoint / "model.safetensors").touch()
    (checkpoint / "dtypes.json").write_text("{}")
    biases = {key: index / 10 + 0.1 for index, key in enumerate(BIAS_KEYS)}
    calibration = checkpoint / "viterbi_calibration.json"
    calibration.write_text(json.dumps({"operating_points": {"default": {"biases": biases}}}))
    runtime = SimpleNamespace(
        model=MagicMock(),
        encoding=Encoding(),
        label_info=SimpleNamespace(span_class_names=("O", *PRIVACY_LABELS)),
        n_ctx=4,
    )
    load = MagicMock(return_value=runtime)
    predict = MagicMock(
        side_effect=lambda _runtime, text, **_kwargs: SimpleNamespace(text=text, spans=(), decoded_mismatch=False)
    )
    factory = MagicMock(return_value=SimpleNamespace(decode=lambda scores: [0] * len(scores)))
    resolve = MagicMock(
        side_effect=lambda path: json.loads(Path(path).read_text())["operating_points"]["default"]["biases"]
    )
    modules = {name: ModuleType(name) for name in ("opf", "opf._core", "opf._core.runtime", "opf._core.decoding")}
    modules["opf"].__path__ = []
    modules["opf._core"].__path__ = []
    modules["opf._core.runtime"].load_inference_runtime = load
    modules["opf._core.runtime"].predict_text = predict
    modules["opf._core.decoding"].ViterbiCRFDecoder = factory
    modules["opf._core.decoding"].resolve_viterbi_biases_from_calibration_path = resolve
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    download = MagicMock(return_value=str(tmp_path))
    monkeypatch.setattr(privacy_module, "snapshot_download", download)
    return SimpleNamespace(
        root=tmp_path,
        checkpoint=checkpoint,
        biases=biases,
        runtime=runtime,
        load=load,
        predict=predict,
        factory=factory,
        resolve=resolve,
        download=download,
    )


def loaded(native_modules, *, window=4):
    native_modules.runtime.n_ctx = window
    adapter = PrivacyFilterAdapter("openai/privacy-filter", max_seq_length=window, revision="a" * 40)
    adapter.load("cpu")
    return adapter


def test_load_pins_original_artifacts_and_native_context(native_modules):
    adapter = loaded(native_modules)
    native_modules.download.assert_called_once_with(
        repo_id="openai/privacy-filter",
        revision="a" * 40,
        allow_patterns=[
            "original/config.json",
            "original/model.safetensors",
            "original/dtypes.json",
            "original/viterbi_calibration.json",
        ],
    )
    native_modules.load.assert_called_once_with(
        checkpoint=str(native_modules.checkpoint),
        device_name="cpu",
        n_ctx_override=4,
        trim_span_whitespace=True,
        discard_overlapping_predicted_spans=False,
        output_mode="typed",
    )
    native_modules.resolve.assert_called_once_with(str(native_modules.checkpoint / "viterbi_calibration.json"))
    native_modules.runtime.model.to.assert_not_called()
    assert adapter._biases == native_modules.biases
    adapter.unload()
    assert adapter._model is None
    assert adapter._runtime is None
    assert adapter._biases is None


def test_default_preserves_native_mixed_precision(native_modules):
    model = torch.nn.Module()
    model.register_parameter("projection", torch.nn.Parameter(torch.tensor([0.75], dtype=torch.bfloat16)))
    model.register_parameter("norm_scale", torch.nn.Parameter(torch.tensor([1.003], dtype=torch.float32)))
    model.register_buffer("rotary_table", torch.tensor([0.1234567], dtype=torch.float32))
    native_modules.runtime.model = model
    norm = model.norm_scale.detach().clone()
    rotary = model.rotary_table.clone()
    loaded(native_modules)
    assert model.projection.dtype == torch.bfloat16
    assert model.norm_scale.dtype == torch.float32
    assert model.rotary_table.dtype == torch.float32
    assert torch.equal(model.norm_scale, norm)
    assert torch.equal(model.rotary_table, rotary)


def test_local_native_checkpoint_does_not_download(native_modules):
    adapter = PrivacyFilterAdapter(
        native_modules.checkpoint, max_seq_length=4, checkpoint_subdir="", compute_precision="float32"
    )
    adapter.load("cpu")
    native_modules.download.assert_not_called()
    native_modules.runtime.model.to.assert_called_once_with(dtype=torch.float32)


def test_missing_calibration_fails_before_loading_weights(native_modules):
    (native_modules.checkpoint / "viterbi_calibration.json").unlink()
    with pytest.raises(ValueError, match="missing viterbi_calibration"):
        loaded(native_modules)
    native_modules.load.assert_not_called()


def test_invalid_calibration_fails_before_loading_weights(native_modules):
    native_modules.resolve.return_value = {**native_modules.biases, BIAS_KEYS[0]: float("nan")}
    native_modules.resolve.side_effect = None
    with pytest.raises(ValueError, match="six finite"):
        loaded(native_modules)
    native_modules.load.assert_not_called()


def test_window_beyond_checkpoint_capacity_is_rejected(native_modules):
    with pytest.raises(ValueError, match="capacity"):
        loaded(native_modules, window=131073)
    native_modules.load.assert_not_called()


def test_delta_preserves_nonzero_calibration_and_request_isolation(native_modules):
    adapter = loaded(native_modules)
    for delta in (0.5, -0.5, 0.0):
        adapter.extract([Item(text="short")], options={"span_entry_bias": delta})
        expected = dict(native_modules.biases)
        expected["transition_bias_background_to_start"] += delta
        assert native_modules.factory.call_args.kwargs == {"label_info": native_modules.runtime.label_info, **expected}
    assert adapter._biases == native_modules.biases
    assert native_modules.runtime.encoding.calls == [("short", "all")] * 3


def test_full_token_window_errors_are_positional_and_not_forwarded(native_modules):
    adapter = loaded(native_modules)
    output = adapter.extract([Item(text="longer"), Item(text="short"), Item(text="")])
    assert output.entities == [[], [], []]
    assert output.input_token_counts == [0, 2, 0]
    assert [error.code if error else None for error in output.errors] == ["INPUT_TOO_LONG", None, None]
    assert [call.args[1] for call in native_modules.predict.call_args_list] == ["short", ""]
    assert output.data[1] == {"confidence_scores_available": False, "decoder": "viterbi"}


def test_exact_token_limit_is_accepted(native_modules):
    adapter = loaded(native_modules, window=5)
    output = adapter.extract([Item(text="longer")])
    assert output.errors is None
    assert output.input_token_counts == [5]
    native_modules.predict.assert_called_once()


def test_compressed_one_token_input_is_accepted(native_modules):
    adapter = loaded(native_modules, window=1)
    text = " " * 128
    native_modules.runtime.encoding.rows[text] = [100]
    native_modules.runtime.encoding.decoded[(100,)] = text
    output = adapter.extract([Item(text=text)])
    assert output.errors is None
    assert output.input_token_counts == [1]
    native_modules.predict.assert_called_once()


def test_vocabulary_derived_oversize_bound_avoids_tokenization(native_modules):
    adapter = loaded(native_modules, window=1)
    output = adapter.extract([Item(text=" " * 129)])
    assert output.errors[0].code == "INPUT_TOO_LONG"
    assert native_modules.runtime.encoding.calls == []
    native_modules.predict.assert_not_called()


def test_roundtrip_mismatch_never_runs_model(native_modules):
    adapter = loaded(native_modules)
    native_modules.runtime.encoding.decoded[(1, 2)] = "replaced text"
    output = adapter.extract([Item(text="short")])
    assert output.errors[0].code == "INVALID_INPUT"
    assert output.input_token_counts == [0]
    native_modules.predict.assert_not_called()


def test_original_unicode_spans_and_unscored_wire_contract(native_modules):
    adapter = loaded(native_modules, window=8)
    text = "🙂 İpek met İpek."
    native_modules.runtime.encoding.rows[text] = [11, 12, 13]
    native_modules.runtime.encoding.decoded[(11, 12, 13)] = text
    spans = (
        SimpleNamespace(label="private_person", start=2, end=6, text="İpek"),
        SimpleNamespace(label="private_person", start=11, end=15, text="İpek"),
    )
    native_modules.predict.return_value = SimpleNamespace(text=text, spans=spans, decoded_mismatch=False)
    native_modules.predict.side_effect = None
    output = adapter.extract([Item(id="unicode", text=text)], labels=["private_person"])
    assert output.input_token_counts == [3]
    assert all(text[entity["start"] : entity["end"]] == entity["text"] for entity in output.entities[0])
    assert all("score" not in entity for entity in output.entities[0])
    wire = _build_response(
        "openai/privacy-filter", [Item(id="unicode", text=text)], ExtractHandler.format_output(output)
    )
    assert [entity["score"] for entity in wire["items"][0]["entities"]] == [1.0, 1.0]
    assert wire["items"][0]["data"]["confidence_scores_available"] is False


def test_subset_filters_after_same_native_prediction(native_modules):
    adapter = loaded(native_modules)
    native_modules.predict.side_effect = None
    native_modules.predict.return_value = SimpleNamespace(
        text="short",
        decoded_mismatch=False,
        spans=(SimpleNamespace(label="secret", start=0, end=5, text="short"),),
    )
    output = adapter.extract([Item(text="short")], labels=["private_person"])
    assert output.entities == [[]]
    assert output.input_token_counts == [2]
    native_modules.predict.assert_called_once()


def test_source_mismatch_and_invalid_spans_fail_closed(native_modules):
    adapter = loaded(native_modules)
    native_modules.predict.side_effect = None
    native_modules.predict.return_value = SimpleNamespace(text="other", spans=(), decoded_mismatch=True)
    output = adapter.extract([Item(text="short")])
    assert output.errors[0].code == "INFERENCE_ERROR"
    assert output.entities == [[]]
    assert output.input_token_counts == [0]
    native_modules.predict.return_value = SimpleNamespace(
        text="short",
        decoded_mismatch=False,
        spans=(SimpleNamespace(label="secret", start=0, end=2, text="wrong"),),
    )
    output = adapter.extract([Item(text="short")])
    assert output.errors[0].code == "INFERENCE_ERROR"
    assert output.entities == [[]]


@pytest.mark.parametrize("labels", [[], ["person"], ["secret", "secret"], "secret", [None]])
def test_dynamic_or_invalid_label_policy_is_rejected(native_modules, labels):
    adapter = loaded(native_modules)
    with pytest.raises(InvalidInputError, match="labels"):
        adapter.extract([Item(text="short")], labels=labels)
    native_modules.predict.assert_not_called()


@pytest.mark.parametrize("delta", [True, "0.5", float("nan"), float("inf"), 10.1, -10.1])
def test_bias_validation_precedes_inference(native_modules, delta):
    adapter = loaded(native_modules)
    with pytest.raises(InvalidInputError, match="span_entry_bias"):
        adapter.extract([Item(text="short")], options={"span_entry_bias": delta})
    native_modules.predict.assert_not_called()


@pytest.mark.parametrize(
    "kwargs", [{"options": {"threshold": 0.5}}, {"output_schema": {}}, {"instruction": "mask dates"}]
)
def test_unsupported_request_controls_are_rejected(native_modules, kwargs):
    adapter = loaded(native_modules)
    with pytest.raises(InvalidInputError):
        adapter.extract([Item(text="short")], **kwargs)
    native_modules.predict.assert_not_called()


@pytest.mark.parametrize("labels", [[0], [33, 0], [True, 0]])
def test_invalid_decoder_cannot_fall_back_to_argmax(labels):
    decoder = _StrictDecoder(SimpleNamespace(decode=lambda scores: labels))
    with pytest.raises(ValueError, match="invalid token labels"):
        decoder.decode(torch.zeros((2, 33)))


def test_nonfinite_scores_never_reach_decoder():
    native = MagicMock()
    with pytest.raises(ValueError, match="invalid token scores"):
        _StrictDecoder(native).decode(torch.full((2, 33), float("nan")))
    native.decode.assert_not_called()


def test_extreme_finite_scores_never_enter_native_argmax_fallback():
    native = MagicMock()
    for key in BIAS_KEYS:
        setattr(native, key, 0.0)
    with pytest.raises(ValueError, match="safe Viterbi range"):
        _StrictDecoder(native).decode(torch.full((2, 33), -3e38))
    native.decode.assert_not_called()


@pytest.mark.parametrize("labels", [[2, 2], [1, 0], [1, 2], [3, 0]])
def test_illegal_bioes_paths_fail_closed(labels):
    info = SimpleNamespace(
        token_boundary_tags={0: None, 1: "B", 2: "I", 3: "E", 4: "S"},
        token_to_span_label={0: 0, 1: 1, 2: 1, 3: 1, 4: 1},
    )
    native = SimpleNamespace(decode=lambda scores: labels, label_info=info)
    with pytest.raises(ValueError, match="BIOES path"):
        _StrictDecoder(native).decode(torch.full((len(labels), 33), -3.5))


def test_legal_bioes_path_uses_native_decoder_unchanged():
    info = SimpleNamespace(
        token_boundary_tags={0: None, 1: "B", 2: "I", 3: "E", 4: "S"},
        token_to_span_label={0: 0, 1: 1, 2: 1, 3: 1, 4: 1},
    )
    labels = [1, 2, 3, 0, 4]
    native = SimpleNamespace(decode=lambda scores: labels, label_info=info)
    assert _StrictDecoder(native).decode(torch.full((5, 33), -3.5)) == labels
