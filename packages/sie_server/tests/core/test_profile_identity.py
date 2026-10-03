from copy import deepcopy
from types import SimpleNamespace

import pytest
from sie_server.config.engine import EngineConfig
from sie_server.config.model import ModelConfig
from sie_server.core import profile_identity

PIN = "a" * 40


def config(**changes: object) -> ModelConfig:
    data = {
        "sie_id": "catalog/model",
        "hf_id": "weights/model",
        "hf_revision": PIN,
        "inputs": {"text": True},
        "tasks": {"encode": {"dense": {"dim": 384}}},
        "max_sequence_length": 512,
        "profiles": {
            "default": {
                "adapter_path": "sie_server.adapters.bge_m3:BGEM3Adapter",
                "max_batch_tokens": 8192,
                "compute_precision": "float32",
                "adapter_options": {
                    "loadtime": {"pooling": "mean", "trust_remote_code": False},
                    "runtime": {"normalize": True},
                },
            }
        },
        **changes,
    }
    return ModelConfig.model_validate(data)


@pytest.fixture(autouse=True)
def execution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        profile_identity, "_execution_code", lambda: {"sources": "f" * 64, "dependencies": {"torch": "2.9"}}
    )


def identity(model: ModelConfig, profile: str = "default", *, device: str = "cpu") -> str | None:
    return profile_identity.local_profile_identity(model, profile, device=device)


def test_alias_and_inheritance_preserve_resolved_identity() -> None:
    original = config()
    renamed = config(sie_id="alias/model")
    inherited = config(profiles={**original.model_dump()["profiles"], "other": {"extends": "default"}})
    assert identity(original) == identity(renamed)
    assert identity(original) == identity(inherited, "other")
    result = identity(original)
    assert result is not None
    assert result.startswith("v1:sha256:")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hf_id", "different/weights"),
        ("hf_revision", "b" * 40),
        ("max_sequence_length", 1024),
        ("hf_tokenizer_dependencies", {"tokenizer/model": "b" * 40}),
    ],
)
def test_model_semantics_change_identity(field: str, value: object) -> None:
    assert identity(config()) != identity(config(**{field: value}))


@pytest.mark.parametrize(("section", "key", "value"), [("loadtime", "pooling", "cls"), ("runtime", "normalize", False)])
def test_profile_semantics_change_identity(section: str, key: str, value: object) -> None:
    data = config().model_dump()
    data["profiles"]["default"]["adapter_options"][section][key] = value
    assert identity(config()) != identity(ModelConfig.model_validate(data))


def test_precision_and_device_change_identity() -> None:
    data = config().model_dump()
    data["profiles"]["default"]["compute_precision"] = "float16"
    assert identity(config()) != identity(ModelConfig.model_validate(data))
    assert identity(config(), device="cpu") != identity(config(), device="cuda:0")


def test_execution_stack_change_changes_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    first = identity(config())
    monkeypatch.setattr(
        profile_identity, "_execution_code", lambda: {"sources": "e" * 64, "dependencies": {"torch": "2.9"}}
    )
    assert identity(config()) != first
    monkeypatch.setattr(
        profile_identity, "_execution_code", lambda: {"sources": "f" * 64, "dependencies": {"torch": "2.10"}}
    )
    assert identity(config()) != first


def test_runtime_settings_are_observed_again(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile_identity, "_execution_runtime", lambda: {"matmul_precision": "highest"})
    first = identity(config())
    assert first is not None
    monkeypatch.setattr(profile_identity, "_execution_runtime", lambda: {"matmul_precision": "high"})
    assert identity(config()) != first


def test_runtime_version_failure_does_not_break_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    def incompatible_cudnn() -> None:
        raise RuntimeError("incompatible cuDNN")

    monkeypatch.setattr(profile_identity.torch.backends.cudnn, "version", incompatible_cudnn)
    assert identity(config()) is None


@pytest.mark.parametrize("revision", [None, "main", "short"])
def test_mutable_weights_have_no_identity(revision: str | None) -> None:
    assert identity(config(hf_revision=revision)) is None


def test_unpinned_tokenizer_has_no_identity() -> None:
    assert identity(config(hf_tokenizer_dependencies={"tokenizer/model": "main"})) is None


def test_remote_profile_cannot_claim_local_identity() -> None:
    data = config().model_dump()
    data["profiles"]["remote"] = {
        "adapter_path": "sie_server.adapters.remote.sie:SieUpstreamAdapter",
        "max_batch_tokens": 8192,
        "adapter_options": {"loadtime": {"upstream": "private", "upstream_model": "served/model"}},
    }
    assert identity(ModelConfig.model_validate(data), "remote") is None


def test_unidentified_custom_adapter_has_no_identity() -> None:
    data = config().model_dump()
    data["profiles"]["default"]["adapter_path"] = "custom.vendor:Adapter"
    assert identity(ModelConfig.model_validate(data)) is None


def test_unserializable_or_nonfinite_options_fail_closed() -> None:
    data = config().model_dump()
    for value in (object(), float("nan")):
        changed = deepcopy(data)
        changed["profiles"]["default"]["adapter_options"]["loadtime"]["custom"] = value
        assert identity(ModelConfig.model_validate(changed)) is None


def test_settings_order_does_not_change_identity() -> None:
    data = config().model_dump()
    data["profiles"]["default"]["adapter_options"] = {
        "runtime": {"normalize": True},
        "loadtime": {"pooling": "mean", "trust_remote_code": False},
    }
    assert identity(config()) == identity(ModelConfig.model_validate(data))


@pytest.mark.parametrize("source", ["/operator/local/model", "./local-model", "../local-model"])
def test_path_like_hf_source_has_no_identity(source: str) -> None:
    assert identity(config(hf_id=source)) is None


def test_local_weights_override_has_no_identity() -> None:
    assert identity(config(weights_path="/operator/local/model")) is None


def test_repo_shaped_local_directory_has_no_identity(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "weights" / "model").mkdir(parents=True)
    assert identity(config()) is None


@pytest.mark.parametrize(
    "adapter",
    [
        "sie_server.adapters.sglang:SGLangGenerationAdapter",
        "sie_server.adapters.mlx:MLXGenerationAdapter",
        "sie_server.adapters.bge_m3_flag:BGEM3FlagAdapter",
        "sie_server.adapters.bge_m3_flash:BGEM3FlashAdapter",
        "sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
        "sie_server.adapters.sentence_transformer:SentenceTransformerSparseAdapter",
        "sie_server.adapters.cross_encoder:CrossEncoderAdapter",
    ],
)
def test_child_or_alternative_weights_engine_has_no_identity(adapter: str) -> None:
    data = config().model_dump()
    data["profiles"]["default"]["adapter_path"] = adapter
    data["profiles"]["default"]["adapter_options"]["loadtime"]["mlx_repo"] = "mutable/alternate-repo"
    assert identity(ModelConfig.model_validate(data)) is None


def test_lora_weights_have_no_identity() -> None:
    data = config().model_dump()
    data["profiles"]["default"]["adapter_options"]["runtime"]["lora_id"] = "mutable/adapter"
    assert identity(ModelConfig.model_validate(data)) is None


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("revision", "main"),
        ("model_name_or_path", "mutable/alternate"),
        ("config_kwargs", {"revision": "main"}),
        ("config_kwargs", {"trust_remote_code": True}),
    ],
)
def test_loadtime_weight_or_remote_code_overrides_have_no_identity(key: str, value: object) -> None:
    data = config().model_dump()
    data["profiles"]["default"]["adapter_options"]["loadtime"][key] = value
    assert identity(ModelConfig.model_validate(data)) is None


def test_engine_defaults_must_be_identified_and_change_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile_identity, "_execution_hardware", lambda device: {"cuda": "observed test hardware"})
    data = config().model_dump()
    data["profiles"]["default"]["compute_precision"] = None
    model = ModelConfig.model_validate(data)
    assert identity(model, device="cuda:0") is None
    first = profile_identity.local_profile_identity(
        model, "default", device="cuda:0", engine_config=EngineConfig(default_compute_precision="float16")
    )
    second = profile_identity.local_profile_identity(
        model, "default", device="cuda:0", engine_config=EngineConfig(default_compute_precision="float32")
    )
    assert first is not None
    assert second is not None
    assert first != second


def test_remote_code_execution_has_no_identity() -> None:
    data = config().model_dump()
    data["profiles"]["default"]["adapter_options"]["loadtime"]["trust_remote_code"] = True
    assert identity(ModelConfig.model_validate(data)) is None


def test_deep_options_fail_closed() -> None:
    value: object = "nested"
    for _ in range(1200):
        value = [value]
    data = config().model_dump()
    data["profiles"]["default"]["adapter_options"]["loadtime"]["custom"] = value
    assert identity(ModelConfig.model_validate(data)) is None


@pytest.mark.parametrize(
    "facts",
    [{"cpu": {"brand": "another CPU"}}, {"cuda": {"name": "another GPU"}}, {"cuda": {"driver": "new revision"}}],
)
def test_observed_hardware_changes_identity(monkeypatch: pytest.MonkeyPatch, facts: dict) -> None:
    monkeypatch.setattr(profile_identity, "_execution_hardware", lambda device: {"cpu": {"brand": "first CPU"}})
    first = identity(config())
    assert first is not None
    monkeypatch.setattr(profile_identity, "_execution_hardware", lambda device: facts)
    assert identity(config()) != first


def test_unavailable_hardware_never_claims_execution_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(device: str) -> dict:
        raise ValueError("cannot observe device")

    monkeypatch.setattr(profile_identity, "_execution_hardware", unavailable)
    assert identity(config()) is None


@pytest.mark.parametrize("variable", profile_identity._NUMERICAL_ENVIRONMENT)
def test_numerical_environment_changes_identity(monkeypatch: pytest.MonkeyPatch, variable: str) -> None:
    monkeypatch.delenv(variable, raising=False)
    first = identity(config())
    assert first is not None
    monkeypatch.setenv(variable, "different-runtime-setting")
    assert identity(config()) != first


def test_linux_cpu_facts_ignore_frequency_but_bind_all_observed_core_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(profile_identity.platform, "system", lambda: "Linux")
    cpuinfo = "model name: CPU A\nflags: avx2 fma\nmicrocode: 1\nprocessor: 0\ncpu MHz: 3000\n"
    monkeypatch.setattr(profile_identity, "_hardware_text", lambda path: cpuinfo)
    first = profile_identity._cpu_hardware()
    cpuinfo += "model name: CPU A\nflags: avx2 fma\nmicrocode: 1\nprocessor: 1\ncpu MHz: 2500\n"
    assert profile_identity._cpu_hardware() == first
    cpuinfo += "model name: CPU B\nflags: avx2\nmicrocode: 2\n"
    assert profile_identity._cpu_hardware() != first


@pytest.mark.parametrize("cpuinfo", ["processor: 0\n", "model name: unidentified\n", "flags: avx2\n"])
def test_incomplete_cpu_hardware_facts_fail_closed(monkeypatch: pytest.MonkeyPatch, cpuinfo: str) -> None:
    monkeypatch.setattr(profile_identity.platform, "system", lambda: "Linux")
    monkeypatch.setattr(profile_identity, "_hardware_text", lambda path: cpuinfo)
    assert identity(config()) is None


def test_cuda_hardware_is_observed_with_installed_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile_identity, "_numerical_libraries", lambda: {"blas": "observed test BLAS"})
    monkeypatch.setattr(profile_identity, "_cpu_hardware", lambda: {"model": "CPU A"})
    monkeypatch.setattr(profile_identity.platform, "system", lambda: "Linux")
    monkeypatch.setattr(profile_identity.torch.version, "cuda", "13.0")
    monkeypatch.setattr(
        profile_identity.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            name="GPU A", major=9, minor=0, total_memory=80 << 30, multi_processor_count=132
        ),
    )
    driver = "NVRM version: installed driver A"
    monkeypatch.setattr(profile_identity, "_hardware_text", lambda path: driver)
    first = identity(config(), device="cuda:0")
    assert first is not None
    driver = "NVRM version: installed driver B"
    assert identity(config(), device="cuda:0") != first
    driver = "NVRM version: installed driver A"
    monkeypatch.setattr(profile_identity, "_cpu_hardware", lambda: {"model": "CPU B"})
    assert identity(config(), device="cuda:0") != first


def test_device_label_cannot_replace_missing_driver_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile_identity, "_cpu_hardware", lambda: {"model": "CPU A"})
    monkeypatch.setattr(profile_identity.platform, "system", lambda: "Linux")
    monkeypatch.setattr(profile_identity.torch.version, "cuda", "13.0")
    monkeypatch.setattr(
        profile_identity.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            name="GPU A", major=9, minor=0, total_memory=80 << 30, multi_processor_count=132
        ),
    )

    def missing(path: str) -> str:
        raise OSError("driver not visible")

    monkeypatch.setattr(profile_identity, "_hardware_text", missing)
    assert identity(config(), device="cuda:0") is None


@pytest.mark.parametrize("content", [b"", b" \n\t", b"x" * ((1 << 20) + 1)])
def test_empty_or_unbounded_hardware_facts_are_refused(tmp_path, content: bytes) -> None:
    path = tmp_path / "facts"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="hardware facts are unavailable"):
        profile_identity._hardware_text(str(path))


@pytest.mark.parametrize("error", [AssertionError, AttributeError, RuntimeError])
def test_cuda_observation_failure_does_not_break_model_metadata(
    monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    monkeypatch.setattr(profile_identity, "_cpu_hardware", lambda: {"model": "CPU A"})
    monkeypatch.setattr(profile_identity.platform, "system", lambda: "Linux")
    monkeypatch.setattr(profile_identity.torch.version, "cuda", "13.0")

    monkeypatch.setattr(profile_identity, "_numerical_libraries", lambda: {"blas": "observed test BLAS"})

    def unavailable(device: object) -> None:
        raise error("CUDA hardware not available")

    monkeypatch.setattr(profile_identity.torch.cuda, "get_device_properties", unavailable)
    assert identity(config(), device="cuda:99") is None


def test_loaded_blas_kernel_and_threads_change_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        profile_identity.np, "show_config", lambda **kwargs: {"Build Dependencies": {"blas": {"name": "openblas"}}}
    )
    library = {
        "user_api": "blas",
        "internal_api": "openblas",
        "version": "0.3.30",
        "num_threads": 1,
        "architecture": "Haswell",
        "threading_layer": "pthreads",
    }
    monkeypatch.setattr(profile_identity, "threadpool_info", lambda: [library])
    first = identity(config())
    assert first is not None
    library["architecture"] = "SkylakeX"
    assert identity(config()) != first
    library["architecture"] = "Haswell"
    library["num_threads"] = 2
    assert identity(config()) != first


def test_library_paths_do_not_enter_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        profile_identity.np, "show_config", lambda **kwargs: {"Build Dependencies": {"blas": {"name": "openblas"}}}
    )
    library = {
        "user_api": "blas",
        "internal_api": "openblas",
        "version": "0.3.30",
        "filepath": "/private/env/a/libblas.so",
        "architecture": "Haswell",
        "threading_layer": "pthreads",
        "num_threads": 1,
    }
    monkeypatch.setattr(profile_identity, "threadpool_info", lambda: [library])
    first = identity(config())
    assert first is not None
    library["filepath"] = "/private/env/b/libblas.so"
    assert identity(config()) == first


@pytest.mark.parametrize("libraries", [[], [{"user_api": "blas", "internal_api": "openblas", "version": None}]])
def test_unknown_numerical_libraries_refuse_identity(monkeypatch: pytest.MonkeyPatch, libraries: list[dict]) -> None:
    monkeypatch.setattr(profile_identity, "threadpool_info", lambda: libraries)
    monkeypatch.setattr(profile_identity.platform, "system", lambda: "Linux")
    monkeypatch.setattr(profile_identity, "_execution_hardware", lambda device: {"cpu": "test CPU"})
    assert identity(config()) is None


@pytest.mark.parametrize("other_blas", [False, True])
def test_accelerate_identity_binds_installed_os_version(monkeypatch: pytest.MonkeyPatch, other_blas: bool) -> None:
    libraries = (
        [
            {
                "user_api": "blas",
                "internal_api": "openblas",
                "version": "0.3.30",
                "num_threads": 1,
                "threading_layer": "pthreads",
                "architecture": "Haswell",
            }
        ]
        if other_blas
        else []
    )
    monkeypatch.setattr(profile_identity, "threadpool_info", lambda: libraries)
    monkeypatch.setattr(profile_identity.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        profile_identity.np, "show_config", lambda **kwargs: {"Build Dependencies": {"blas": {"name": "accelerate"}}}
    )
    monkeypatch.setattr(profile_identity, "_execution_hardware", lambda device: {"cpu": "test CPU"})
    monkeypatch.setattr(profile_identity.platform, "mac_ver", lambda: ("15.7", (), "arm64"))
    first = identity(config())
    assert first is not None
    monkeypatch.setattr(profile_identity.platform, "mac_ver", lambda: ("15.8", (), "arm64"))
    assert identity(config()) != first


@pytest.mark.parametrize("api", ["flexiblas", "mkl", "unknown"])
def test_unobserved_kernel_dispatch_never_claims_identity(monkeypatch: pytest.MonkeyPatch, api: str) -> None:
    library = {
        "user_api": "blas",
        "internal_api": api,
        "version": "1.0",
        "num_threads": 1,
        "threading_layer": "pthreads",
        "architecture": "Haswell",
        "current_backend": "OPENBLAS",
    }
    monkeypatch.setattr(profile_identity, "threadpool_info", lambda: [library])
    assert identity(config()) is None
    library["current_backend"] = "MKL"
    assert identity(config()) is None


@pytest.mark.parametrize("build", ["unidentified-system-blas", "mkl", "flexiblas", "blis"])
def test_unrelated_blas_cannot_authorize_numpy_backend(monkeypatch: pytest.MonkeyPatch, build: str) -> None:
    library = {
        "user_api": "blas",
        "internal_api": "openblas",
        "version": "0.3.30",
        "num_threads": 1,
        "threading_layer": "pthreads",
        "architecture": "Haswell",
    }
    monkeypatch.setattr(profile_identity, "threadpool_info", lambda: [library])
    monkeypatch.setattr(
        profile_identity.np, "show_config", lambda **kwargs: {"Build Dependencies": {"blas": {"name": build}}}
    )
    assert identity(config()) is None
