"""SGLang-backed vision-to-text extraction adapter.

This bridges SIE's synchronous ``extract`` contract to the existing SGLang
generation engine. Each image is submitted as an independent async generation
request so SGLang, rather than the worker's static batch, owns token-level
continuous batching.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huggingface_hub import snapshot_download

from sie_server.adapters._generation_base import GenerationResult, collect_generation
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ComputePrecision
from sie_server.adapters.base import ModelCapabilities, ModelDims
from sie_server.adapters.sglang.generation import SGLangGenerationAdapter
from sie_server.core.inference_output import ExtractOutput
from sie_server.types.inputs import InvalidMediaError
from sie_server.types.responses import Entity

if TYPE_CHECKING:
    from sie_server.types.inputs import ImageInput, Item

logger = logging.getLogger(__name__)

_ERR_NOT_LOADED = "Model not loaded. Call load() first."
_ERR_NO_IMAGES = "SGLangVisionExtractAdapter requires image input for extraction"


class SGLangVisionExtractAdapter(SGLangGenerationAdapter):
    """Run image-to-text extraction through SGLang continuous batching.

    The adapter inherits the established SGLang subprocess lifecycle and
    request implementation, then presents its output through SIE's ``extract``
    shape. A dedicated event-loop thread keeps SGLang's shared async HTTP client
    bound to one loop across repeated synchronous worker calls.
    """

    spec = AdapterSpec(
        inputs=("image",),
        outputs=("tokens", "json"),
        unload_fields=("_process", "_server_url", "_processor"),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        max_seq_length: int = 16384,
        mem_fraction_static: float = 0.85,
        compute_precision: ComputePrecision = "bfloat16",
        trust_remote_code: bool = True,
        processor_use_fast: bool | None = None,
        revision: str | None = None,
        served_model_name: str | None = None,
        max_new_tokens: int = 4096,
        num_beams: int = 1,
        system_prompt: str | None = None,
        default_prompt: str | None = None,
        default_task: str | None = None,
        task_prompts: dict[str, str] | None = None,
        entity_label: str = "markdown",
        meter_pages: bool = False,
        task_labels: dict[str, str] | None = None,
        max_concurrent_requests: int = 4,
        disable_cuda_graph: bool = False,
        attention_backend: str | None = None,
        extra_launch_args: list[str] | None = None,
        extra_env: dict[str, str] | None = None,
        startup_timeout_s: float | None = None,
        **kwargs: Any,
    ) -> None:
        compat_dir = Path(__file__).with_name("_compat")
        child_env = dict(extra_env or {})
        inherited_pythonpath = child_env.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
        child_env["PYTHONPATH"] = os.pathsep.join(path for path in (str(compat_dir), inherited_pythonpath) if path)

        super().__init__(
            model_name_or_path=str(model_name_or_path),
            max_seq_length=max_seq_length,
            mem_fraction_static=mem_fraction_static,
            compute_precision=compute_precision,
            trust_remote_code=trust_remote_code,
            revision=revision,
            served_model_name=served_model_name,
            disable_cuda_graph=disable_cuda_graph,
            attention_backend=attention_backend,
            grammar_backend=None,
            extra_launch_args=extra_launch_args,
            extra_env=child_env,
            startup_timeout_s=startup_timeout_s,
            **kwargs,
        )
        self._max_new_tokens = max_new_tokens
        self._num_beams = num_beams
        self._processor_use_fast = processor_use_fast
        self._system_prompt = system_prompt
        self._default_prompt = default_prompt
        self._default_task = default_task
        self._task_prompts = dict(task_prompts or {})
        self._entity_label = entity_label
        self._meter_pages = meter_pages
        self._task_labels = dict(task_labels or {})
        self._max_concurrent_requests = max(1, max_concurrent_requests)
        self._processor: Any = None
        self._request_loop: asyncio.AbstractEventLoop | None = None
        self._request_loop_thread: threading.Thread | None = None

    @classmethod
    def create_for_device(cls, device: str, **kwargs: Any) -> SGLangVisionExtractAdapter:
        """Keep the OCR profile CUDA-only instead of applying the MLX text swap."""
        if not device.startswith("cuda"):
            msg = f"SGLangVisionExtractAdapter requires CUDA, got {device!r}"
            raise ValueError(msg)
        return cls(**kwargs)

    @property
    def capabilities(self) -> ModelCapabilities:
        """Expose only the extraction surface declared by the model config."""
        return ModelCapabilities(inputs=["image"], outputs=["json"])

    @property
    def dims(self) -> ModelDims:
        return ModelDims()

    def load(self, device: str) -> None:
        """Load the pinned processor locally before starting the SGLang engine."""
        if not device.startswith("cuda"):
            msg = f"SGLangVisionExtractAdapter requires CUDA, got {device!r}"
            raise ValueError(msg)

        from transformers import AutoProcessor

        processor_dir = self._resolve_processor_dir()
        processor_kwargs: dict[str, Any] = {"trust_remote_code": self._trust_remote_code}
        if self._processor_use_fast is not None:
            processor_kwargs["use_fast"] = self._processor_use_fast
        self._processor = AutoProcessor.from_pretrained(processor_dir, **processor_kwargs)

        try:
            super().load(device)
            self._start_request_loop()
        except BaseException:
            self._processor = None
            super().unload()
            raise

    def _resolve_processor_dir(self) -> str:
        """Resolve the processor to the same pinned snapshot served by SGLang."""
        path = Path(self._model_name_or_path)
        if path.is_dir():
            return str(path)
        download_kwargs: dict[str, Any] = {}
        if self._revision is not None:
            download_kwargs["revision"] = self._revision
        return snapshot_download(self._model_name_or_path, **download_kwargs)

    def _start_request_loop(self) -> None:
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def run() -> None:
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()
            loop.close()

        thread = threading.Thread(
            target=run,
            name=f"sglang-extract-{self._served_model_name}",
            daemon=True,
        )
        thread.start()
        if not ready.wait(timeout=5):
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5)
            msg = "Timed out starting the SGLang extraction request loop"
            raise RuntimeError(msg)
        self._request_loop = loop
        self._request_loop_thread = thread

    def unload(self) -> None:
        loop = self._request_loop
        thread = self._request_loop_thread
        self._request_loop = None
        self._request_loop_thread = None

        if loop is not None and loop.is_running():
            if self._http_client is not None:
                try:
                    asyncio.run_coroutine_threadsafe(self.aclose_client(), loop).result(timeout=3)
                except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
                    logger.warning("Failed to close SGLang OCR HTTP client cleanly: %s", exc)
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5)
            if thread.is_alive():
                logger.warning("SGLang OCR request-loop thread did not stop within 5s")

        self._processor = None
        super().unload()

    async def aclose_client(self) -> None:
        """Close the shared client on the dedicated loop that owns it."""
        request_loop = self._request_loop
        if request_loop is None or not request_loop.is_running():
            await super().aclose_client()
            return

        current_loop = asyncio.get_running_loop()
        if current_loop is request_loop:
            await super().aclose_client()
            return

        future = asyncio.run_coroutine_threadsafe(super().aclose_client(), request_loop)
        await asyncio.wrap_future(future)

    def get_preprocessor(self) -> None:
        """SGLang owns multimodal preprocessing in its tokenizer process."""

    def count_input_images(self, items: list[Item]) -> list[int] | None:
        """The engine consumes the first image from each extraction item."""
        if self._meter_pages:
            return None
        return [1 if item.images else 0 for item in items]

    def extract(
        self,
        items: list[Item],
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
        prepared_items: list[Any] | None = None,
    ) -> ExtractOutput:
        del labels, output_schema, prepared_items
        if self._processor is None or self._request_loop is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        runtime = options or {}
        max_new_tokens = int(runtime.get("max_new_tokens", self._max_new_tokens))
        num_beams = int(runtime.get("num_beams", self._num_beams))
        if num_beams != 1:
            msg = "SGLangVisionExtractAdapter supports greedy decoding only (num_beams=1)"
            raise ValueError(msg)

        if self._meter_pages:
            for item in items:
                if item.images is None or len(item.images) != 1:
                    msg = "Page-metered SGLang OCR requires exactly one image per item"
                    raise InvalidMediaError(msg)

        images: list[ImageInput] = []
        for item in items:
            if not item.images:
                raise ValueError(_ERR_NO_IMAGES)
            images.append(item.images[0])

        prompt_text, entity_label = self._resolve_prompt_and_label(instruction, runtime)
        prompt = self._build_prompt(prompt_text)
        future = asyncio.run_coroutine_threadsafe(
            self._extract_async(prompt, images, max_new_tokens=max_new_tokens),
            self._request_loop,
        )
        results = future.result()
        entities = [[Entity(text=result.text.strip(), label=entity_label, score=1.0)] for result in results]
        return ExtractOutput(entities=entities, pages=[1 for _ in results] if self._meter_pages else None)

    def _resolve_prompt_and_label(
        self,
        instruction: str | None,
        runtime: dict[str, Any],
    ) -> tuple[str | None, str]:
        task_value = runtime.get("task", self._default_task)
        task = str(task_value) if task_value is not None else None
        if self._task_prompts and task not in self._task_prompts:
            msg = f"task {task!r} must be one of {tuple(self._task_prompts)}"
            raise ValueError(msg)

        task_prompt = self._task_prompts.get(task) if task is not None else None
        prompt = instruction or task_prompt or self._default_prompt
        label = self._task_labels.get(task, self._entity_label) if task is not None else self._entity_label
        return prompt, label

    def _build_prompt(self, prompt_text: str | None) -> str:
        user_content: list[dict[str, str]] = [{"type": "image"}]
        if prompt_text is not None:
            user_content.append({"type": "text", "text": prompt_text})
        messages: list[dict[str, Any]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": user_content})
        return self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    async def _extract_async(
        self,
        prompt: str,
        images: list[ImageInput],
        *,
        max_new_tokens: int,
    ) -> list[GenerationResult]:
        semaphore = asyncio.Semaphore(self._max_concurrent_requests)

        async def generate_one(image: ImageInput) -> GenerationResult:
            async with semaphore:
                return await collect_generation(
                    self.generate(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=0.0,
                        top_p=1.0,
                        images=[image],
                    )
                )

        results = await asyncio.gather(*(generate_one(image) for image in images))
        for result in results:
            if result.finish_reason == "error":
                msg = "SGLang OCR generation failed"
                raise RuntimeError(msg)
        return results
