"""Benchmark observation and fixed KV limits around production inference."""

from dataclasses import asdict
from pathlib import Path

from sie_server.adapters.sglang._memory import SGLangMemoryBudget
from sie_server.adapters.sglang.embedding import SGLangEmbeddingAdapter
from sie_server.adapters.sglang.generation import SGLangGenerationAdapter

KV_CAPACITY = 4096
STATIC_MEMORY_FRACTION = 0.5


class CappedEmbeddingMemoryBudget(SGLangMemoryBudget):
    def launch_args(self, device_index: int) -> list[str]:
        if self.mem_fraction_static != STATIC_MEMORY_FRACTION or self.kv_budget_tokens != KV_CAPACITY:
            raise ValueError("The benchmark requires fraction 0.5 and a 4096-token KV limit")
        return [*super().launch_args(device_index), "--max-total-tokens", str(KV_CAPACITY)]


class CappedEmbeddingAdapter(SGLangEmbeddingAdapter):
    def load(self, device: str) -> None:
        self._memory_budget = CappedEmbeddingMemoryBudget(**asdict(self._memory_budget))
        super().load(device)


class ObservedGenerationAdapter(SGLangGenerationAdapter):
    def _compat_pythonpath_entries(self) -> tuple[str, ...]:
        source = Path(__file__).resolve()
        return (str(source.parent / "_hooks"), str(source.parents[3]), *super()._compat_pythonpath_entries())
