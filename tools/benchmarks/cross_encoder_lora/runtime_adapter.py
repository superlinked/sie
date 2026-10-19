"""Untimed observation of the explicitly selected runtime-LoRA production class."""

from sie_server.adapters.cross_encoder.runtime_lora import CrossEncoderLoraAdapter

from tools.benchmarks.cross_encoder_lora.adapter import ObservedCrossEncoder


class ObservedRuntimeCrossEncoder(ObservedCrossEncoder, CrossEncoderLoraAdapter):
    production_adapter_class = CrossEncoderLoraAdapter
