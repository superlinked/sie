"""Grouped FA2 training; synchronization keeps all reference scoring on FP32 SDPA."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import random
import time
from dataclasses import asdict


def grouped_windows(encoded, micro_batch=4, grad_accum=4):
    size = micro_batch * grad_accum
    for start in range(0, len(encoded), size):
        window = list(encoded[start:start + size])
        if len(window) % micro_batch == 0:
            window.sort(key=lambda item: len(item["ids"]))
        yield window


def encoded_epoch(lt, tokenizer, items, config, epoch, order_rng, max_len, head_max_len, trace=None):
    ordered = list(range(len(items)))
    random.Random(config.seed + epoch).shuffle(ordered)
    encoded, assignment = [], []
    for index in ordered:
        item = items[index]
        options = lt.draw_option_order(item, order_rng, config.shuffle_options)
        assignment.append([index, options])
        encoded.append(lt.encode_item(tokenizer, item, max_len, head_max_len, options, False))
    if trace is not None:
        trace["original_item_option_assignment_sha256"] = hashlib.sha256(
            json.dumps(assignment, separators=(",", ":")).encode()
        ).hexdigest()
    return encoded


def require_sdpa_fp32(model):
    import torch

    if model.encoder.config._attn_implementation != "sdpa":
        raise ValueError("Selection/export/scoring must use the SDPA model")
    if any(p.dtype != torch.float32 for p in model.parameters() if p.is_floating_point()):
        raise ValueError("Selection/export/scoring parameters must be FP32")
    if torch.is_autocast_enabled("cuda"):
        raise ValueError("Reference scoring cannot run inside CUDA autocast")


def synchronized_state(source, destination):
    import torch

    state = source.state_dict()
    destination.load_state_dict(state, strict=True)
    target = destination.state_dict()
    if set(state) != set(target) or any(not torch.equal(value, target[key]) for key, value in state.items()):
        raise ValueError("FA2 to SDPA synchronization changed checkpoint values")


def training_copy(model, device):
    import torch
    from peft import get_peft_model
    from transformers import AutoModel
    from transformers.modeling_utils import no_init_weights

    require_sdpa_fp32(model)
    if set(model.encoder.peft_config) != {"default"}:
        raise ValueError("Expected the single frozen default LoRA adapter")
    configuration = copy.deepcopy(model.encoder.config)
    configuration.reference_compile = False
    lora = copy.deepcopy(model.encoder.peft_config["default"])
    trained = copy.deepcopy(model)
    del trained.encoder
    with no_init_weights():
        encoder = AutoModel.from_config(configuration, attn_implementation="flash_attention_2")
    encoder = get_peft_model(encoder, lora)
    encoder.load_state_dict(model.encoder.state_dict(), strict=True)
    trained.encoder = encoder.to(device)
    synchronized_state(model, trained)
    if trained.encoder.config._attn_implementation != "flash_attention_2":
        raise ValueError("FA2 construction did not select the requested backend")
    if hasattr(trained.encoder, "gradient_checkpointing_disable"):
        trained.encoder.gradient_checkpointing_disable()
    trained.head_checkpointing = False
    trained.to(device).train()
    if any(p.dtype != torch.float32 for p in trained.parameters() if p.is_floating_point()):
        raise ValueError("Training parameters changed dtype")
    return trained


def parameter_fingerprint(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_model(lt, model, tokenizer, items, config, device, max_len, head_max_len,
                on_epoch_end=None, parallel=False, *, phase, record):
    import torch

    config.validate()
    if (not items or parallel or config.freeze_encoder or config.loss != "soft-ce"
            or config.micro_batch != 4 or config.grad_accum != 4 or config.amp not in (None, True)):
        raise ValueError("Unsupported configuration for the versioned fast recipe")
    require_sdpa_fp32(model)
    started = time.perf_counter()
    trained = training_copy(model, device)
    torch.cuda.synchronize()
    copied_seconds = time.perf_counter() - started
    initial = parameter_fingerprint(trained)

    class CountedAdamW(torch.optim.AdamW):
        actual_steps = 0

        def step(self, closure=None):
            result = super().step(closure)
            self.actual_steps += 1
            return result

    groups = [
        {"params": [p for n, p in trained.named_parameters() if n.startswith("encoder.") and p.requires_grad],
         "lr": config.encoder_lr},
        {"params": [p for n, p in trained.named_parameters() if not n.startswith("encoder.") and p.requires_grad],
         "lr": config.head_lr},
    ]
    optimizer = CountedAdamW(groups, weight_decay=config.weight_decay)
    steps_per_epoch = math.ceil(len(items) / config.micro_batch)
    updates = max(1, math.ceil(steps_per_epoch / config.grad_accum) * config.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=config.min_lr)
    scaler = torch.amp.GradScaler("cuda")
    torch.manual_seed(config.seed)
    rng = {"cpu_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
           "cuda_sha256": [hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
                           for x in torch.cuda.get_rng_state_all()]}
    order_rng = random.Random(config.seed)
    parameters = [p for group in groups for p in group["params"]]
    history, epochs = [], []
    effective = dict(asdict(config), gradient_checkpointing=False)
    record({"event": "phase_start", "phase": phase, "effective_config": effective,
            "initial_trainable_sha256": initial, "model_copy_seconds": copied_seconds,
            "initial_rng": rng,
            "scheduler_attempted_windows": updates, "items_per_epoch": len(items)})
    try:
        for epoch in range(config.epochs):
            trained.train()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            epoch_started = time.perf_counter()
            assignment = {}
            encoded = encoded_epoch(lt, tokenizer, items, config, epoch, order_rng, max_len, head_max_len, assignment)
            total, micro_steps, actual_tokens, padded_tokens = 0.0, 0, 0, 0
            windows, before_steps = [], optimizer.actual_steps
            optimizer.zero_grad(set_to_none=True)
            for window_index, window in enumerate(grouped_windows(encoded, config.micro_batch, config.grad_accum)):
                chunks = [window[start:start + config.micro_batch]
                          for start in range(0, len(window), config.micro_batch)]
                for chunk in chunks:
                    batch = lt.collate_items([chunk], tokenizer.pad_token_id)
                    lengths = [len(item["ids"]) for item in chunk]
                    actual_tokens += sum(lengths)
                    padded_tokens += len(lengths) * max(lengths)
                    logits = lt._forward(trained, batch, device, True, False)
                    loss = lt.soft_ce_loss(logits, batch["target"].to(device), batch["marker_mask"].to(device))
                    scaler.scale(loss / len(chunks)).backward()
                    total += loss.item()
                    micro_steps += 1
                scaler.unscale_(optimizer)
                norm = float(torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip))
                scale_before, steps_before = scaler.get_scale(), optimizer.actual_steps
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                windows.append({"window": window_index, "gradient_norm": norm if math.isfinite(norm) else None,
                                "scale_before": scale_before, "scale_after": scaler.get_scale(),
                                "optimizer_step": optimizer.actual_steps != steps_before})
            torch.cuda.synchronize()
            train_seconds = time.perf_counter() - epoch_started
            loss = total / micro_steps
            if not math.isfinite(loss) or any(not torch.isfinite(p).all() for p in parameters):
                raise ValueError("Nonfinite loss or final trainable parameter")
            history.append(loss)
            sync_started = time.perf_counter()
            synchronized_state(trained, model)
            torch.cuda.synchronize()
            sync_seconds = time.perf_counter() - sync_started
            measured = {
                "event": "epoch", "phase": phase, "epoch": epoch + 1, "loss": loss,
                "training_seconds": train_seconds, "synchronization_seconds": sync_seconds,
                "decision_items": len(encoded), "micro_batches": micro_steps,
                "attempted_optimizer_windows": len(windows),
                "actual_optimizer_steps": optimizer.actual_steps - before_steps,
                "skipped_optimizer_windows": [w["window"] for w in windows if not w["optimizer_step"]],
                "actual_tokens": actual_tokens, "padded_tokens": padded_tokens,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "windows": windows,
                **assignment,
            }
            callback_started = time.perf_counter()
            if on_epoch_end is not None:
                on_epoch_end(epoch, loss)
            measured["callback_seconds_including_reference_validation_and_persistence"] = (
                time.perf_counter() - callback_started
            )
            epochs.append(measured)
            record(measured)
            del encoded, batch, logits, loss
        model.eval()
        record({"event": "phase_complete", "phase": phase, "elapsed_seconds": time.perf_counter() - started,
                "epochs": len(epochs), "actual_optimizer_steps": optimizer.actual_steps,
                "attempted_optimizer_windows": sum(e["attempted_optimizer_windows"] for e in epochs)})
        return history
    finally:
        del trained, optimizer, scheduler, scaler, parameters, groups
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
