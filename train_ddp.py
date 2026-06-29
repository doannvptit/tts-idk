from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from model.baselines import build_rvq_projector_from_model_config
from model.components.moss import MockMossCodec
from model.config import TrainAppConfig, load_train_config, save_json, to_dict
from model.io import load_records, record_to_stream_sample
from model.llm import GPTPhase1Model
from model.schemas import StreamSample
from model.streaming_tts import RealtimeChunkedTTSPipeline, TimestampPassthroughChunker


class TTSLossModule(nn.Module):
    def __init__(self, layer0_generator: GPTPhase1Model, rvq_projector: nn.Module) -> None:
        super().__init__()
        self.layer0_generator = layer0_generator
        self.rvq_projector = rvq_projector

    def forward(
        self,
        prompt,
        target_sequence: torch.Tensor,
        target_layer0_embeddings: torch.Tensor,
        target_acoustic: torch.Tensor,
        acoustic_scale: float,
        code_loss_weight: float,
        acoustic_l1_weight: float,
        acoustic_cosine_weight: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        train_output = self.layer0_generator.forward_train(prompt, target_sequence)
        target_labels = target_sequence[1:].to(train_output.logits.device)
        aligned_count = min(train_output.logits.shape[0], target_labels.shape[0])
        code_loss_mask = train_output.code_loss_mask[:aligned_count]
        if code_loss_mask.any():
            code_loss = F.cross_entropy(
                train_output.logits[:aligned_count][code_loss_mask],
                target_labels[:aligned_count][code_loss_mask],
            )
        else:
            code_loss = train_output.logits[:aligned_count].sum() * 0.0

        acoustic_pred = self.rvq_projector(train_output.audio_hidden_states, target_layer0_embeddings)
        acoustic_count = min(acoustic_pred.shape[0], target_acoustic.shape[0])
        acoustic_target = target_acoustic[:acoustic_count]
        acoustic_l1 = F.smooth_l1_loss(acoustic_pred, acoustic_target)
        acoustic_cosine = 1.0 - F.cosine_similarity(
            acoustic_pred.reshape(acoustic_count, -1),
            acoustic_target.reshape(acoustic_count, -1),
            dim=-1,
        ).mean()
        loss = (
            code_loss_weight * code_loss
            + acoustic_scale * acoustic_l1_weight * acoustic_l1
            + acoustic_scale * acoustic_cosine_weight * acoustic_cosine
        )
        return loss, {
            "code_loss": code_loss.detach(),
            "acoustic_l1": acoustic_l1.detach(),
            "acoustic_cosine": acoustic_cosine.detach(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP train realtime TTS with timing breakdown.")
    parser.add_argument("--config", type=str, default="configs/train.default.json")
    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Optimizer steps per rank.")
    parser.add_argument("--chunker", type=str, default=None, help="Override data.chunker, e.g. model.chunkers:ChunkFormerTimestampChunker.")
    parser.add_argument("--codec", type=str, default=None, help="Override data.codec. Use none for MockMossCodec.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--profile-every", type=int, default=1)
    parser.add_argument(
        "--skip-python-finalizers",
        action="store_true",
        help="Exit with os._exit(0) after flushing output. Useful for HF streaming/audio finalizer crashes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ddp = init_distributed()
    rank = ddp["rank"]
    local_rank = ddp["local_rank"]
    world_size = ddp["world_size"]
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    app_config = resolve_train_config(args)
    app_config.train.device = str(device)
    set_seed(app_config.train.seed + rank)
    output_dir = Path(app_config.train.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(output_dir / "resolved_config.json", to_dict(app_config))
    barrier()

    codec = build_component(
        app_config.data.codec,
        default_factory=lambda: MockMossCodec(
            num_layers=app_config.model.num_layers,
            embedding_dim=app_config.model.input_dim,
            vocab_size=app_config.model.vocab_size,
        ),
        device=str(device),
    )
    chunker = build_component(
        app_config.data.chunker,
        default_factory=TimestampPassthroughChunker,
        device=str(device),
    )

    layer0_generator = GPTPhase1Model(app_config.model.build_llm_config()).to(device)
    rvq_projector = build_rvq_projector_from_model_config(app_config.model).to(device)
    train_module = TTSLossModule(layer0_generator, rvq_projector).to(device)
    if world_size > 1:
        train_module = DistributedDataParallel(
            train_module,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(
        [
            {"params": unwrap(train_module).layer0_generator.parameters(), "lr": app_config.train.learning_rate},
            {"params": unwrap(train_module).rvq_projector.parameters(), "lr": app_config.train.postnet_learning_rate},
        ]
    )
    if rank == 0:
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "device": str(device),
                    "phase1_params": unwrap(train_module).layer0_generator.num_parameters(),
                    "postnet_params": sum(p.numel() for p in unwrap(train_module).rvq_projector.parameters()),
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                }
            )
        )

    pipeline = RealtimeChunkedTTSPipeline(
        chunker=chunker,
        codec=codec,
        layer0_generator=unwrap(train_module).layer0_generator,
        rvq_projector=unwrap(train_module).rvq_projector,
        config=app_config.model.build_pipeline_config(app_config.train.seed + rank),
    )

    metrics_path = output_dir / "train_ddp_metrics.jsonl"
    step = 0
    consumed = 0
    skipped = 0
    timings = new_timings()
    started = time.perf_counter()
    records = sharded_records(app_config, rank, world_size)
    accum = max(1, args.gradient_accumulation_steps)

    optimizer.zero_grad(set_to_none=True)
    while step < app_config.train.max_steps:
        micro_metrics: list[dict[str, float]] = []
        made_progress = False
        for micro_step in range(accum):
            sample_result = next_train_sample(records, pipeline, app_config, device, timings)
            if sample_result is None:
                skipped += 1
                continue
            consumed += 1
            made_progress = True
            training_example, target_sequence, target_layer0_embeddings, target_acoustic, sample = sample_result
            current_step = step + 1
            acoustic_scale = compute_acoustic_scale(
                step=current_step,
                llm_only_steps=app_config.train.llm_only_steps,
                acoustic_ramp_steps=app_config.train.acoustic_ramp_steps,
            )
            postnet_lr_scale = interpolate_scale(
                acoustic_scale,
                start=app_config.train.postnet_start_lr_scale,
                end=app_config.train.postnet_final_lr_scale,
            )
            optimizer.param_groups[0]["lr"] = app_config.train.learning_rate
            optimizer.param_groups[1]["lr"] = app_config.train.postnet_learning_rate * postnet_lr_scale

            sync(device)
            t0 = time.perf_counter()
            no_sync_context = (
                train_module.no_sync()
                if isinstance(train_module, DistributedDataParallel) and micro_step < accum - 1
                else nullcontext()
            )
            with no_sync_context:
                loss, losses = train_module(
                    training_example.prompt,
                    target_sequence,
                    target_layer0_embeddings,
                    target_acoustic,
                    acoustic_scale,
                    app_config.train.code_loss_weight,
                    app_config.train.acoustic_l1_weight,
                    app_config.train.acoustic_cosine_weight,
                )
                (loss / accum).backward()
            sync(device)
            timings["forward_backward"] += time.perf_counter() - t0
            micro_metrics.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "code_loss": float(losses["code_loss"].cpu()),
                    "acoustic_l1": float(losses["acoustic_l1"].cpu()),
                    "acoustic_cosine": float(losses["acoustic_cosine"].cpu()),
                    "acoustic_scale": float(acoustic_scale),
                    "postnet_lr_scale": float(postnet_lr_scale),
                    "num_chunks": float(len(training_example.chunks)),
                    "sample_id": sample.sample_id,
                }
            )

        if not made_progress:
            break

        sync(device)
        t0 = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        sync(device)
        timings["optimizer_step"] += time.perf_counter() - t0

        step += 1
        should_profile = step == 1 or step % max(1, args.profile_every) == 0
        reduced_timings = reduce_timings(timings, device, world_size) if should_profile else {}
        if rank == 0 and (step % app_config.train.log_every == 0 or step == 1):
            metric = {
                "step": step,
                "global_samples": step * accum * world_size,
                "rank0_consumed": consumed,
                "rank0_skipped": skipped,
                **average_micro_metrics(micro_metrics),
                "timing_sec": reduced_timings,
                "timing_pct": timing_percentages(reduced_timings),
                "elapsed_sec": time.perf_counter() - started,
            }
            print(json.dumps(metric, ensure_ascii=False))
            append_jsonl(metrics_path, metric)
        if rank == 0 and step % app_config.train.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint_step_{step}.pt", train_module, app_config, step)

    if rank == 0:
        final_timings = reduce_timings(timings, device, world_size)
        save_checkpoint(output_dir / "last.pt", train_module, app_config, step)
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "step": step,
                    "global_samples": step * accum * world_size,
                    "elapsed_sec": elapsed,
                    "sec_per_step": elapsed / max(step, 1),
                    "sec_per_global_sample": elapsed / max(step * accum * world_size, 1),
                    "timing_sec": final_timings,
                    "checkpoint": str(output_dir / "last.pt"),
                },
                ensure_ascii=False,
            )
        )
    else:
        _ = reduce_timings(timings, device, world_size)
    cleanup_distributed()
    if args.skip_python_finalizers:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def next_train_sample(records, pipeline, app_config, device, timings):
    while True:
        t0 = time.perf_counter()
        try:
            record = next(records)
        except StopIteration:
            return None
        timings["stream_next"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        sample = record_to_stream_sample(record)
        timings["record_to_sample"] += time.perf_counter() - t0
        if not (sample.audio_path or sample.waveform is not None):
            continue

        t0 = time.perf_counter()
        training_example = pipeline.prepare_training_example(sample)
        timings["preprocess_chunk_moss"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        target_sequence = pipeline.layer0_generator.build_training_sequence(
            [(chunk.text, chunk.layer0_codes) for chunk in training_example.chunks]
        )
        target_layer0_embeddings = torch.cat(
            [chunk.layer0_embeddings for chunk in training_example.chunks],
            dim=0,
        ).float().to(device)
        target_acoustic = torch.cat(
            [chunk.continuous_rvq_layers for chunk in training_example.chunks],
            dim=0,
        ).float().to(device)
        timings["build_batch"] += time.perf_counter() - t0
        return training_example, target_sequence, target_layer0_embeddings, target_acoustic, sample
    return None


def sharded_records(app_config: TrainAppConfig, rank: int, world_size: int):
    records = load_records(
        input_path=app_config.data.input_path,
        dataset_name=app_config.data.dataset_name,
        dataset_names=app_config.data.dataset_names,
        split=app_config.data.split,
    )
    for index, record in enumerate(records):
        if index % world_size == rank:
            yield record


def init_distributed() -> dict[str, int]:
    if "RANK" not in os.environ:
        return {"rank": 0, "local_rank": 0, "world_size": 1}
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return {
        "rank": int(os.environ["RANK"]),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        "world_size": int(os.environ["WORLD_SIZE"]),
    }


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def unwrap(module: nn.Module) -> TTSLossModule:
    return module.module if isinstance(module, DistributedDataParallel) else module


def build_component(spec: str | None, default_factory, device: str):
    if not spec:
        return default_factory()
    if spec.lower() == "none":
        return default_factory()
    module_name, attr_name = spec.split(":", maxsplit=1)
    attr = getattr(importlib.import_module(module_name), attr_name)
    if not callable(attr):
        return attr
    signature = inspect.signature(attr)
    if "device" in signature.parameters:
        return attr(device=device)
    return attr()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_checkpoint(path: Path, train_module: nn.Module, app_config: TrainAppConfig, step: int) -> None:
    module = unwrap(train_module)
    torch.save(
        {
            "step": step,
            "config": to_dict(app_config.model),
            "train_config": to_dict(app_config.train),
            "layer0_generator": module.layer0_generator.state_dict(),
            "rvq_projector": module.rvq_projector.state_dict(),
        },
        path,
    )


def resolve_train_config(args: argparse.Namespace) -> TrainAppConfig:
    config = load_train_config(args.config)
    if args.input_path is not None:
        config.data.input_path = args.input_path
    if args.dataset_name is not None:
        config.data.dataset_name = args.dataset_name
        config.data.dataset_names = []
    if args.output_dir is not None:
        config.train.output_dir = args.output_dir
    if args.max_steps is not None:
        config.train.max_steps = args.max_steps
    if args.chunker is not None:
        config.data.chunker = None if args.chunker.lower() == "none" else args.chunker
    if args.codec is not None:
        config.data.codec = None if args.codec.lower() == "none" else args.codec
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_acoustic_scale(step: int, llm_only_steps: int, acoustic_ramp_steps: int) -> float:
    if step <= llm_only_steps:
        return 0.0
    if acoustic_ramp_steps <= 0:
        return 1.0
    progress = (step - llm_only_steps) / acoustic_ramp_steps
    return max(0.0, min(1.0, progress))


def interpolate_scale(progress: float, start: float, end: float) -> float:
    return start + (end - start) * progress


def new_timings() -> dict[str, float]:
    return {
        "stream_next": 0.0,
        "record_to_sample": 0.0,
        "preprocess_chunk_moss": 0.0,
        "build_batch": 0.0,
        "forward_backward": 0.0,
        "optimizer_step": 0.0,
    }


def reduce_timings(timings: dict[str, float], device: torch.device, world_size: int) -> dict[str, float]:
    values = torch.tensor([timings[key] for key in timings], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= world_size
    return {key: float(value) for key, value in zip(timings, values.detach().cpu().tolist(), strict=True)}


def timing_percentages(timings: dict[str, float]) -> dict[str, float]:
    total = sum(timings.values())
    if total <= 0:
        return {key: 0.0 for key in timings}
    return {key: value / total * 100.0 for key, value in timings.items()}


def average_micro_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    numeric_keys = [
        "loss",
        "code_loss",
        "acoustic_l1",
        "acoustic_cosine",
        "acoustic_scale",
        "postnet_lr_scale",
        "num_chunks",
    ]
    if not items:
        return {}
    result = {key: sum(float(item[key]) for item in items) / len(items) for key in numeric_keys}
    result["sample_id"] = str(items[-1]["sample_id"])
    return result


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
