from __future__ import annotations

import argparse
import importlib
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from model.baselines import build_rvq_projector_from_model_config
from model.components.moss import MockMossCodec
from model.config import TrainAppConfig, load_train_config, save_json, to_dict
from model.io import load_records, record_to_stream_sample
from model.llm import GPTPhase1Model
from model.schemas import StreamSample
from model.streaming_tts import (
    RealtimeChunkedTTSPipeline,
    TimestampPassthroughChunker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train realtime TTS phase-1 with explicit config.")
    parser.add_argument("--config", type=str, default=None, help="Path to train JSON config.")
    parser.add_argument("--input-path", type=str, default=None, help="Override data.input_path.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Override data.dataset_name.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override train.output_dir.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override train.max_steps.")
    parser.add_argument("--device", type=str, default=None, help="Override train.device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_config = resolve_train_config(args)
    set_seed(app_config.train.seed)
    output_dir = Path(app_config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_config.json", to_dict(app_config))

    codec = build_component(app_config.data.codec, default_factory=lambda: MockMossCodec(
        num_layers=app_config.model.num_layers,
        embedding_dim=app_config.model.input_dim,
        vocab_size=app_config.model.vocab_size,
    ))
    chunker = build_component(app_config.data.chunker, default_factory=TimestampPassthroughChunker)

    llm_config = app_config.model.build_llm_config()
    layer0_generator = GPTPhase1Model(llm_config).to(app_config.train.device)
    rvq_projector = build_rvq_projector_from_model_config(app_config.model).to(app_config.train.device)
    optimizer = torch.optim.AdamW(
        [
            {"params": layer0_generator.parameters(), "lr": app_config.train.learning_rate},
            {"params": rvq_projector.parameters(), "lr": app_config.train.postnet_learning_rate},
        ]
    )
    print(
        json.dumps(
            {
                "phase1_params": layer0_generator.num_parameters(),
                "postnet_params": sum(p.numel() for p in rvq_projector.parameters()),
                "device": app_config.train.device,
                "config_path": args.config,
            }
        )
    )

    pipeline = RealtimeChunkedTTSPipeline(
        chunker=chunker,
        codec=codec,
        layer0_generator=layer0_generator,
        rvq_projector=rvq_projector,
        config=app_config.model.build_pipeline_config(app_config.train.seed),
    )

    metrics_path = output_dir / "train_metrics.jsonl"

    step = 0
    start_time = time.perf_counter()
    while step < app_config.train.max_steps:
        records = load_records(
            input_path=app_config.data.input_path,
            dataset_name=app_config.data.dataset_name,
            dataset_names=app_config.data.dataset_names,
            split=app_config.data.split,
        )
        made_progress = False
        for record in records:
            current_step = step + 1
            sample = record_to_stream_sample(record)
            target_sample = build_target_sample(sample)
            if not target_sample.audio_path:
                continue

            training_example = pipeline.prepare_training_example(sample)
            target_sequence = layer0_generator.build_training_sequence(
                [(chunk.text, chunk.layer0_codes) for chunk in training_example.chunks]
            )
            target_layer0_embeddings = torch.cat(
                [chunk.layer0_embeddings for chunk in training_example.chunks],
                dim=0,
            ).float().to(app_config.train.device)
            target_acoustic = torch.cat(
                [chunk.continuous_rvq_layers for chunk in training_example.chunks],
                dim=0,
            ).float().to(app_config.train.device)

            train_output = layer0_generator.forward_train(training_example.prompt, target_sequence)
            target_labels = target_sequence[1:].to(app_config.train.device)
            aligned_count = min(train_output.logits.shape[0], target_labels.shape[0])
            code_loss_mask = train_output.code_loss_mask[:aligned_count]
            if code_loss_mask.any():
                code_loss = F.cross_entropy(
                    train_output.logits[:aligned_count][code_loss_mask],
                    target_labels[:aligned_count][code_loss_mask],
                )
            else:
                code_loss = train_output.logits[:aligned_count].sum() * 0.0
            acoustic_pred = rvq_projector(train_output.audio_hidden_states, target_layer0_embeddings)
            acoustic_count = min(acoustic_pred.shape[0], target_acoustic.shape[0])
            acoustic_target = target_acoustic[:acoustic_count]
            acoustic_l1 = F.smooth_l1_loss(acoustic_pred, acoustic_target)
            acoustic_cosine = 1.0 - F.cosine_similarity(
                acoustic_pred.reshape(acoustic_count, -1),
                acoustic_target.reshape(acoustic_count, -1),
                dim=-1,
            ).mean()
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
            loss = (
                app_config.train.code_loss_weight * code_loss
                + acoustic_scale * app_config.train.acoustic_l1_weight * acoustic_l1
                + acoustic_scale * app_config.train.acoustic_cosine_weight * acoustic_cosine
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            made_progress = True
            metric = {
                "step": step,
                "sample_id": sample.sample_id,
                "chunk_mode": training_example.chunk_mode,
                "selected_text": (
                    training_example.selected_span.text
                    if training_example.selected_span
                    else sample.ref_text or sample.text
                ),
                "num_chunks": len(training_example.chunks),
                "code_loss": float(code_loss.detach().cpu()),
                "acoustic_l1": float(acoustic_l1.detach().cpu()),
                "acoustic_cosine": float(acoustic_cosine.detach().cpu()),
                "acoustic_scale": float(acoustic_scale),
                "postnet_lr_scale": float(postnet_lr_scale),
                "llm_lr": float(optimizer.param_groups[0]["lr"]),
                "postnet_lr": float(optimizer.param_groups[1]["lr"]),
                "loss": float(loss.detach().cpu()),
            }
            append_jsonl(metrics_path, metric)
            if step % app_config.train.log_every == 0 or step == 1:
                print(json.dumps(metric, ensure_ascii=False))
            if step % app_config.train.save_every == 0:
                save_checkpoint(
                    output_dir / f"checkpoint_step_{step}.pt",
                    layer0_generator,
                    rvq_projector,
                    app_config,
                    step,
                )
            if step >= app_config.train.max_steps:
                break
        if not made_progress:
            break

    save_checkpoint(output_dir / "last.pt", layer0_generator, rvq_projector, app_config, step)
    elapsed = time.perf_counter() - start_time
    print(
        json.dumps(
            {
                "step": step,
                "elapsed_sec": elapsed,
                "sec_per_step": elapsed / max(step, 1),
                "checkpoint": str(output_dir / "last.pt"),
            }
        )
    )


def build_target_sample(sample: StreamSample) -> StreamSample:
    return StreamSample(
        sample_id=f"{sample.sample_id}:target",
        text=sample.text,
        audio_path=sample.audio_path or sample.ref_audio_path,
        ref_text=sample.ref_text,
        ref_audio_path=sample.ref_audio_path,
        timestamps=sample.timestamps,
        metadata=sample.metadata,
    )


def build_component(spec: str | None, default_factory):
    if not spec:
        return default_factory()
    module_name, attr_name = spec.split(":", maxsplit=1)
    attr = getattr(importlib.import_module(module_name), attr_name)
    return attr() if callable(attr) else attr


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_checkpoint(
    path: Path,
    layer0_generator: torch.nn.Module,
    rvq_projector: torch.nn.Module,
    app_config: TrainAppConfig,
    step: int,
) -> None:
    torch.save(
        {
            "step": step,
            "config": to_dict(app_config.model),
            "train_config": to_dict(app_config.train),
            "layer0_generator": layer0_generator.state_dict(),
            "rvq_projector": rvq_projector.state_dict(),
        },
        path,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_train_config(args: argparse.Namespace) -> TrainAppConfig:
    config = load_train_config(args.config)
    if args.input_path is not None:
        config.data.input_path = args.input_path
    if args.dataset_name is not None:
        config.data.dataset_name = args.dataset_name
    if args.output_dir is not None:
        config.train.output_dir = args.output_dir
    if args.max_steps is not None:
        config.train.max_steps = args.max_steps
    if args.device is not None:
        config.train.device = args.device
    return config


def compute_acoustic_scale(step: int, llm_only_steps: int, acoustic_ramp_steps: int) -> float:
    if step <= llm_only_steps:
        return 0.0
    if acoustic_ramp_steps <= 0:
        return 1.0
    progress = (step - llm_only_steps) / acoustic_ramp_steps
    return max(0.0, min(1.0, progress))


def interpolate_scale(progress: float, start: float, end: float) -> float:
    return start + (end - start) * progress


if __name__ == "__main__":
    main()
