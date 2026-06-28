from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from model.baselines import build_rvq_projector_from_model_config
from model.components.moss import MockMossCodec
from model.config import InferenceAppConfig, ModelConfig, load_inference_config, save_json, to_dict
from model.io import load_records, record_to_stream_sample
from model.llm import GPTPhase1Model
from model.streaming_tts import (
    RealtimeChunkedTTSPipeline,
    TimestampPassthroughChunker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run realtime TTS inference with explicit config.")
    parser.add_argument("--config", type=str, default=None, help="Path to inference JSON config.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override inference.checkpoint.")
    parser.add_argument("--input-path", type=str, default=None, help="Override data.input_path.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Override data.dataset_name.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override inference.output_dir.")
    parser.add_argument("--device", type=str, default=None, help="Override inference.device.")
    parser.add_argument("--max-samples", type=int, default=None, help="Override inference.max_samples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_config = resolve_inference_config(args)
    output_dir = Path(app_config.inference.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_config.json", to_dict(app_config))

    codec = build_component(app_config.data.codec, default_factory=lambda: MockMossCodec(
        num_layers=app_config.model.num_layers,
        embedding_dim=app_config.model.input_dim,
        vocab_size=app_config.model.vocab_size,
    ))
    chunker = build_component(app_config.data.chunker, default_factory=TimestampPassthroughChunker)
    layer0_generator, rvq_projector = build_models(app_config)
    pipeline = RealtimeChunkedTTSPipeline(
        chunker=chunker,
        codec=codec,
        layer0_generator=layer0_generator,
        rvq_projector=rvq_projector,
        config=app_config.model.build_pipeline_config(app_config.inference.seed),
    )

    records = load_records(
        input_path=app_config.data.input_path,
        dataset_name=app_config.data.dataset_name,
        dataset_names=app_config.data.dataset_names,
        split=app_config.data.split,
    )
    manifest_path = output_dir / "predictions.jsonl"
    count = 0
    start_time = time.perf_counter()
    for record in records:
        sample = record_to_stream_sample(record)
        result = pipeline.process_sample(sample)
        sample_dir = output_dir / sample.sample_id.replace("/", "_")
        sample_dir.mkdir(parents=True, exist_ok=True)
        phase1_path = sample_dir / "layer0_codes.pt"
        phase2_path = sample_dir / "continuous_32_layers.pt"
        torch.save(torch.as_tensor(result.phase1.layer0_codes, dtype=torch.long), phase1_path)
        torch.save(result.phase2.continuous_32_layers.detach().cpu(), phase2_path)

        manifest = {
            "sample_id": sample.sample_id,
            "chunk_mode": result.chunk_mode,
            "voice_clone_text": result.prompt.voice_clone_text,
            "target_text": result.prompt.target_text,
            "selected_span": (
                {
                    "start_sec": result.selected_span.start_sec,
                    "end_sec": result.selected_span.end_sec,
                    "text": result.selected_span.text,
                }
                if result.selected_span
                else None
            ),
            "phase1_path": str(phase1_path),
            "phase2_path": str(phase2_path),
            "phase2_shape": list(result.phase2.continuous_32_layers.shape),
        }
        append_jsonl(manifest_path, manifest)
        print(json.dumps(manifest, ensure_ascii=False))

        count += 1
        if app_config.inference.max_samples and count >= app_config.inference.max_samples:
            break
    elapsed = time.perf_counter() - start_time
    print(json.dumps({"samples": count, "elapsed_sec": elapsed, "sec_per_sample": elapsed / max(count, 1)}))


def build_models(app_config: InferenceAppConfig) -> tuple[torch.nn.Module, torch.nn.Module]:
    if app_config.inference.checkpoint:
        state = torch.load(app_config.inference.checkpoint, map_location=app_config.inference.device)
        model_config = ModelConfig(**state["config"])
    else:
        state = None
        model_config = app_config.model
    llm_config = model_config.build_llm_config()
    layer0_generator = GPTPhase1Model(llm_config).to(app_config.inference.device)
    rvq_projector = build_rvq_projector_from_model_config(model_config).to(app_config.inference.device)
    if state is not None:
        layer0_generator.load_state_dict(state["layer0_generator"])
        if "rvq_projector" in state:
            rvq_projector.load_state_dict(state["rvq_projector"])
    layer0_generator.eval()
    rvq_projector.eval()
    return layer0_generator, rvq_projector


def build_component(spec: str | None, default_factory):
    if not spec:
        return default_factory()
    module_name, attr_name = spec.split(":", maxsplit=1)
    attr = getattr(importlib.import_module(module_name), attr_name)
    return attr() if callable(attr) else attr


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def resolve_inference_config(args: argparse.Namespace) -> InferenceAppConfig:
    config = load_inference_config(args.config)
    if args.checkpoint is not None:
        config.inference.checkpoint = args.checkpoint
    if args.input_path is not None:
        config.data.input_path = args.input_path
    if args.dataset_name is not None:
        config.data.dataset_name = args.dataset_name
    if args.output_dir is not None:
        config.inference.output_dir = args.output_dir
    if args.device is not None:
        config.inference.device = args.device
    if args.max_samples is not None:
        config.inference.max_samples = args.max_samples
    return config


if __name__ == "__main__":
    main()
