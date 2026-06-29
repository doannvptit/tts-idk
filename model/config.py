from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from model.components.postnet import ResidualPostNetConfig
from model.llm import GPTPhase1Config
from model.streaming_tts import PipelineConfig


@dataclass(slots=True)
class DataConfig:
    input_path: str | None = None
    dataset_name: str | None = None
    dataset_names: list[str] = field(default_factory=list)
    split: str = "train"
    chunker: str | None = None
    codec: str | None = None


@dataclass(slots=True)
class ModelConfig:
    tokenizer_path: str = "assets/vi_wikipedia_bbpe_2048_copy.json"
    input_dim: int = 8
    text_vocab_size: int = 2048
    audio_codebook_size: int = 1024
    audio_token_offset: int = 7
    vocab_size: int = 1024
    num_layers: int = 16
    llm_d_model: int = 640
    llm_heads: int = 10
    llm_layers: int = 10
    llm_mlp_ratio: int = 4
    llm_max_seq_len: int = 2048
    llm_max_audio_tokens: int = 256
    llm_max_reference_frames: int = 64
    llm_dropout: float = 0.1
    audio_tokens_per_text_token: int = 4
    postnet_hidden_layers: list[int] = field(default_factory=lambda: [2, 4, 6, 8, 10])
    postnet_model_dim: int = 640
    postnet_num_steps: int = 6
    postnet_dropout: float = 0.0
    multi_chunk_text_strategy: str = "replace_target_text"
    min_chunk_duration_sec: float = 0.05

    def build_llm_config(self) -> GPTPhase1Config:
        return GPTPhase1Config(
            tokenizer_path=self.tokenizer_path,
            text_vocab_size=self.text_vocab_size,
            audio_codebook_size=self.audio_codebook_size,
            audio_token_offset=self.audio_token_offset,
            max_seq_len=self.llm_max_seq_len,
            max_audio_tokens=self.llm_max_audio_tokens,
            d_model=self.llm_d_model,
            n_head=self.llm_heads,
            n_layer=self.llm_layers,
            mlp_ratio=self.llm_mlp_ratio,
            dropout=self.llm_dropout,
            reference_dim=self.input_dim,
            num_audio_layers=self.num_layers,
            max_reference_frames=self.llm_max_reference_frames,
            audio_tokens_per_text_token=self.audio_tokens_per_text_token,
            postnet_hidden_layers=self.postnet_hidden_layers,
        )

    def build_postnet_config(self) -> ResidualPostNetConfig:
        active_hidden_layers = [
            layer for layer in self.postnet_hidden_layers if 1 <= layer <= self.llm_layers
        ] or [self.llm_layers]
        return ResidualPostNetConfig(
            layer0_dim=self.input_dim,
            llm_hidden_dim=self.llm_d_model * len(active_hidden_layers),
            output_dim=self.input_dim,
            model_dim=self.postnet_model_dim,
            num_layers=self.num_layers,
            num_steps=self.postnet_num_steps,
            dropout=self.postnet_dropout,
        )

    def build_pipeline_config(self, seed: int) -> PipelineConfig:
        return PipelineConfig(
            multi_chunk_text_strategy=self.multi_chunk_text_strategy,
            random_seed=seed,
            min_chunk_duration_sec=self.min_chunk_duration_sec,
        )


@dataclass(slots=True)
class TrainConfig:
    output_dir: str = "./outputs/train"
    max_steps: int = 100
    log_every: int = 10
    save_every: int = 50
    learning_rate: float = 1e-3
    postnet_learning_rate: float = 5e-4
    code_loss_weight: float = 1.0
    acoustic_l1_weight: float = 0.5
    acoustic_cosine_weight: float = 0.1
    llm_only_steps: int = 2000
    acoustic_ramp_steps: int = 4000
    postnet_start_lr_scale: float = 0.0
    postnet_final_lr_scale: float = 1.0
    seed: int = 7
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


@dataclass(slots=True)
class InferenceConfig:
    output_dir: str = "./outputs/infer"
    checkpoint: str | None = None
    seed: int = 7
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    max_samples: int = 0


@dataclass(slots=True)
class TrainAppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


@dataclass(slots=True)
class InferenceAppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def load_train_config(path: str | None) -> TrainAppConfig:
    if not path:
        return TrainAppConfig()
    payload = _load_json(path)
    return TrainAppConfig(
        data=DataConfig(**payload.get("data", {})),
        model=ModelConfig(**payload.get("model", {})),
        train=TrainConfig(**payload.get("train", {})),
    )


def load_inference_config(path: str | None) -> InferenceAppConfig:
    if not path:
        return InferenceAppConfig()
    payload = _load_json(path)
    return InferenceAppConfig(
        data=DataConfig(**payload.get("data", {})),
        model=ModelConfig(**payload.get("model", {})),
        inference=InferenceConfig(**payload.get("inference", {})),
    )


def save_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def to_dict(config: Any) -> dict[str, Any]:
    return asdict(config)


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
