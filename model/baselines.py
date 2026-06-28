from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from model.components.postnet import ResidualPostNet, ResidualPostNetConfig
from model.config import ModelConfig
from model.llm import Layer0GeneratorProtocol, RVQProjectorProtocol
from model.schemas import Phase1Prompt, Phase1Result, Phase2Result


class BaselineLayer0Generator(nn.Module, Layer0GeneratorProtocol):
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        vocab_size: int = 1024,
        max_tokens: int = 64,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.code_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, reference_continuous: torch.Tensor, target_token_count: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = self.encoder(reference_continuous)
        logits = self.code_head(hidden_states)
        codes = logits.argmax(dim=-1)
        if target_token_count is not None:
            hidden_states = self._resize_sequence(hidden_states, target_token_count)
            logits = self._resize_sequence(logits, target_token_count)
            codes = self._resize_sequence(codes.unsqueeze(-1).float(), target_token_count).squeeze(-1).long()
        else:
            hidden_states = hidden_states[: self.max_tokens]
            logits = logits[: self.max_tokens]
            codes = codes[: self.max_tokens]
        return codes, hidden_states, logits

    def infer_layer0(self, prompt: Phase1Prompt) -> Phase1Result:
        reference_continuous = prompt.voice_clone_reference.continuous_32_layers[:, 0, :]
        codes, hidden_states, _ = self.forward(reference_continuous)
        return Phase1Result(
            layer0_codes=codes.tolist(),
            layer0_hidden_states=hidden_states,
            metadata={
                "voice_clone_text": prompt.voice_clone_text,
                "target_text": prompt.target_text,
            },
        )

    def _resize_sequence(self, tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        if tensor.shape[0] == target_len:
            return tensor
        if tensor.shape[0] > target_len:
            return tensor[:target_len]
        repeat_count = target_len - tensor.shape[0]
        pad = tensor[-1:].expand(repeat_count, *tensor.shape[1:])
        return torch.cat([tensor, pad], dim=0)


class BaselineRVQProjector(nn.Module, RVQProjectorProtocol):
    def __init__(
        self,
        hidden_dim: int = 64,
        continuous_dim: int = 8,
        num_layers: int = 32,
        postnet_config: ResidualPostNetConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.continuous_dim = continuous_dim
        self.num_layers = num_layers
        self.postnet = ResidualPostNet(
            postnet_config
            or ResidualPostNetConfig(
                layer0_dim=continuous_dim,
                llm_hidden_dim=hidden_dim,
                output_dim=continuous_dim,
                model_dim=hidden_dim,
                num_layers=num_layers,
            )
        )

    def forward(self, layer0_hidden_states: torch.Tensor, layer0_continuous: torch.Tensor) -> torch.Tensor:
        continuous = layer0_continuous.to(layer0_hidden_states.device)
        return self.postnet(
            layer0_embeddings=continuous,
            llm_hidden_states=layer0_hidden_states,
        )

    def project_to_full_rvq(self, layer0_hidden_states: torch.Tensor, layer0_continuous: torch.Tensor) -> Phase2Result:
        return Phase2Result(
            continuous_32_layers=self.forward(layer0_hidden_states, layer0_continuous),
            metadata={"num_layers": self.num_layers},
        )


@dataclass(slots=True)
class BaselineBundle:
    layer0_generator: BaselineLayer0Generator
    rvq_projector: BaselineRVQProjector


def build_baseline_bundle(input_dim: int = 8, hidden_dim: int = 64, vocab_size: int = 1024, num_layers: int = 32) -> BaselineBundle:
    return BaselineBundle(
        layer0_generator=BaselineLayer0Generator(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
        ),
        rvq_projector=BaselineRVQProjector(
            hidden_dim=hidden_dim,
            continuous_dim=input_dim,
            num_layers=num_layers,
        ),
    )


def build_rvq_projector_from_model_config(model_config: ModelConfig) -> BaselineRVQProjector:
    return BaselineRVQProjector(
        hidden_dim=model_config.llm_d_model,
        continuous_dim=model_config.input_dim,
        num_layers=model_config.num_layers,
        postnet_config=model_config.build_postnet_config(),
    )
