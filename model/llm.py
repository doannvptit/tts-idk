from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn

from model.components.attention import build_causal_mask
from model.components.decoder_block import GPTDecoderBlock
from model.components.positional_encoding import LearnedPositionalEmbedding
from model.schemas import Phase1Prompt, Phase1Result, Phase2Result


class Layer0GeneratorProtocol(Protocol):
    def infer_layer0(self, prompt: Phase1Prompt) -> Phase1Result:
        """Run phase 1 and return layer-0 RVQ codes plus hidden states."""


class RVQProjectorProtocol(Protocol):
    def project_to_full_rvq(self, layer0_hidden_states: Any, layer0_continuous: Any) -> Phase2Result:
        """Run phase 2 and return continuous embeddings for the full RVQ stack."""


class ByteTokenizer:
    special_tokens = ["<pad>", "<bos>", "<sep>", "<audio>"]

    def __init__(self) -> None:
        self.token_to_id = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.base_offset = len(self.special_tokens)
        self.vocab_size = self.base_offset + 256

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def sep_id(self) -> int:
        return self.token_to_id["<sep>"]

    @property
    def audio_id(self) -> int:
        return self.token_to_id["<audio>"]

    def encode(self, text: str) -> list[int]:
        return [self.base_offset + value for value in text.encode("utf-8", errors="replace")]


@dataclass(slots=True)
class GPTPhase1Config:
    text_vocab_size: int = 260
    audio_vocab_size: int = 1024
    max_seq_len: int = 2048
    max_audio_tokens: int = 256
    d_model: int = 640
    n_head: int = 10
    n_layer: int = 10
    mlp_ratio: int = 4
    dropout: float = 0.1
    reference_dim: int = 8
    max_reference_frames: int = 64
    audio_tokens_per_text_token: int = 4


class GPTPhase1Model(nn.Module, Layer0GeneratorProtocol):
    def __init__(self, config: GPTPhase1Config) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = ByteTokenizer()
        self.audio_eos_token_id = config.audio_vocab_size
        self.text_embedding = nn.Embedding(config.text_vocab_size, config.d_model)
        self.position_embedding = LearnedPositionalEmbedding(config.max_seq_len, config.d_model)
        self.reference_projection = nn.Linear(config.reference_dim * 32, config.d_model)
        self.audio_embedding = nn.Embedding(config.audio_vocab_size + 1, config.d_model)
        self.blocks = nn.ModuleList(
            [
                GPTDecoderBlock(
                    d_model=config.d_model,
                    n_head=config.n_head,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.final_ln = nn.LayerNorm(config.d_model)
        self.code_head = nn.Linear(config.d_model, config.audio_vocab_size + 1)

    def forward_train(self, prompt: Phase1Prompt, target_codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = self._build_text_tokens(prompt)
        text_embeddings = self.text_embedding(token_ids).unsqueeze(0)
        reference_embeddings = self._build_reference_embeddings(prompt).unsqueeze(0)
        target_codes = target_codes.to(text_embeddings.device)[: self.config.max_audio_tokens]
        audio_input_tokens = target_codes[:-1]
        audio_embeddings = self.audio_embedding(audio_input_tokens).unsqueeze(0)
        x = torch.cat([reference_embeddings, text_embeddings, audio_embeddings], dim=1)
        if x.shape[1] > self.config.max_seq_len:
            x = x[:, -self.config.max_seq_len :]
        pos = self.position_embedding(x.shape[1], x.device).unsqueeze(0)
        x = x + pos
        causal_mask = build_causal_mask(x.shape[1], device=x.device)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, causal_mask=causal_mask)
        x = self.final_ln(x)
        predict_len = audio_input_tokens.shape[0] + 1
        audio_hidden = x[:, -predict_len:, :].squeeze(0)
        logits = self.code_head(audio_hidden)
        return logits, audio_hidden

    def infer_layer0(self, prompt: Phase1Prompt) -> Phase1Result:
        codes, hidden_states, reached_eos = self.generate(prompt)
        return Phase1Result(
            layer0_codes=codes,
            layer0_hidden_states=hidden_states,
            metadata={
                "voice_clone_text": prompt.voice_clone_text,
                "target_text": prompt.target_text,
                "reached_audio_eos": reached_eos,
            },
        )

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @torch.no_grad()
    def generate(self, prompt: Phase1Prompt) -> tuple[list[int], torch.Tensor, bool]:
        token_ids = self._build_text_tokens(prompt)
        text_embeddings = self.text_embedding(token_ids).unsqueeze(0)
        reference_embeddings = self._build_reference_embeddings(prompt).unsqueeze(0)
        generated: list[int] = []
        hidden_states = []
        reached_eos = False

        for _ in range(self.config.max_audio_tokens):
            if generated:
                audio_prefix = torch.tensor(generated, dtype=torch.long, device=text_embeddings.device)
                audio_embeddings = self.audio_embedding(audio_prefix).unsqueeze(0)
                x = torch.cat([reference_embeddings, text_embeddings, audio_embeddings], dim=1)
            else:
                x = torch.cat([reference_embeddings, text_embeddings], dim=1)
            if x.shape[1] > self.config.max_seq_len:
                x = x[:, -self.config.max_seq_len :]
            pos = self.position_embedding(x.shape[1], x.device).unsqueeze(0)
            x = x + pos
            causal_mask = build_causal_mask(x.shape[1], device=x.device)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            for block in self.blocks:
                x = block(x, causal_mask=causal_mask)
            x = self.final_ln(x)
            next_hidden = x[:, -1, :]
            next_token = self.code_head(next_hidden).argmax(dim=-1).item()
            if next_token == self.audio_eos_token_id:
                reached_eos = True
                break
            generated.append(next_token)
            hidden_states.append(next_hidden.squeeze(0))

        if hidden_states:
            hidden_tensor = torch.stack(hidden_states, dim=0)
        else:
            hidden_tensor = torch.empty(
                0, self.config.d_model, device=self.text_embedding.weight.device, dtype=self.text_embedding.weight.dtype
            )
        return generated, hidden_tensor, reached_eos

    def _build_text_tokens(self, prompt: Phase1Prompt) -> torch.Tensor:
        token_ids = [self.tokenizer.bos_id]
        token_ids.extend(self.tokenizer.encode(prompt.voice_clone_text))
        token_ids.append(self.tokenizer.sep_id)
        token_ids.extend(self.tokenizer.encode(prompt.target_text))
        token_ids.append(self.tokenizer.audio_id)
        return torch.tensor(token_ids, dtype=torch.long, device=self.text_embedding.weight.device)

    def _build_reference_embeddings(self, prompt: Phase1Prompt) -> torch.Tensor:
        continuous = prompt.voice_clone_reference.continuous_32_layers.float().to(
            self.reference_projection.weight.device
        )
        if continuous.shape[0] > self.config.max_reference_frames:
            indices = torch.linspace(
                0,
                continuous.shape[0] - 1,
                self.config.max_reference_frames,
                device=continuous.device,
            ).round().long()
            continuous = continuous.index_select(0, indices)
        flattened = continuous.reshape(continuous.shape[0], -1)
        return self.reference_projection(flattened)

    def _estimate_audio_token_count(self, target_text: str) -> int:
        text_tokens = max(1, len(self.tokenizer.encode(target_text)))
        return min(self.config.max_audio_tokens, text_tokens * self.config.audio_tokens_per_text_token)

    def build_target_sequence(self, target_codes: torch.Tensor) -> torch.Tensor:
        target_codes = target_codes.to(self.text_embedding.weight.device)
        target_codes = target_codes[: self.config.max_audio_tokens - 1]
        eos = torch.tensor([self.audio_eos_token_id], dtype=torch.long, device=target_codes.device)
        return torch.cat([target_codes, eos], dim=0)


@dataclass(slots=True)
class MockLayer0Generator:
    hidden_dim: int = 16
    token_count: int = 24

    def infer_layer0(self, prompt: Phase1Prompt) -> Phase1Result:
        layer0_codes = list(range(self.token_count))
        hidden_states = torch.arange(
            self.token_count * self.hidden_dim, dtype=torch.float32
        ).reshape(self.token_count, self.hidden_dim)
        return Phase1Result(
            layer0_codes=layer0_codes,
            layer0_hidden_states=hidden_states,
            metadata={
                "voice_clone_text": prompt.voice_clone_text,
                "target_text": prompt.target_text,
            },
        )


@dataclass(slots=True)
class MockRVQProjector:
    num_layers: int = 32

    def project_to_full_rvq(self, layer0_hidden_states: Any, layer0_continuous: Any) -> Phase2Result:
        aligned_count = min(layer0_hidden_states.shape[0], layer0_continuous.shape[0])
        hidden = layer0_hidden_states[:aligned_count]
        continuous = layer0_continuous[:aligned_count]
        token_count, hidden_dim = hidden.shape
        expanded_hidden = hidden.unsqueeze(1).expand(token_count, self.num_layers, hidden_dim)
        expanded_continuous = continuous.unsqueeze(1).expand(
            continuous.shape[0], self.num_layers, continuous.shape[-1]
        )
        return Phase2Result(
            continuous_32_layers=torch.cat([expanded_hidden, expanded_continuous], dim=-1),
            metadata={"num_layers": self.num_layers},
        )
