from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn
from tokenizers import Tokenizer

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


@dataclass(slots=True)
class GPTTrainOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    audio_hidden_states: torch.Tensor
    audio_target_ids: torch.Tensor
    code_loss_mask: torch.Tensor


@dataclass(slots=True)
class GPTPhase1Config:
    tokenizer_path: str = "assets/vi_wikipedia_bbpe_2048_copy.json"
    text_vocab_size: int = 2048
    audio_codebook_size: int = 1024
    audio_token_offset: int = 7
    max_seq_len: int = 2048
    max_audio_tokens: int = 256
    d_model: int = 640
    n_head: int = 10
    n_layer: int = 10
    mlp_ratio: int = 4
    dropout: float = 0.1
    reference_dim: int = 8
    num_audio_layers: int = 16
    max_reference_frames: int = 64
    audio_tokens_per_text_token: int = 4
    postnet_hidden_layers: list[int] = field(default_factory=lambda: [2, 4, 6, 8, 10])

    @property
    def total_vocab_size(self) -> int:
        return self.text_vocab_size


class GPTPhase1Model(nn.Module, Layer0GeneratorProtocol):
    def __init__(self, config: GPTPhase1Config) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = Tokenizer.from_file(str(Path(config.tokenizer_path)))
        self.bos_id = self._required_token_id("<BOS>")
        self.eos_id = self._required_token_id("<EOS>")
        self.audio_start_id = self._required_token_id("<AUDIO>")
        self.audio_end_id = self._required_token_id("</AUDIO>")
        self.voice_clone_start_id = self._required_token_id("<VOICE_CLONE>")
        self.voice_clone_end_id = self._required_token_id("</VOICE_CLONE>")
        self.vocab_size = self.tokenizer.get_vocab_size()
        audio_token_ids = [self._required_token_id(f"[audio_token_{index}]") for index in range(config.audio_codebook_size)]
        token_id_to_audio_code = torch.full((self.vocab_size,), -1, dtype=torch.long)
        token_id_to_audio_code[torch.tensor(audio_token_ids, dtype=torch.long)] = torch.arange(
            config.audio_codebook_size,
            dtype=torch.long,
        )
        self.register_buffer("audio_token_ids", torch.tensor(audio_token_ids, dtype=torch.long), persistent=False)
        self.register_buffer("token_id_to_audio_code", token_id_to_audio_code, persistent=False)
        self.token_embedding = nn.Embedding(self.vocab_size, config.d_model)
        self.position_embedding = LearnedPositionalEmbedding(config.max_seq_len, config.d_model)
        self.reference_projection = nn.Linear(config.reference_dim * config.num_audio_layers, config.d_model)
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
        self.code_head = nn.Linear(config.d_model, self.vocab_size)

    def forward_train(self, prompt: Phase1Prompt, target_token_ids: torch.Tensor) -> GPTTrainOutput:
        condition = self._build_condition_embeddings(prompt).unsqueeze(0)
        target_token_ids = target_token_ids.to(self.device)
        input_token_ids = target_token_ids[:-1]
        label_token_ids = target_token_ids[1:]
        token_embeddings = self.token_embedding(input_token_ids).unsqueeze(0)
        x = torch.cat([condition, token_embeddings], dim=1)
        if x.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"Training sequence length {x.shape[1]} exceeds llm_max_seq_len={self.config.max_seq_len}. "
                "Increase llm_max_seq_len or split the sample into shorter chunks; audio tokens are not truncated."
            )
        final_hidden, selected_hidden = self._run_transformer(x)
        target_hidden = selected_hidden[:, -input_token_ids.shape[0] :, :].squeeze(0)
        logits = self.code_head(final_hidden[:, -input_token_ids.shape[0] :, :]).squeeze(0)
        audio_mask = self.is_audio_token(label_token_ids)
        code_loss_mask = self.is_code_loss_target(label_token_ids)
        return GPTTrainOutput(
            logits=logits,
            hidden_states=target_hidden,
            audio_hidden_states=target_hidden[audio_mask],
            audio_target_ids=label_token_ids[audio_mask],
            code_loss_mask=code_loss_mask,
        )

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
        prefix_ids = self._build_inference_prefix_tokens(prompt)
        condition = self._build_condition_embeddings(prompt).unsqueeze(0)
        prefix_embeddings = self.token_embedding(prefix_ids).unsqueeze(0)
        generated: list[int] = []
        hidden_states = []
        reached_eos = False

        for _ in range(self.config.max_audio_tokens):
            if generated:
                audio_prefix = self.audio_codes_to_token_ids(
                    torch.tensor(generated, dtype=torch.long, device=self.device)
                )
                audio_embeddings = self.token_embedding(audio_prefix).unsqueeze(0)
                x = torch.cat([condition, prefix_embeddings, audio_embeddings], dim=1)
            else:
                x = torch.cat([condition, prefix_embeddings], dim=1)
            if x.shape[1] > self.config.max_seq_len:
                raise ValueError(
                    f"Generation sequence length {x.shape[1]} exceeds llm_max_seq_len={self.config.max_seq_len}. "
                    "Increase llm_max_seq_len or reduce max_audio_tokens; audio tokens are not truncated."
                )
            final_hidden, selected_hidden = self._run_transformer(x)
            next_hidden = selected_hidden[:, -1, :]
            logits = self.code_head(final_hidden[:, -1, :]).squeeze(0)
            audio_logits = logits.index_select(0, self.audio_token_ids)
            stop_logit = logits[self.audio_end_id].unsqueeze(0)
            next_token = int(torch.cat([audio_logits, stop_logit], dim=0).argmax(dim=-1).item())
            if next_token == self.config.audio_codebook_size:
                reached_eos = True
                break
            next_code = next_token
            generated.append(next_code)
            hidden_states.append(next_hidden.squeeze(0))

        if hidden_states:
            hidden_tensor = torch.stack(hidden_states, dim=0)
        else:
            hidden_tensor = torch.empty(
                0, self.postnet_hidden_dim, device=self.device, dtype=self.token_embedding.weight.dtype
            )
        return generated, hidden_tensor, reached_eos

    @property
    def device(self) -> torch.device:
        return self.token_embedding.weight.device

    @property
    def postnet_hidden_dim(self) -> int:
        return self.config.d_model * len(self._postnet_layer_indices())

    def build_training_sequence(self, chunks: list[tuple[str, torch.Tensor]]) -> torch.Tensor:
        token_ids = [self.bos_id]
        for text, layer0_codes in chunks:
            token_ids.extend(self.encode_text(text))
            token_ids.append(self.audio_start_id)
            token_ids.extend(self.audio_codes_to_token_ids(layer0_codes).detach().cpu().tolist())
            token_ids.append(self.audio_end_id)
        token_ids.append(self.eos_id)
        return torch.tensor(token_ids, dtype=torch.long, device=self.device)

    def audio_codes_to_token_ids(self, codes: torch.Tensor) -> torch.Tensor:
        codes = codes.long().to(self.device)
        return self.audio_token_ids.index_select(0, codes)

    def token_ids_to_audio_codes(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.long().to(self.device)
        return self.token_id_to_audio_code.index_select(0, token_ids)

    def is_audio_token(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.long().to(self.device)
        return self.token_id_to_audio_code.index_select(0, token_ids) >= 0

    def is_code_loss_target(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.to(self.device)
        return self.is_audio_token(token_ids) | (token_ids == self.audio_end_id)

    def build_code_loss_inputs(
        self,
        logits: torch.Tensor,
        label_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        label_token_ids = label_token_ids.to(logits.device)
        mask = self.is_code_loss_target(label_token_ids)
        selected_logits = logits[mask]
        selected_labels = label_token_ids[mask]
        audio_logits = selected_logits.index_select(1, self.audio_token_ids.to(logits.device))
        stop_logits = selected_logits[:, self.audio_end_id : self.audio_end_id + 1]
        restricted_logits = torch.cat([audio_logits, stop_logits], dim=-1)
        restricted_labels = torch.where(
            self.is_audio_token(selected_labels),
            self.token_ids_to_audio_codes(selected_labels),
            torch.full_like(selected_labels, self.config.audio_codebook_size),
        )
        return restricted_logits, restricted_labels

    def encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def _build_inference_prefix_tokens(self, prompt: Phase1Prompt) -> torch.Tensor:
        token_ids = [self.bos_id]
        token_ids.extend(self.encode_text(prompt.target_text))
        token_ids.append(self.audio_start_id)
        return torch.tensor(token_ids, dtype=torch.long, device=self.device)

    def _build_reference_embeddings(self, prompt: Phase1Prompt) -> torch.Tensor:
        continuous = prompt.voice_clone_reference.continuous_rvq_layers.float().to(
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

    def _build_condition_embeddings(self, prompt: Phase1Prompt) -> torch.Tensor:
        token_ids = [self.voice_clone_start_id]
        token_ids.extend(self.encode_text(prompt.voice_clone_text))
        text_prefix = self.token_embedding(torch.tensor(token_ids, dtype=torch.long, device=self.device))
        reference = self._build_reference_embeddings(prompt)
        close = self.token_embedding(torch.tensor([self.voice_clone_end_id], dtype=torch.long, device=self.device))
        return torch.cat([text_prefix, reference, close], dim=0)

    def _run_transformer(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self.position_embedding(x.shape[1], x.device).unsqueeze(0)
        x = x + pos
        causal_mask = build_causal_mask(x.shape[1], device=x.device)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        selected = []
        selected_layers = set(self._postnet_layer_indices())
        for index, block in enumerate(self.blocks, start=1):
            x = block(x, causal_mask=causal_mask)
            if index in selected_layers:
                selected.append(x)
        x = self.final_ln(x)
        if not selected:
            selected = [x]
        selected_hidden = torch.cat(selected, dim=-1)
        return x, selected_hidden

    def _postnet_layer_indices(self) -> list[int]:
        return [
            layer
            for layer in self.config.postnet_hidden_layers
            if 1 <= layer <= self.config.n_layer
        ] or [self.config.n_layer]

    def _required_token_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Tokenizer is missing required token: {token}")
        return int(token_id)

    def _estimate_audio_token_count(self, target_text: str) -> int:
        text_tokens = max(1, len(self.encode_text(target_text)))
        return min(self.config.max_audio_tokens, text_tokens * self.config.audio_tokens_per_text_token)

    def build_target_sequence(self, target_codes: torch.Tensor) -> torch.Tensor:
        target_codes = target_codes.to(self.device)
        return torch.cat(
            [
                torch.tensor([self.bos_id, self.audio_start_id], dtype=torch.long, device=self.device),
                self.audio_codes_to_token_ids(target_codes),
                torch.tensor([self.audio_end_id, self.eos_id], dtype=torch.long, device=self.device),
            ],
            dim=0,
        )


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
    num_layers: int = 16

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
            continuous_rvq_layers=torch.cat([expanded_hidden, expanded_continuous], dim=-1),
            metadata={"num_layers": self.num_layers},
        )
