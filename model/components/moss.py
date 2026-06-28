from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from model.schemas import StreamSample, TimestampSpan


@dataclass(slots=True)
class CodecFeatures:
    discrete_codes: Any
    continuous_embeddings: Any
    frame_rate_hz: float
    num_layers: int

    def slice_by_time(self, start_sec: float, end_sec: float) -> "CodecFeatures":
        start_idx = max(0, int(start_sec * self.frame_rate_hz))
        end_idx = max(start_idx + 1, int(end_sec * self.frame_rate_hz))
        return CodecFeatures(
            discrete_codes=self.discrete_codes[start_idx:end_idx],
            continuous_embeddings=self.continuous_embeddings[start_idx:end_idx],
            frame_rate_hz=self.frame_rate_hz,
            num_layers=self.num_layers,
        )

    def slice_by_span(self, span: TimestampSpan) -> "CodecFeatures":
        return self.slice_by_time(span.start_sec, span.end_sec)


class MossCodecProtocol(Protocol):
    def encode(self, sample: StreamSample) -> CodecFeatures:
        """Return RVQ codes and continuous embeddings for the full sample."""


class MockMossCodec:
    """Deterministic mock used to validate chunk selection and phase wiring."""

    def __init__(
        self,
        num_layers: int = 32,
        embedding_dim: int = 8,
        frame_rate_hz: float = 12.5,
        vocab_size: int = 1024,
    ):
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.frame_rate_hz = frame_rate_hz
        self.vocab_size = vocab_size

    def encode(self, sample: StreamSample) -> CodecFeatures:
        import torch

        if sample.waveform is not None and sample.sample_rate:
            duration_sec = float(sample.waveform.shape[-1]) / float(sample.sample_rate)
        else:
            duration_sec = float(sample.metadata.get("duration_sec", 8.0))
        total_frames = max(1, int(duration_sec * self.frame_rate_hz))
        discrete = (
            torch.arange(total_frames * self.num_layers, dtype=torch.long) % self.vocab_size
        ).reshape(total_frames, self.num_layers)
        continuous = torch.arange(
            total_frames * self.num_layers * self.embedding_dim, dtype=torch.float32
        ).reshape(total_frames, self.num_layers, self.embedding_dim)
        return CodecFeatures(
            discrete_codes=discrete,
            continuous_embeddings=continuous,
            frame_rate_hz=self.frame_rate_hz,
            num_layers=self.num_layers,
        )
