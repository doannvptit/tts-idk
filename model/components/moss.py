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


class MossAudioTokenizerCodec:
    """Hugging Face adapter for OpenMOSS MOSS-Audio-Tokenizer-Nano."""

    def __init__(
        self,
        repo_id: str = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
        device: str | None = None,
        num_layers: int = 16,
        frame_rate_hz: float = 12.5,
        chunk_duration: float | None = None,
    ) -> None:
        import torch
        from transformers import AutoModel

        self.repo_id = repo_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_layers = num_layers
        self.frame_rate_hz = frame_rate_hz
        self.chunk_duration = chunk_duration
        self.model = AutoModel.from_pretrained(repo_id, trust_remote_code=True).to(self.device).eval()
        self.sample_rate = int(getattr(self.model.config, "sampling_rate", 48000))
        quantizer_kwargs = getattr(self.model.config, "quantizer_kwargs", {})
        self.embedding_dim = int(quantizer_kwargs.get("codebook_dim", 8))
        self.vocab_size = int(quantizer_kwargs.get("codebook_size", 1024))

    def encode(self, sample: StreamSample) -> CodecFeatures:
        import torch

        waveform, sample_rate = self._load_waveform(sample)
        waveform = waveform.to(self.device, dtype=torch.float32)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] == 1 and int(getattr(self.model.config, "number_channels", 2)) == 2:
            waveform = waveform.expand(2, -1)

        with torch.inference_mode():
            encoded = self.model.encode(
                waveform,
                num_quantizers=self.num_layers,
                return_dict=True,
                chunk_duration=self.chunk_duration,
            )
            codes = encoded.audio_codes[:, 0, : int(encoded.audio_codes_lengths[0].item())].long()
            continuous = self._codebook_embeddings(codes)
        return CodecFeatures(
            discrete_codes=codes.transpose(0, 1).detach().cpu(),
            continuous_embeddings=continuous.detach().cpu(),
            frame_rate_hz=self.frame_rate_hz,
            num_layers=self.num_layers,
        )

    def _load_waveform(self, sample: StreamSample):
        import torch
        import torchaudio

        if sample.waveform is not None and sample.sample_rate:
            waveform = torch.as_tensor(sample.waveform)
            sample_rate = int(sample.sample_rate)
        elif sample.audio_path:
            waveform, sample_rate = torchaudio.load(sample.audio_path)
        else:
            raise ValueError(f"MOSS tokenizer needs waveform or audio_path: {sample.sample_id}")

        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform.float(), sample_rate, self.sample_rate)
            sample_rate = self.sample_rate
        return waveform, sample_rate

    def _codebook_embeddings(self, codes):
        import torch

        embeddings = []
        quantizers = getattr(self.model.quantizer, "quantizers", [])
        for layer_index, quantizer in enumerate(quantizers[: codes.shape[0]]):
            layer_codes = codes[layer_index].unsqueeze(0)
            if hasattr(quantizer, "decode_code_wo_out_proj"):
                emb = quantizer.decode_code_wo_out_proj(layer_codes)
            elif hasattr(quantizer, "embed_code"):
                emb = quantizer.embed_code(layer_codes).transpose(1, 2)
            else:
                emb = quantizer.decode_code(layer_codes)
            embeddings.append(emb.squeeze(0).transpose(0, 1).float())
        if not embeddings:
            raise RuntimeError("MOSS tokenizer returned no quantizer embeddings.")
        return torch.stack(embeddings, dim=1)


class MockMossCodec:
    """Deterministic mock used to validate chunk selection and phase wiring."""

    def __init__(
        self,
        num_layers: int = 16,
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
