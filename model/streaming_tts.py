from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, Protocol

from model.components.moss import CodecFeatures, MossCodecProtocol
from model.llm import Layer0GeneratorProtocol, RVQProjectorProtocol
from model.schemas import (
    InferenceExample,
    Phase1Prompt,
    StreamSample,
    TimestampSpan,
    VoiceCloneReference,
)


class ChunkerProtocol(Protocol):
    def chunk(self, sample: StreamSample) -> list[TimestampSpan]:
        """Return timestamped text spans for the sample."""


@dataclass(slots=True)
class PipelineConfig:
    multi_chunk_text_strategy: Literal["voice_clone_only", "replace_target_text"] = "replace_target_text"
    random_seed: int = 0
    min_chunk_duration_sec: float = 0.05


class TimestampPassthroughChunker:
    """Uses timestamps already attached to the streamed sample."""

    def chunk(self, sample: StreamSample) -> list[TimestampSpan]:
        return list(sample.timestamps)


class RealtimeChunkedTTSPipeline:
    def __init__(
        self,
        chunker: ChunkerProtocol,
        codec: MossCodecProtocol,
        layer0_generator: Layer0GeneratorProtocol,
        rvq_projector: RVQProjectorProtocol,
        config: PipelineConfig | None = None,
    ) -> None:
        self.chunker = chunker
        self.codec = codec
        self.layer0_generator = layer0_generator
        self.rvq_projector = rvq_projector
        self.config = config or PipelineConfig()
        self._rng = random.Random(self.config.random_seed)

    def process_stream(self, samples: Iterable[StreamSample]) -> Iterator[InferenceExample]:
        for sample in samples:
            yield self.process_sample(sample)

    def prepare_prompt(self, sample: StreamSample) -> tuple[Phase1Prompt, str, TimestampSpan | None, CodecFeatures]:
        reference_sample = self._build_reference_sample(sample)
        codec_features = self.codec.encode(reference_sample)
        spans = self._normalize_spans(self.chunker.chunk(sample))
        is_multi_chunk = len(spans) > 1

        if is_multi_chunk:
            selected_span = self._rng.choice(spans)
            reference_features = codec_features.slice_by_span(selected_span)
            voice_clone_text = selected_span.text
            target_text = (
                selected_span.text
                if self.config.multi_chunk_text_strategy == "replace_target_text"
                else sample.text
            )
            chunk_mode = "multi_chunk"
            reference_spans = [selected_span]
        else:
            selected_span = None
            reference_features = codec_features
            voice_clone_text = sample.ref_text or sample.text
            target_text = sample.text
            chunk_mode = "single_chunk"
            reference_spans = spans or [
                TimestampSpan(
                    start_sec=0.0,
                    end_sec=self._infer_duration_sec(codec_features),
                    text=voice_clone_text,
                )
            ]

        prompt = Phase1Prompt(
            voice_clone_text=voice_clone_text,
            target_text=target_text,
            voice_clone_reference=VoiceCloneReference(
                text=voice_clone_text,
                timestamps=reference_spans,
                continuous_32_layers=reference_features.continuous_embeddings,
                source_sample_id=sample.sample_id,
            ),
        )
        return prompt, chunk_mode, selected_span, codec_features

    def process_sample(self, sample: StreamSample) -> InferenceExample:
        prompt, chunk_mode, selected_span, codec_features = self.prepare_prompt(sample)
        phase1 = self.layer0_generator.infer_layer0(prompt)
        layer0_continuous = self._select_layer0_continuous(codec_features, phase1.layer0_codes)
        phase2 = self.rvq_projector.project_to_full_rvq(
            layer0_hidden_states=phase1.layer0_hidden_states,
            layer0_continuous=layer0_continuous,
        )
        return InferenceExample(
            sample=sample,
            chunk_mode=chunk_mode,
            selected_span=selected_span,
            prompt=prompt,
            phase1=phase1,
            phase2=phase2,
        )

    def _normalize_spans(self, spans: list[TimestampSpan]) -> list[TimestampSpan]:
        filtered = [
            span
            for span in spans
            if span.duration_sec() >= self.config.min_chunk_duration_sec and span.text.strip()
        ]
        filtered.sort(key=lambda span: (span.start_sec, span.end_sec))
        return filtered

    def _infer_duration_sec(self, codec_features: CodecFeatures) -> float:
        total_frames = int(codec_features.continuous_embeddings.shape[0])
        return total_frames / codec_features.frame_rate_hz

    def _select_layer0_continuous(self, codec_features: CodecFeatures, layer0_codes: list[int] | tuple[int, ...]):
        available = codec_features.continuous_embeddings[:, 0, :]
        target_count = len(layer0_codes)
        if available.shape[0] >= target_count:
            return available[:target_count]
        last_frame = available[-1:].expand(target_count - available.shape[0], available.shape[-1])
        return __import__("torch").cat([available, last_frame], dim=0)

    def _build_reference_sample(self, sample: StreamSample) -> StreamSample:
        if not sample.ref_audio_path:
            return sample
        return StreamSample(
            sample_id=f"{sample.sample_id}:ref",
            text=sample.ref_text or sample.text,
            audio_path=sample.ref_audio_path,
            ref_text=sample.ref_text,
            ref_audio_path=sample.ref_audio_path,
            timestamps=sample.timestamps,
            metadata=sample.metadata,
        )
