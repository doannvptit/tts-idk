from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


@dataclass(slots=True)
class TimestampSpan:
    start_sec: float
    end_sec: float
    text: str

    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass(slots=True)
class StreamSample:
    sample_id: str
    text: str
    audio_path: str | None = None
    ref_text: str | None = None
    ref_audio_path: str | None = None
    waveform: Any | None = None
    sample_rate: int | None = None
    timestamps: list[TimestampSpan] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VoiceCloneReference:
    text: str
    timestamps: list[TimestampSpan]
    continuous_rvq_layers: Any
    source_sample_id: str

    @property
    def continuous_32_layers(self) -> Any:
        return self.continuous_rvq_layers


@dataclass(slots=True)
class Phase1Prompt:
    voice_clone_text: str
    target_text: str
    voice_clone_reference: VoiceCloneReference


@dataclass(slots=True)
class Phase1Result:
    layer0_codes: Sequence[int]
    layer0_hidden_states: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Phase2Result:
    continuous_rvq_layers: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def continuous_32_layers(self) -> Any:
        return self.continuous_rvq_layers


@dataclass(slots=True)
class InferenceExample:
    sample: StreamSample
    chunk_mode: str
    selected_span: TimestampSpan | None
    prompt: Phase1Prompt
    phase1: Phase1Result
    phase2: Phase2Result


def iter_stream(sample_iterable: Iterable[StreamSample]) -> Iterable[StreamSample]:
    for sample in sample_iterable:
        yield sample
