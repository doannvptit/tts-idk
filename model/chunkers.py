from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch
from chunkformer import ChunkFormerModel

from model.schemas import StreamSample, TimestampSpan


def _parse_chunkformer_time(value: str) -> float:
    hours, minutes, seconds, milliseconds = value.split(":")
    return (
        int(hours) * 3600.0
        + int(minutes) * 60.0
        + int(seconds)
        + int(milliseconds) / 1000.0
    )


@lru_cache(maxsize=4)
def _load_chunkformer_model(model_name: str, device: str) -> ChunkFormerModel:
    model = ChunkFormerModel.from_pretrained(model_name)
    return model.to(device)


@contextmanager
def _disable_chunkformer_progress():
    import chunkformer.chunkformer_model as chunkformer_model

    original_tqdm = chunkformer_model.tqdm

    def quiet_tqdm(*args, **kwargs):
        kwargs["disable"] = True
        return original_tqdm(*args, **kwargs)

    chunkformer_model.tqdm = quiet_tqdm
    try:
        yield
    finally:
        chunkformer_model.tqdm = original_tqdm


@dataclass(slots=True)
class ChunkFormerTimestampChunker:
    model_name: str = "khanhld/chunkformer-ctc-large-vie"
    chunk_size: int = 64
    left_context_size: int = 128
    right_context_size: int = 128
    total_batch_duration: int = 14400
    max_silence_duration: float = 0.5
    show_progress: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def chunk(self, sample: StreamSample) -> list[TimestampSpan]:
        if sample.timestamps:
            return list(sample.timestamps)
        audio_path = self._resolve_audio_path(sample)
        if audio_path is None:
            raise ValueError(
                f"ChunkFormerTimestampChunker needs audio_path or waveform when timestamps are missing: {sample.sample_id}"
            )
        try:
            model = _load_chunkformer_model(self.model_name, self.device)
            decode_context = nullcontext() if self.show_progress else _disable_chunkformer_progress()
            with decode_context:
                transcription = model.endless_decode(
                    audio_path=str(audio_path),
                    chunk_size=self.chunk_size,
                    left_context_size=self.left_context_size,
                    right_context_size=self.right_context_size,
                    total_batch_duration=self.total_batch_duration,
                    return_timestamps=True,
                    max_silence_duration=self.max_silence_duration,
                )
        finally:
            if sample.audio_path is None and audio_path.exists():
                audio_path.unlink(missing_ok=True)
        spans = _chunkformer_output_to_spans(transcription)
        sample.timestamps = spans
        if not sample.ref_text:
            sample.ref_text = " ".join(span.text for span in spans).strip() or sample.text
        return spans

    def _resolve_audio_path(self, sample: StreamSample) -> Path | None:
        if sample.audio_path:
            audio_path = Path(sample.audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio path does not exist for sample {sample.sample_id}: {audio_path}")
            return audio_path
        if sample.waveform is None or sample.sample_rate is None:
            return None

        import soundfile as sf

        waveform = torch.as_tensor(sample.waveform).detach().cpu()
        if waveform.dim() == 2:
            waveform = waveform.transpose(0, 1)
        with NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            sf.write(handle.name, waveform.numpy(), int(sample.sample_rate))
            return Path(handle.name)


def _chunkformer_output_to_spans(items: Any) -> list[TimestampSpan]:
    if not isinstance(items, list):
        raise TypeError(f"Unexpected ChunkFormer output type: {type(items)!r}")
    spans: list[TimestampSpan] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("decode", "")).strip()
        if not text:
            continue
        start_raw = item.get("start")
        end_raw = item.get("end")
        if not isinstance(start_raw, str) or not isinstance(end_raw, str):
            continue
        start_sec = _parse_chunkformer_time(start_raw)
        end_sec = _parse_chunkformer_time(end_raw)
        if end_sec <= start_sec:
            continue
        spans.append(TimestampSpan(start_sec=start_sec, end_sec=end_sec, text=text))
    return spans
