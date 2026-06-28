from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from datasets import IterableDataset, interleave_datasets, load_dataset

from model.schemas import StreamSample, TimestampSpan


def load_records(
    input_path: str | None = None,
    dataset_name: str | None = None,
    dataset_names: list[str] | None = None,
    split: str = "train",
    streaming: bool = True,
) -> Iterable[dict[str, Any]]:
    if input_path:
        path = Path(input_path)
        if path.suffix != ".jsonl":
            raise ValueError(f"Only .jsonl is supported for local streaming input, got: {path}")
        return _iter_jsonl(path)
    if dataset_name:
        dataset = load_dataset(dataset_name, split=split, streaming=streaming)
        if not isinstance(dataset, IterableDataset) and streaming:
            return iter(dataset)
        return dataset
    if dataset_names:
        datasets = [load_dataset(name, split=split, streaming=streaming) for name in dataset_names]
        return interleave_datasets(datasets)
    raise ValueError("Either input_path or dataset_name must be provided.")


def record_to_stream_sample(record: dict[str, Any]) -> StreamSample:
    sample_id = str(record.get("sample_id", record.get("id", "unknown")))
    target_text = str(record.get("target_text", record.get("text", "")))
    ref_text = str(record.get("ref_text", target_text))
    ref_audio_path = record.get("ref_audio_path")
    target_audio_path = record.get("target_audio_path", record.get("audio_path", ref_audio_path))
    timestamps = [_parse_timestamp(item) for item in record.get("timestamps", [])]
    waveform, sample_rate, audio_path = _extract_audio_fields(record.get("audio"))
    if target_audio_path is None:
        target_audio_path = audio_path

    metadata = dict(record.get("metadata", {}))
    for key in ("duration_sec", "speaker_id", "lang", "source"):
        if key in record and key not in metadata:
            metadata[key] = record[key]
    if "speaker" in record and "speaker_id" not in metadata:
        metadata["speaker_id"] = record["speaker"]
    if sample_rate is not None and "sample_rate" not in metadata:
        metadata["sample_rate"] = sample_rate
    if waveform is not None and "duration_sec" not in metadata and sample_rate:
        metadata["duration_sec"] = float(waveform.shape[-1]) / float(sample_rate)

    return StreamSample(
        sample_id=sample_id,
        text=target_text,
        audio_path=target_audio_path,
        ref_text=ref_text,
        ref_audio_path=ref_audio_path,
        waveform=waveform,
        sample_rate=sample_rate,
        timestamps=timestamps,
        metadata=metadata,
    )


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc


def _parse_timestamp(item: dict[str, Any]) -> TimestampSpan:
    return TimestampSpan(
        start_sec=float(item["start_sec"]),
        end_sec=float(item["end_sec"]),
        text=str(item["text"]),
    )


def _extract_audio_fields(audio_value: Any) -> tuple[Any | None, int | None, str | None]:
    import torch

    if audio_value is None:
        return None, None, None
    if isinstance(audio_value, dict):
        if "array" in audio_value:
            waveform = torch.as_tensor(audio_value["array"])
            return waveform, audio_value.get("sampling_rate"), audio_value.get("path")
        return None, audio_value.get("sampling_rate"), audio_value.get("path")
    if hasattr(audio_value, "get_all_samples"):
        samples = audio_value.get_all_samples()
        return samples.data, getattr(samples, "sample_rate", None), None
    return None, None, None
