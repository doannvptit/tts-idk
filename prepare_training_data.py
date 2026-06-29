from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from model.chunkers import ChunkFormerTimestampChunker
from model.config import load_train_config
from model.io import load_records, record_to_stream_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute timestamps with ChunkFormer for train data.")
    parser.add_argument("--config", type=str, default="configs/train.default.json")
    parser.add_argument("--output-path", type=str, required=True, help="Output .jsonl with timestamps.")
    parser.add_argument("--input-path", type=str, default=None, help="Override data.input_path.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Override data.dataset_name.")
    parser.add_argument("--split", type=str, default=None, help="Override data.split.")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N records. 0 means no limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_config = load_train_config(args.config)
    if args.input_path is not None:
        app_config.data.input_path = args.input_path
    if args.dataset_name is not None:
        app_config.data.dataset_name = args.dataset_name
        app_config.data.dataset_names = []
    if args.split is not None:
        app_config.data.split = args.split

    chunker = ChunkFormerTimestampChunker()
    records = load_records(
        input_path=app_config.data.input_path,
        dataset_name=app_config.data.dataset_name,
        dataset_names=app_config.data.dataset_names,
        split=app_config.data.split,
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        started = time.perf_counter()
        for record in records:
            sample = record_to_stream_sample(record)
            timestamps = chunker.chunk(sample)
            payload = _build_output_record(record, sample, timestamps, output_path.parent)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
            if count == 1 or count % 10 == 0:
                elapsed = time.perf_counter() - started
                print(
                    json.dumps(
                        {
                            "count": count,
                            "elapsed_sec": elapsed,
                            "sec_per_sample": elapsed / count,
                            "sample_id": sample.sample_id,
                            "num_timestamps": len(timestamps),
                        },
                        ensure_ascii=False,
                    )
                )
            if args.limit and count >= args.limit:
                break

    print(json.dumps({"output_path": str(output_path), "count": count}, ensure_ascii=False))


def _build_output_record(
    record: dict[str, Any],
    sample,
    timestamps,
    output_dir: Path,
) -> dict[str, Any]:
    payload = dict(record)
    payload.pop("audio", None)
    payload["sample_id"] = sample.sample_id
    payload["text"] = sample.text
    payload["ref_text"] = sample.ref_text or sample.text
    payload["audio_path"] = sample.audio_path or _materialize_waveform(sample, output_dir)
    payload["timestamps"] = [
        {"start_sec": span.start_sec, "end_sec": span.end_sec, "text": span.text}
        for span in timestamps
    ]
    metadata = dict(payload.get("metadata", {}))
    metadata.update(sample.metadata)
    payload["metadata"] = metadata
    return payload


def _materialize_waveform(sample, output_dir: Path) -> str | None:
    if sample.waveform is None or sample.sample_rate is None:
        return None

    import re

    import soundfile as sf
    import torch

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample.sample_id)
    audio_path = audio_dir / f"{safe_id}.wav"
    waveform = torch.as_tensor(sample.waveform).detach().cpu()
    if waveform.dim() == 2:
        waveform = waveform.transpose(0, 1)
    sf.write(audio_path, waveform.numpy(), int(sample.sample_rate))
    return str(audio_path)


if __name__ == "__main__":
    main()
