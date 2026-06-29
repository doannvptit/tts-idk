from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import chunkformer.chunkformer_model as chunkformer_model

from model.chunkers import ChunkFormerTimestampChunker
from model.schemas import StreamSample


class _DummyChunkFormerModel:
    def endless_decode(self, **kwargs):
        for _ in chunkformer_model.tqdm([0]):
            pass
        return [{"decode": "xin chao", "start": "00:00:00:000", "end": "00:00:01:000"}]


class ChunkFormerTimestampChunkerTest(TestCase):
    def test_disables_chunkformer_tqdm_by_default(self) -> None:
        calls: list[bool | None] = []

        def fake_tqdm(*args, **kwargs):
            calls.append(kwargs.get("disable"))
            return iter(args[0])

        with TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "sample.wav"
            audio_path.write_bytes(b"placeholder")
            sample = StreamSample(sample_id="s1", text="", audio_path=str(audio_path))

            original_tqdm = chunkformer_model.tqdm
            chunkformer_model.tqdm = fake_tqdm
            try:
                with patch("model.chunkers._load_chunkformer_model", return_value=_DummyChunkFormerModel()):
                    spans = ChunkFormerTimestampChunker(device="cpu").chunk(sample)
            finally:
                chunkformer_model.tqdm = original_tqdm

        self.assertEqual(calls, [True])
        self.assertEqual(spans[0].text, "xin chao")

    def test_can_keep_chunkformer_tqdm_visible(self) -> None:
        calls: list[bool | None] = []

        def fake_tqdm(*args, **kwargs):
            calls.append(kwargs.get("disable"))
            return iter(args[0])

        with TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "sample.wav"
            audio_path.write_bytes(b"placeholder")
            sample = StreamSample(sample_id="s1", text="", audio_path=str(audio_path))

            original_tqdm = chunkformer_model.tqdm
            chunkformer_model.tqdm = fake_tqdm
            try:
                with patch("model.chunkers._load_chunkformer_model", return_value=_DummyChunkFormerModel()):
                    ChunkFormerTimestampChunker(device="cpu", show_progress=True).chunk(sample)
            finally:
                chunkformer_model.tqdm = original_tqdm

        self.assertEqual(calls, [None])
