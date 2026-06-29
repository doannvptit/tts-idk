from __future__ import annotations

import unittest

from model.components.moss import MockMossCodec
from model.llm import MockLayer0Generator, MockRVQProjector
from model.schemas import StreamSample, TimestampSpan
from model.streaming_tts import PipelineConfig, RealtimeChunkedTTSPipeline, TimestampPassthroughChunker


class StreamingTTSPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = RealtimeChunkedTTSPipeline(
            chunker=TimestampPassthroughChunker(),
            codec=MockMossCodec(),
            layer0_generator=MockLayer0Generator(token_count=10),
            rvq_projector=MockRVQProjector(),
            config=PipelineConfig(random_seed=5),
        )

    def test_single_chunk_uses_sample_text_for_both_prompt_and_target(self) -> None:
        sample = StreamSample(
            sample_id="single",
            text="mot cau duy nhat",
            timestamps=[TimestampSpan(start_sec=0.0, end_sec=2.0, text="mot cau duy nhat")],
            metadata={"duration_sec": 2.0},
        )

        result = self.pipeline.process_sample(sample)

        self.assertEqual(result.chunk_mode, "single_chunk")
        self.assertIsNone(result.selected_span)
        self.assertEqual(result.prompt.voice_clone_text, sample.text)
        self.assertEqual(result.prompt.target_text, sample.text)
        self.assertEqual(result.prompt.voice_clone_reference.continuous_rvq_layers.shape[1], 16)

    def test_multi_chunk_uses_timestamp_text_for_voice_clone_and_target(self) -> None:
        sample = StreamSample(
            sample_id="multi",
            text="day la toan bo cau",
            timestamps=[
                TimestampSpan(start_sec=0.0, end_sec=0.8, text="day la"),
                TimestampSpan(start_sec=1.2, end_sec=2.0, text="toan bo"),
                TimestampSpan(start_sec=2.5, end_sec=3.2, text="cau"),
            ],
            metadata={"duration_sec": 3.5},
        )

        result = self.pipeline.process_sample(sample)

        self.assertEqual(result.chunk_mode, "multi_chunk")
        self.assertIsNotNone(result.selected_span)
        self.assertEqual(result.prompt.voice_clone_text, result.selected_span.text)
        self.assertEqual(result.prompt.target_text, result.selected_span.text)
        self.assertEqual(result.prompt.voice_clone_reference.continuous_rvq_layers.shape[1], 16)
        self.assertLess(
            result.prompt.voice_clone_reference.continuous_rvq_layers.shape[0],
            int(sample.metadata["duration_sec"] * 12.5),
        )

    def test_multi_chunk_can_replace_target_text_if_needed(self) -> None:
        pipeline = RealtimeChunkedTTSPipeline(
            chunker=TimestampPassthroughChunker(),
            codec=MockMossCodec(),
            layer0_generator=MockLayer0Generator(token_count=10),
            rvq_projector=MockRVQProjector(),
            config=PipelineConfig(
                random_seed=5,
                multi_chunk_text_strategy="replace_target_text",
            ),
        )
        sample = StreamSample(
            sample_id="multi",
            text="day la toan bo cau",
            timestamps=[
                TimestampSpan(start_sec=0.0, end_sec=0.8, text="day la"),
                TimestampSpan(start_sec=1.2, end_sec=2.0, text="toan bo"),
            ],
            metadata={"duration_sec": 2.5},
        )

        result = pipeline.process_sample(sample)

        self.assertEqual(result.prompt.target_text, result.selected_span.text)


if __name__ == "__main__":
    unittest.main()
