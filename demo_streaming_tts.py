from __future__ import annotations

from model.components.moss import MockMossCodec
from model.llm import MockLayer0Generator, MockRVQProjector
from model.schemas import StreamSample, TimestampSpan
from model.streaming_tts import PipelineConfig, RealtimeChunkedTTSPipeline, TimestampPassthroughChunker


def build_demo_samples() -> list[StreamSample]:
    return [
        StreamSample(
            sample_id="single",
            text="xin chao toi la lumivoice",
            timestamps=[
                TimestampSpan(start_sec=0.0, end_sec=4.0, text="xin chao toi la lumivoice"),
            ],
            metadata={"duration_sec": 4.0},
        ),
        StreamSample(
            sample_id="multi",
            text="hom nay troi dep va toi dang thu nghiem mo hinh",
            timestamps=[
                TimestampSpan(start_sec=0.0, end_sec=1.2, text="hom nay troi dep"),
                TimestampSpan(start_sec=1.7, end_sec=2.6, text="va toi dang"),
                TimestampSpan(start_sec=3.2, end_sec=4.4, text="thu nghiem mo hinh"),
            ],
            metadata={"duration_sec": 5.0},
        ),
    ]


def main() -> None:
    pipeline = RealtimeChunkedTTSPipeline(
        chunker=TimestampPassthroughChunker(),
        codec=MockMossCodec(),
        layer0_generator=MockLayer0Generator(),
        rvq_projector=MockRVQProjector(),
        config=PipelineConfig(random_seed=7),
    )
    for result in pipeline.process_stream(build_demo_samples()):
        selected_text = result.selected_span.text if result.selected_span else "<full-sample>"
        print(
            f"{result.sample.sample_id}: mode={result.chunk_mode}, "
            f"voice_clone_text={result.prompt.voice_clone_text!r}, "
            f"target_text={result.prompt.target_text!r}, selected={selected_text!r}, "
            f"phase1_tokens={len(result.phase1.layer0_codes)}, "
            f"phase2_shape={tuple(result.phase2.continuous_32_layers.shape)}"
        )


if __name__ == "__main__":
    main()
