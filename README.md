# Lumivoice

Realtime Vietnamese TTS pipeline with:
- phase 1 LLM over a shared text+audio vocabulary
- voice-clone conditioning with timestamp text plus MOSS continuous RVQ embeddings
- residual postnet for MOSS Nano acoustic continuous embeddings
- stream training/inference over Hugging Face audio datasets

## Tokenizer Asset

Default tokenizer asset:
- `assets/vi_wikipedia_bbpe_2048_copy.json`

All default configs use a relative path:
- `configs/train.default.json`
- `configs/inference.default.json`

## Config

Main config groups:
- `data`: input stream source such as `input_path`, `dataset_name`, or `dataset_names`
- `model`: tokenizer path, LLM size, Moss/postnet dimensions, chunk routing
- `train`: optimizer, curriculum loss schedule, checkpoint frequency
- `inference`: checkpoint path, output directory, sample limit

Default audio tokenizer:
- `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`
- `frame_rate_hz = 12.5`
- `num_layers = 16`
- `audio_codebook_size = 1024`
- tokenizer vocab has 2048 ids total
- layer-0 audio codes map to existing `[audio_token_0]` ... `[audio_token_1023]` ids

LLM training sequence:

```text
<VOICE_CLONE> clone text tokens + clone continuous RVQ embeddings </VOICE_CLONE>
<BOS> text chunk <AUDIO> layer-0 audio token ids </AUDIO> ... <EOS>
```

## Run

Train:

```bash
.venv/bin/python train.py --config configs/train.default.json
```

Train with 2 GPUs and timing breakdown:

```bash
.venv/bin/python -m torch.distributed.run --nproc_per_node=2 train_ddp.py \
  --config configs/train.default.json \
  --max-steps 10 \
  --output-dir ./outputs/train_ddp
```

Precompute timestamps for train-only preprocessing with ChunkFormer, then train from the materialized JSONL:

```bash
.venv/bin/python prepare_training_data.py \
  --config configs/train.default.json \
  --output-path ./outputs/preprocessed/train_with_timestamps.jsonl

.venv/bin/python train.py \
  --config configs/train.default.json \
  --input-path ./outputs/preprocessed/train_with_timestamps.jsonl
```

Inference:

```bash
.venv/bin/python inference.py --config configs/inference.default.json
```

## Streaming Datasets

Validated schema for streaming input:
- `capleaf/viVoice`: `audio`, `text`, `channel`
- `thivux/phoaudiobook`: `audio`, `text`, `speaker`

If streamed records do not already contain `timestamps`, use:
- `model.chunkers:ChunkFormerTimestampChunker` indirectly through `prepare_training_data.py`
- the pretrained model `khanhld/chunkformer-ctc-large-vie` to create timestamp spans before training

To interleave datasets, set:

```json
{
  "data": {
    "dataset_names": ["capleaf/viVoice", "thivux/phoaudiobook"]
  }
}
```
