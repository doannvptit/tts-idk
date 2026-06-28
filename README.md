# Lumivoice

Realtime Vietnamese TTS pipeline with:
- phase 1 LLM for `codebook layer 0 + </audio>`
- residual postnet for Moss acoustic continuous embeddings
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

## Run

Train:

```bash
.venv/bin/python train.py --config configs/train.default.json
```

Inference:

```bash
.venv/bin/python inference.py --config configs/inference.default.json
```

## Streaming Datasets

Validated schema for streaming input:
- `capleaf/viVoice`: `audio`, `text`, `channel`
- `thivux/phoaudiobook`: `audio`, `text`, `speaker`

To interleave datasets, set:

```json
{
  "data": {
    "dataset_names": ["capleaf/viVoice", "thivux/phoaudiobook"]
  }
}
```
