# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Parameter Golf is an OpenAI Model Craft Challenge to train the best language model fitting within a **16MB artifact** (code + compressed model) that trains in **under 10 minutes on 8×H100 GPUs**. The optimization target is bits-per-byte (BPB) on the FineWeb validation set — lower is better.

## Common Commands

### Local Development (Apple Silicon Mac)

```bash
# Install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm

# Download dataset (10 shards for smoke tests)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Quick smoke test run
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py
```

### GPU Training

```bash
# Install dependencies
pip install -r requirements.txt

# Download full dataset
python3 data/cached_challenge_fineweb.py --variant sp1024

# Single H100
RUN_ID=baseline_sp1024 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 torchrun --standalone --nproc_per_node=1 train_gpt.py

# 8×H100 (competition)
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Set `MAX_WALLCLOCK_SECONDS=0` to disable the default 600-second (10-minute) training cap.

## Architecture

### Model (GPT Transformer, ~17M params by default)
- 9 layers × 512 dim, RMSNorm, RoPE, causal attention
- **Grouped Query Attention**: 8 query heads, 4 KV heads
- **Tied input/output embeddings** (halves embedding parameter count)
- ReLU² MLP with 2× expansion
- Default vocabulary: 1024 (SentencePiece BPE)

### Optimizer Strategy (three groups)
1. **Token embedding** → Adam at `EMBED_LR=0.6` (or `TIED_EMBED_LR=0.05` when tied)
2. **Matrix-shaped parameters** → Muon (orthogonalizes gradients via Newton-Schulz) at `MATRIX_LR=0.04`
3. **Scalars/vectors** → Adam at `SCALAR_LR=0.04`

### Quantization & Submission Size
After training, weights are quantized to int8 + zlib compressed:
- 2D matrices: per-row int8 scales
- Vectors/scalars: per-tensor int8 scales
- Control parameters (scales, skip weights): kept as float16
- **Submission = Bytes(compressed model) + Bytes(code) ≤ 16,000,000**

### Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VOCAB_SIZE` | 1024 | Tokenizer vocabulary size |
| `NUM_LAYERS` | 9 | Transformer depth |
| `MODEL_DIM` | 512 | Hidden dimension |
| `NUM_HEADS` | 8 | Query attention heads |
| `NUM_KV_HEADS` | 4 | KV attention heads (GQA) |
| `MLP_MULT` | 2 | MLP expansion factor |
| `ITERATIONS` | 20000 | Training steps |
| `TRAIN_BATCH_TOKENS` | 524288 | Tokens per batch |
| `MAX_WALLCLOCK_SECONDS` | 600 | Training time cap (0 = unlimited) |

## Repository Structure

- `train_gpt.py` — Main CUDA/distributed training script
- `train_gpt_mlx.py` — Apple Silicon MLX variant for local development
- `data/` — Dataset download, tokenizer specs, and custom tokenization utilities
- `records/` — Competition submissions (track_10min_16mb and track_non_record_16mb)

## Evaluation

- **Metric**: `val_bpb` (bits-per-byte) — tokenizer-agnostic compression quality
- **Validation set**: First 50k documents of FineWeb
- Submissions must beat existing SOTA by ≥0.005 nats with p < 0.01 statistical confidence

## Submissions

New submissions go in `records/track_10min_16mb/` (≤10 min) or `records/track_non_record_16mb/` (unlimited compute). Each submission directory requires: `README.md`, `submission.json`, `train.log`, `train_gpt.py`, and any additional dependencies. Submit via PR that adds only to `/records`.

## Autoresearch Loop

Baseline to beat: `val_bpb = 1.2244` (post-quantization), artifact size 15,863,489 bytes (~136KB headroom).

### Proxy Command (2-minute single-GPU run)
```bash
TRAIN_BATCH_TOKENS=65536 MAX_WALLCLOCK_SECONDS=120 VAL_LOSS_EVERY=200 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py > proxy.log 2>&1

grep "final_int8_zlib_roundtrip" proxy.log | tail -1
grep "Total submission size int8+zlib" proxy.log | tail -1
```

See `program_golf.md` for the full experiment loop, priority tiers, and keep/discard logic.
