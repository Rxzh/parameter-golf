# parameter-golf autoresearch

You are an autonomous LLM research loop. Your job: iteratively improve `train_gpt.py` to minimize `val_bpb` (bits-per-byte, post-quantization) on FineWeb validation, while keeping the compressed artifact ≤ 16,000,000 bytes and `train_gpt.py` ≤ 1500 lines.

**LOOP FOREVER. NEVER STOP.**

---

## Setup (do once at start of session)

1. Agree on a run tag (e.g. `golf-mar19`). Branch `autoresearch/<tag>` must not exist yet.
2. `git checkout -b autoresearch/<tag>` from `main`.
3. Read these files in full:
   - `CLAUDE.md` — repo context, constraints, key env vars
   - `train_gpt.py` — the ONLY file you edit: model, optimizer, hyperparameters, training loop
4. Verify data exists: `ls ./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin | wc -l` (need ≥ 10 shards)
5. Initialize `results.tsv`:
   ```
   commit	val_bpb	compressed_bytes	status	description
   baseline	1.224400	15863489	kept	baseline (NorMuon, polar_express)
   ```
6. Confirm setup complete, then begin the experiment loop.

---

## Experiment loop (LOOP FOREVER, NEVER STOP)

### 1. Check git state
```bash
git status
git log --oneline -5
```

### 2. Pick next experiment
Choose from the priority tiers below. Pick the highest-tier untried idea.

### 3. Edit `train_gpt.py`
One focused change. Do not bundle multiple hypotheses.
Check line count stays ≤ 1500: `wc -l train_gpt.py`

### 4. Commit
```bash
git add train_gpt.py
git commit -m "exp: <short description>"
```

### 5. Run (always this exact command)
```bash
TRAIN_BATCH_TOKENS=65536 MAX_WALLCLOCK_SECONDS=120 VAL_LOSS_EVERY=200 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py > run.log 2>&1
```

### 6. Read metrics
```bash
grep "^val_bpb:" run.log
grep "^compressed_bytes:" run.log
```

If either grep returns nothing: the run crashed. Check `tail -30 run.log` and fix.

### 7. Hard size gate
If `compressed_bytes > 16000000`: mark `size_fail`, `git reset --hard HEAD~1`, move on.

### 8. BPB gate
Compare `val_bpb` to current best in `results.tsv`.
- **val_bpb ≥ current best**: mark `discarded`, `git reset --hard HEAD~1`, move on.
- **val_bpb < current best**: KEEP. Update `results.tsv`, advance the branch.

### 9. Log to results.tsv
```
<commit_hash>	<val_bpb>	<compressed_bytes>	<kept|discarded|size_fail>	<description>
```

### 10. Strong win check
If improvement > 0.010 BPB: note it as a candidate for a full 8×H100 run.

---

## Output format

`train_gpt.py` emits these lines at the very end (rank 0 only):
```
---
val_bpb:          X.XXXXXX
compressed_bytes: NNNNNNN
```

These are the only lines you need. The `^val_bpb:` and `^compressed_bytes:` prefixes make them greppable.

---

## Constraints (hard — never violate)

| Constraint | Value |
|---|---|
| `compressed_bytes` ≤ | 16,000,000 |
| `VOCAB_SIZE` | 1024 (fixed by challenge) |
| `train_gpt.py` lines ≤ | 1500 |
| Edit only | `train_gpt.py` |
| Quantization code | READ-ONLY (do not touch lines after `POST-TRAINING QUANTIZATION` comment) |
| BPB evaluation logic | READ-ONLY |

---

## Experiment ideas (priority order)

### Tier 1 — Architecture scaling (use ~5.1M param headroom)
Baseline: 17.06M params, 15,863,489 bytes. ~136KB headroom ≈ ~5.1M more params.

| Idea | Param delta | Est. BPB gain | Notes |
|---|---|---|---|
| `MLP_MULT` 2→3 | +4.7M | 0.010–0.025 | Increases MLP width |
| `MODEL_DIM` 512→576 | +4.4M | 0.010–0.020 | Wider model |
| `NUM_LAYERS` 9→11 | +3.7M | 0.008–0.018 | Deeper model |

Try only one at a time. Combine after individual wins confirmed.

### Tier 2 — Architecture quality (same param count)

| Idea | Notes |
|---|---|
| Value embeddings | Add learned per-token value residual (see autoresearch/train.py) |
| SwiGLU MLP | Replace `relu²` with `silu(gate) * value`; add gate proj |
| QAT | Simulate int8 quantization noise during training to close the +0.0072 BPB quant gap |
| Windowed attention | Alternate short-context (128) and full-context layers |

### Tier 3 — Optimizer / schedule tuning

| Hyperparameter | Values to try |
|---|---|
| `MATRIX_LR` | 0.02, 0.03, 0.05, 0.06 |
| `MUON_WEIGHT_DECAY` | 0.1, 0.2, 0.3 |
| `MUON_MOMENTUM` | 0.90, 0.92, 0.97 |
| `warmdown_iters` | 800, 1000, 1500 |
| `MUON_BETA2` | 0.90, 0.99 |

### Tier 4 — Training dynamics

| Idea | Notes |
|---|---|
| `warmup_steps` 20→5 | Faster initial LR ramp |
| `TRAIN_BATCH_TOKENS=1048576` | Larger batch |
| `grad_clip_norm=1.0` | Add gradient clipping |

---

## Notes on the current baseline

The baseline (`main` branch after NorMuon port) uses:
- **NorMuon**: polar_express orthogonalization + variance normalization via second momentum buffer
- **Cautious weight decay**: only decays when update and weight have same sign
- 9 layers × 512 dim, GQA (8 query / 4 KV heads), ReLU² MLP (mult=2), RoPE, RMSNorm
- Tied input/output embeddings
- Vocab: 1024 (SentencePiece BPE)
- Adam for embeddings (lr=0.05), Muon for matrices (lr=0.04), Adam for scalars (lr=0.04)
- 20,000 iterations, 524,288 tokens/batch, 10-minute wall-clock cap

The proxy command uses 120s and 65536 tokens/batch — fast enough to rank experiments, not identical to full run. A proxy BPB improvement of ≥0.005 is reliable signal.

---

## Submitting a strong result

When a full 8×H100 run beats `val_bpb = 1.2244` by ≥ 0.005:
1. Copy `train_gpt.py` to `records/track_10min_16mb/<run_name>/train_gpt.py`
2. Add `README.md`, `submission.json`, `train.log` to that directory
3. Open a PR adding only the `records/` directory
