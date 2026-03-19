# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `train.py` — the file you can modify, you can modify: Model architecture, optimizer, training loop. DO NOT MODIFY: data prep, tokenizer, dataloader, evaluation.
4. **Verify data exists**
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.


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

## Constraints (hard — never violate)

| Constraint | Value |
|---|---|
| `compressed_bytes` ≤ | 16,000,000 |
| `VOCAB_SIZE` | 1024 (fixed by challenge) |
| `train_gpt.py` lines ≤ | 1500 |``
| Training time ≤ | 10 minutes |
| Edit only | `train_gpt.py` |
| Quantization code | READ-ONLY (do not touch lines after `POST-TRAINING QUANTIZATION` comment) |
| BPB evaluation logic | READ-ONLY |

---

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train_gpt.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train_gpt.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!