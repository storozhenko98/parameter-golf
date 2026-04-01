# Record: QKGain4 + SplitLR + SoftRoundQAT + Brotli + BigramHash3072

**val_bpb: TBD** (3-seed mean) | **~15.X MB** | 8xH100 SXM, 600s | No SLOT, no TTT

## Changes from SOTA (PR #1019, 1.1147 BPB)

### 1. QK-Gain 4.0 (from 1.5)
Validated across 45 experiments (PR #1176). Expected -0.006 BPB.

### 2. Split Early/Late Muon LR
Late layers (>=6) use MATRIX_LR=0.030, early layers use 0.025.
Late layers receive weaker gradient signal, benefit from higher LR (PR #1179).

### 3. Soft-Round QAT
Replaces hard STE rounding with temperature-controlled sigmoid:
`soft_rounded = floor(scaled) + sigmoid(alpha * (frac - 0.5))`.
Alpha ramps from 1 to 16 over 500 steps after late QAT activates.

### 4. Brotli-11 + Byte-Shuffle Compression
Auto-selects best of Brotli quality=11 with stride-2 byte-shuffle vs LZMA-9.
Saves ~400-500KB vs LZMA (PR #1179).

### 5. BigramHash 3072x128 (from SOTA's 3072x112)
Wider per-bucket projection (128 vs 112) using Brotli headroom.

### 6. Warmdown 4000 (from 3500)

## Architecture

Same as PR #1019 stack: 11L/512d, 3x MLP with LeakyReLU(0.5)^2, XSA-all,
Full Hessian GPTQ int6 with AR self-gen calibration, EMA(0.997), Partial RoPE,
LN Scale, SmearGate, U-Net skips, Parallel Muon, sliding window eval stride=64.

## Run Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results

| Seed | Steps | ms/step | Sliding BPB | Artifact |
|------|-------|---------|-------------|----------|
| 314 | TBD | TBD | TBD | TBD |
| 42 | TBD | TBD | TBD | TBD |
| 999 | TBD | TBD | TBD | TBD |
| **Mean** | | | **TBD** | |
