# Record: SLOT + Mixed Int5/6 + MLP3.5x + QKGain4 + SplitLR + Brotli

**val_bpb: TBD** (3-seed mean) | **~15.X MB** | 8xH100 SXM, 600s

## Changes from SOTA (PR #1019, 1.1147 BPB)

### 1. SLOT Eval-Time Optimization
Per-batch delta optimization on frozen hidden states (arXiv:2505.12392v2).
After forward pass produces hidden states H, optimize a learned delta vector
d in R^512 via 6 AdamW steps on `loss(H + d)`. Zero parameter cost.

### 2. Mixed int5/int6 Quantization
MLP weights quantized to int5 ([-16,15]) instead of int6 ([-32,31]).
Attention weights remain int6. MLP tolerates lower precision due to
relu-squared creating sparse activations (1.88x compression ratio).

### 3. MLP 3.5x Width (from 3.0x)
SVD rank utilization analysis (PR #1105) showed MLP at 94.4% vs attention Q at
72.6%. Wider MLP funded by int5 compression savings + Brotli improvement.

### 4. QK-Gain 4.0 (from 1.5)
Single hyperparameter change validated across 45 experiments (PR #1176).

### 5. Split Early/Late Muon LR
Late layers (>=6) use MATRIX_LR=0.030, early layers use 0.025.
Late layers receive weaker gradient signal, benefit from higher LR (PR #1179).

### 6. Brotli-11 + Byte-Shuffle Compression
Replaces LZMA-9 when it produces smaller output. Stride-2 byte-shuffle
preprocessing before Brotli quality=11. Saves ~400KB vs LZMA (PR #1179).

### 7. Soft-Round QAT
Replaces hard STE rounding with temperature-controlled sigmoid:
`soft_rounded = floor(scaled) + sigmoid(alpha * (frac - 0.5))`.
Alpha ramps from 1 (smooth) to 16 (near-hard) over 500 steps.
Provides real gradients through rounding (PR #1179).

### 8. Warmdown 4000 (from 3500)
Longer warmdown gives more time for weight averaging to converge.

## Architecture (inherited from PR #1019 stack)

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) |
| MLP | **3.5x (1792)** with LeakyReLU(0.5)^2 |
| Attention | XSA on all 11 layers |
| BigramHash | 2048 x 128 |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/sqrt(layer+1) |
| VE128 | Layers 9-10 |
| SmearGate | Position-mixing gate |
| U-Net skips | Encoder-decoder connections |
| Weight avg | EMA(0.997) + SWA(every 50) |
| Quantization | **Full Hessian GPTQ: int5 MLP / int6 attention** |
| Compression | **Brotli-11 + byte-shuffle (or LZMA-9 fallback)** |
| Eval | **SLOT (6 AdamW steps on hidden delta)** + sliding window stride=64 |

## Requirements

```bash
pip install --break-system-packages flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
pip install sentencepiece zstandard brotli
```

## Run Command

```bash
BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 WARMDOWN_ITERS=4000 \
TARGET_MB=15.9 SEED=314 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results

| Seed | Steps | ms/step | Pre-quant BPB | Sliding BPB | SLOT BPB | Artifact |
|------|-------|---------|---------------|-------------|----------|----------|
| 314 | TBD | TBD | TBD | TBD | TBD | TBD |
| 42 | TBD | TBD | TBD | TBD | TBD | TBD |
| 999 | TBD | TBD | TBD | TBD | TBD | TBD |
| **Mean** | | | | | **TBD** | |
