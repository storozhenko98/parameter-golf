# Development Log — mykyta_v1 Submission

## Goal
Beat the current SOTA (1.1147 BPB) on OpenAI Parameter Golf.
Target: sub-1.10 BPB within 16MB artifact, 10 minutes on 8xH100.

## Base
Starting from PR #1019 (abaybektursun, 2026-03-25, 1.1147 BPB).

## Planned Changes (priority order)

### Tier 1 — High Impact
1. [x] **SLOT eval** — per-batch delta optimization at eval time (-0.010 to -0.021 BPB)
2. [x] **Brotli-11 + byte-shuffle compression** — frees ~400KB vs LZMA-9
3. [x] **Mixed int5/int6 quantization** — int5 for MLP, int6 for attention
4. [x] **MLP 3.5x** — use freed bytes from int5+Brotli for wider MLP

### Tier 2 — Medium Impact
5. [x] **QK-Gain 4.0** — single hyperparameter, validated across 45 experiments (-0.006 BPB)
6. [x] **Split early/late LR** — higher Muon LR (0.030) for late layers (>=6), early stays 0.025
7. [x] **Soft-round QAT** — sigmoid rounding replacing STE, real gradients through rounding
8. [ ] **Code minification** — deferred, code at 114KB is acceptable
9. [x] **Warmdown 4000** — from 3500, validated in SOTA's actual runs

### Tier 3 — Speculative (not yet implemented)
10. [ ] **EGGROLL post-quant refinement** — zeroth-order bin search during eval
11. [ ] **EngramLite** — multi-head bigram+trigram hash
12. [ ] **ASQU per-layer activation slopes** — different LeakyReLU slopes per layer

---

## Change Log

### 2026-03-31 — Session 1: Initial Implementation

**Setup:**
- Created branch `submission/mykyta-v1`
- Created submission folder `2026-03-31_mykyta_v1/`
- Copied SOTA train_gpt.py (2135 lines, 101KB) as starting point

**Changes implemented (7 total):**

1. **QK-Gain 4.0** (line 45)
   - Changed default `QK_GAIN_INIT` from 1.5 to 4.0
   - Source: PR #1176, validated across 45 experiments, expected -0.006 BPB

2. **Warmdown 4000** (line 39)
   - Changed default `WARMDOWN_ITERS` from 3500 to 4000
   - The SOTA submission's actual runs used 4000 via env var

3. **Split early/late Muon LR** (lines 60-61, optimizer setup in main)
   - Added `LATE_MATRIX_LR=0.030` for layers >= `LATE_LAYER_START=6`
   - Early layers (0-5) use standard `MATRIX_LR=0.025`
   - Implemented via per-slice LR multipliers in Muon optimizer
   - Source: PR #1179, expected -0.002 to -0.003 BPB

4. **Brotli-11 + byte-shuffle compression** (serialization section)
   - Added brotli import with fallback to LZMA-9
   - Byte-shuffle (stride=2) before compression for better ratio
   - Auto-selects best of Brotli-11 vs LZMA-9
   - Source: PR #1179, expected ~400KB savings
   - Updated selective pruning to use best compressor too

5. **Mixed int5/int6 quantization** (quantization section)
   - MLP weights: int5 (clip_range=15, range [-16,15])
   - Attention weights: int6 (clip_range=31, range [-32,31])
   - MLP tolerates int5 because relu-squared creates sparse activations
   - Source: PR #1105, Int5MLP submission

6. **MLP 3.5x width** (line 51)
   - Changed `MLP_MULT` from 3.0 to 3.5
   - Funded by int5 MLP savings + Brotli compression improvement
   - MLP is the capacity bottleneck (94.4% SVD rank utilization per PR #1105)

7. **Soft-round QAT** (CastedLinear class)
   - Replaced hard STE rounding with sigmoid soft-rounding
   - `soft_rounded = floor(scaled) + sigmoid(alpha * (frac - 0.5))`
   - Alpha ramps from 1 (smooth) to 16 (near-hard) over 500 steps
   - Source: PR #1179, provides real gradients through rounding

8. **SLOT eval** (new function + eval section)
   - Added `eval_val_sliding_slot()`: per-batch delta optimization on hidden states
   - After forward pass, optimizes R^512 delta vector via AdamW for 6 steps
   - Delta is added to hidden states before logit projection
   - SLOT_ENABLED=1, SLOT_STEPS=6, SLOT_LR=0.01
   - Source: arXiv:2505.12392v2, PR #1176, expected -0.010 to -0.021 BPB

**File stats:**
- Code: 2393 lines, 114,552 bytes (vs SOTA's 2135 lines, 101,850 bytes)
- Syntax check: PASSED

**Expected impact (conservative estimates):**
- QK-Gain 4.0: -0.006 BPB
- Split LR: -0.002 BPB
- MLP 3.5x (funded by int5+Brotli): -0.001 to -0.002 BPB
- SLOT eval: -0.010 to -0.021 BPB
- Soft-round QAT: -0.001 BPB
- **Total estimated: -0.020 to -0.032 BPB → target ~1.083 to ~1.095 BPB**

**Risks:**
- MLP 3.5x + int5 may push artifact over 16MB — need to tune TARGET_MB
- SLOT legality is debated (causality of shared delta) — may need to drop
- Soft-round QAT interaction with bank weights is untested
- Brotli may not be pre-installed on RunPod template

**Next steps:**
- Set up RunPod 8xH100 environment
- Run 3-seed validation (seeds 42, 314, 999)
- If over 16MB: reduce BigramHash size or increase TARGET_MB pruning
- If SLOT is ruled illegal: drop it, still expect ~1.10-1.11 BPB
