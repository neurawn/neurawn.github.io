---
title: "Lecture 7 Study Notes — LLM Optimization"
collection: class
permalink: /class/lecture7-notes/
---

> NYU Efficient DNN & ML Systems | Midterm Coverage: P1–P41, P53–P68

---

## 1. Outlier Distribution in LLMs

Large language models (e.g. CLIP, LLaMA, GPT-2) exhibit an unusual phenomenon: a small number of **activation values** are **orders of magnitude larger** than the rest.

### Massive Activations

Certain positions in the activation tensors have enormously large magnitudes — the paper calls these **massive activations**.

From CLIP model study:
- Positions x₁, y₁, and y₅ (specific spatial/token positions) exhibit outlier activations with magnitudes far above the rest
- These outlier values appear consistently across samples
- **Truncating or modifying** these massive activations (e.g., setting to mean or zero) causes **catastrophic accuracy degradation**

```
Experiment (LLaMA3.2-3B, WikiText perplexity):
  Original:                   5.567
  Set massive activation to mean:  1,124,111.75   ← catastrophic
  Set to zero:                1,138,151.23   ← catastrophic
```

The model has learned to rely on these massive activations as critical information carriers.

### Channelwise Outliers

**x₂** (the key computation input in attention) exhibits a different pattern: **channelwise outliers** — certain channels consistently have large values across all sequence positions and layers.

- 3D plots of x₂ across layers show a **single bright column** (one channel) dominating
- This pattern persists across layers 1-23 (shown in CLIP)
- Channelwise outliers are structured → quantizing the full tensor with a single scale factor leads to poor precision for the "normal" channels

### Why Outliers Matter for Quantization

Quantization uses a scale factor `s` covering the full range `[-L, L]`:
- If one activation is 1000× the typical value → `L` must be set very large
- Normal activations all fall into the bottom few quantization bins → effectively 0-bit precision for them
- Result: massive quantization error on the majority of values

---

## 2. Quantization and Smoothing Techniques for Large Models

### Per-Channel Quantization
Instead of one scale per tensor, use **one scale per channel**:
```
s_c = max(|X[:, c, :]|) / (2^(b-1) - 1)    for each channel c
x_int[c] = round(X[:, c, :] / s_c)
```
Each channel gets its own clipping range → handles channelwise outliers naturally.

But: weight and activation quantization must use compatible granularities — per-channel activation quantization makes the matmul more complex.

### SmoothQuant
**Key idea**: Migrate the quantization difficulty from activations to weights by scaling:
```
Y = X · W = (X · diag(s)⁻¹) · (diag(s) · W) = X̃ · W̃
```
Choose scale `s_c` to balance the ranges:
```
s_c = max(|X[:, c]|)^α / max(|W[c, :]|)^(1-α)    (α ≈ 0.5)
```
- Divide activations by s → smooth out outlier channels
- Multiply corresponding weight rows by s → absorb the scale
- Now both X̃ and W̃ have similar magnitudes → per-tensor quantization works well

**Result**: activation outliers are "smoothed" into the weights, which are easier to quantize (weights are static and can be calibrated offline).

### GPTQ (Post-Training Quantization for LLMs)
Uses Hessian-based (second-order) information to minimize quantization error:
- Quantize weights column by column
- After quantizing each weight, update remaining weights to compensate for the introduced error
- Achieves INT4 quantization with minimal accuracy loss for large models
- No gradient computation needed — pure PTQ using a small calibration dataset

### AWQ (Activation-Aware Weight Quantization)
Observes that not all weights are equally important — weights corresponding to **large activation channels** are more important:
```
Protect salient weights: scale them up before quantization
Scale activations down accordingly (absorbed into prior layer)
```
Achieves INT4 quantization while preserving the important weight-activation pairs.

---

## 3. LLM Pruning

### Why LLM Pruning is Harder
- LLMs have hundreds of billions of parameters
- Standard iterative pruning (prune → fine-tune → repeat) is too expensive
- Need one-shot or post-training pruning methods

### Magnitude-Based (Weight Pruning)
Remove weights with smallest absolute values — same as standard pruning.
Challenge: LLMs have outlier weights (large magnitude but potentially still unimportant in context).

### SparseGPT
Post-training unstructured pruning using Hessian information:
- Prune weight columns one by one
- After pruning each weight, update remaining weights in the same row to compensate
- Similar idea to GPTQ but for pruning instead of quantization
- Achieves 50% sparsity on GPT-175B with minimal perplexity increase

### Wanda (Pruning by Weights and Activations)
Prune weights based on the **product of weight magnitude and input activation norm**:
```
importance(w_ij) = |w_ij| · ||X_j||₂
```
Intuition: a weight is unimportant if it's small OR if its input channel has small activations.
Faster than SparseGPT (no Hessian computation).

### Structured LLM Pruning
Remove entire attention heads, FFN neurons, or transformer layers:
- Provides direct hardware speedup (no sparse hardware needed)
- More accuracy loss than unstructured at the same compression ratio
- Post-pruning fine-tuning (if affordable) recovers accuracy

---

## 4. Low-Rank Decomposition for LLM

### SVD Applied to LLM Weight Matrices

Attention projection matrices (W_q, W_k, W_v, W_out) each of size E×E can be decomposed:
```
W ≈ W₁ · W₂     where W₁: (E×r), W₂: (r×E), r << E
```

Original attention block:
```
x₁ → LayerNorm → z
z → W_q → Q → RoPE
z → W_k → K → RoPE  →  Softmax(QKᵀ)  →  ×V → W_out → x₂
z → W_v → V
```

With SVD decomposition:
```
z → W_dq·W_uq → Q → RoPE       (buffered: W_dq output)
z → W_dk·W_uk → K → RoPE       (buffered: W_dk output)
z → W_dv·W_uv → V              (buffered: W_dv output)
```

**Benefits of SVD in attention**:
- **MAC reduction**: E² → 2Er per projection
- **KV cache size reduction**: K and V are computed via (E×r) projection → cached at dimension r instead of E
- **Memory savings**: weight storage E² → 2Er

**Requirement**: rank r must satisfy `r < E/2` for parameter reduction.

### LoRA (Low-Rank Adaptation) — Fine-Tuning Perspective
Freeze original weights W; add low-rank update:
```
W_new = W + ΔW = W + A·B    where A: (E×r), B: (r×E)
```
Only A and B are trained during fine-tuning → drastically fewer trainable parameters.
At inference: merge `W_new = W + AB` → no extra cost.

---

## Quick Reference

| Problem | Technique | Key Idea |
|---|---|---|
| Channelwise outliers in activations | Per-channel quantization | One scale per channel |
| Activation outliers hard to quantize | SmoothQuant | Migrate scale to weights |
| LLM weight quantization (PTQ) | GPTQ | Hessian-based column-wise PTQ |
| Activation-aware quantization | AWQ | Protect salient weight channels |
| LLM unstructured pruning (PTQ) | SparseGPT | Hessian-based row compensation |
| LLM pruning (magnitude + activation) | Wanda | |w|·||X|| importance score |
| Reduce attention params + KV cache | SVD of W_q,k,v | W ≈ W₁W₂ with r << E |

## Common Exam Traps

- Massive activations are at **specific positions** (e.g. x₁, y₁, y₅), not specific channels
- Channelwise outliers are at **specific channels** (e.g. x₂, one channel consistently bright)
- Truncating massive activations causes **catastrophic** accuracy loss — they are critical
- SmoothQuant: multiplies activation by **s⁻¹** and weights by **s** (scales cancel in product)
- GPTQ/SparseGPT both update **remaining weights** after each quantization/pruning step to compensate
- SVD without rank truncation: parameter count **increases** — must truncate to r << min(m,n)
- KV cache with SVD: cache stores the **low-rank** K, V vectors (size r not E) → cache is smaller
