---
title: "Lecture 5 Study Notes — Quantization"
collection: class
permalink: /class/lecture5-notes/
---

> NYU Efficient DNN & ML Systems | Midterm Coverage: P1–P70

---

## 1. Basic Data Formats

### Fixed Point (INT)
Represents numbers as integers with a fixed number of bits and a scaling factor.

**Symmetrical Fixed Point** — how to convert real number x to INT:
1. Set clipping range: `(-L, L)` and bit-width `b`
2. Compute scale: `s = 2L / (2^b - 2)`
3. Clip input: `x_c = Clip(x, L, -L)`
4. Quantize: `x_int = round(x_c / s)`
5. Dequantize: `x_q = s · x_int`

Properties:
- Uniform spacing between representable values
- Range: `[-L, L]`; step size: `s`
- With `b` bits: `2^b` levels total

**Unsymmetrical (Asymmetric) Fixed Point**:
Allows a different range `[min, max]` with a zero-point offset:
```
x_int = round((x - zero_point) / scale)
x_q = scale · x_int + zero_point
```
More flexible — can represent skewed distributions (e.g., ReLU outputs that are all non-negative).

### Floating Point (FP)
Stores a number as: `x = s · 2^a · (1 + m)` where:
```
s = sign(x)
a = ⌊log₂|x|⌋        (exponent / characteristic)
e = a + bias          (biased exponent, stored in bits)
m = x/2^a - 1         (mantissa, stored in bits)
```

Properties:
- **Non-uniform**: more precision for values near zero, less for large values
- Better representation power for values with **small magnitudes**
- Standard formats: FP32 (1s + 8e + 23m), FP16 (1s + 5e + 10m), BF16 (1s + 8e + 7m)

### Block Floating Point (BFP)
A group of numbers share a **common exponent**, each has its own mantissa:
```
Group: [x₁, x₂, ..., xₙ]
Shared exponent: max_exp = max(⌊log₂|xᵢ|⌋)
Each: xᵢ = shared_exp × mantissa_i
```
Compromise between INT (cheap) and FP (flexible). Reduces the cost of storing individual exponents.

---

## 2. Why These Formats Save Computation

| Format | Why It Saves Compute |
|---|---|
| **INT (fixed-point)** | Integer multiplications are cheaper and faster than FP; INT8 can pack 4× more ops than FP32 in SIMD units |
| **FP (low precision)** | FP16/BF16 use half the bits → 2× memory bandwidth; modern GPUs have dedicated FP16 tensor cores (e.g. 2× throughput) |
| **BFP** | Shared exponent reduces mantissa bits needed; allows efficient vectorized integer ops with one exponent lookup |
| **Log quantization** | Multiplication becomes addition in log space: `log(ab) = log(a) + log(b)`; replaces expensive multiplications with additions |

---

## 3. Quantization: Symmetric vs. Unsymmetric

**Symmetric**: clipping range symmetric around 0 → `[-L, L]`
- Scale: `s = 2L / (2^b - 2)`
- Zero-point is always 0
- Simpler math; slight waste for asymmetric distributions

**Unsymmetric (Asymmetric)**: range `[min, max]`
- Scale: `s = (max - min) / (2^b - 1)`
- Zero-point: `z = round(-min/s)`
- Better for distributions like ReLU outputs (all positive)

---

## 4. Straight Through Estimator (STE)

**The Problem**: Quantization is a staircase (piecewise constant) function — its derivative is 0 almost everywhere, making backpropagation through it impossible.

```
Forward:  W' = Q(W) = round(W/s)·s    (staircase)
∂W'/∂W = 0 almost everywhere → gradient is zero → no learning
```

**STE Solution**: During the **backward pass**, approximate ∂W'/∂W = 1 (identity):
```
∂W'/∂W ≈ 1    (straight through — ignore the quantization step)
```

**Gradient flow with STE:**
```
∂L/∂W = ∂L/∂Z · ∂Z/∂Y · ∂Y/∂W' · ∂W'/∂W
                                        ↑
                                STE sets this = 1
```

So: `∂L/∂W ≈ ∂L/∂W'` — pass the gradient straight through the quantizer.

**Forward pass**: apply quantization (use W').
**Backward pass**: ignore quantization (treat it as identity).

---

## 5. When to Quantize: PTQ vs. QAT

### Post-Training Quantization (PTQ)
```
1. Train model with full precision (FP32)
2. Quantize weights (and optionally activations) after training
```
- Lower computational cost (no retraining)
- Accuracy drop, especially at very low bit-widths (INT4 or below)
- Used for LLMs (too expensive to retrain): GPTQ, AWQ, etc.

### Quantization-Aware Training (QAT)
```
1. Quantize weights/activations during forward pass
2. Use STE for backward pass
3. Train until convergence
4. Deploy quantized model
```
- Better accuracy than PTQ (model adapts to quantization noise during training)
- Higher training cost
- Produces quantized model directly

**Rule of thumb**: PTQ preferred for large models (LLMs) due to training cost. QAT preferred when highest accuracy is required and retraining is feasible.

---

## 6. Taxonomy of Quantization

| Dimension | Options |
|---|---|
| **Timing** | Post-training (PTQ) vs. Quantization-aware training (QAT) |
| **Granularity** | Per-tensor vs. Per-channel vs. Per-group |
| **What is quantized** | Weights only vs. Weights + activations |
| **Bit-width** | 8-bit (INT8), 4-bit (INT4), mixed-precision |
| **Symmetry** | Symmetric vs. Asymmetric |

**Per-channel quantization**: each output channel of a weight tensor has its own scale/zero-point → better accuracy than per-tensor.

**Mixed-precision**: different layers use different bit-widths (e.g. sensitive layers keep FP16; other layers use INT8).

---

## 7. Quantization During Training — Stochastic Quantization

Instead of deterministic rounding, use **probabilistic rounding**:
```
x_int = ⌊x/s⌋ + Bernoulli(x/s - ⌊x/s⌋)
```
The fractional part becomes the probability of rounding up. This introduces noise that acts as regularization and helps the model generalize better to quantization.

---

## 8. Learnable Adaptive Quantization

Instead of fixed clipping ranges and scales, **learn them** as part of training:
```
scale s, clip range L  →  treated as learnable parameters
∂L/∂s computed with STE  →  s updated via gradient descent
```

Allows the network to find the optimal quantization parameters per layer, improving accuracy especially at very low bit-widths (4-bit, 2-bit).

Examples: **PACT** (learns the clipping threshold α), **LSQ** (learned step size).

---

## Quick Reference

| Format | Uniform? | Cheap multiply? | Precision distribution |
|---|---|---|---|
| INT (fixed-point) | Yes | Yes (integer) | Uniform |
| FP | No | No (float) | High near 0, low for large |
| BFP | Approx | Partially | Group-shared exponent |
| Log | No | Yes (add in log) | More near 0 |

## Common Exam Traps

- STE sets ∂W'/∂W = **1**, not the actual derivative (which is 0)
- STE is used in the **backward pass only**; forward still uses quantized values
- PTQ is preferred for LLMs because retraining is **too expensive**, not because it's more accurate
- Symmetric quantization has zero-point = 0; asymmetric does **not**
- BFP is not the same as FP: it groups values sharing a **common exponent**
- Stochastic quantization: rounding probability = **fractional part** of x/s
- Floating point has **better precision for small values** (dense spacing near 0)
