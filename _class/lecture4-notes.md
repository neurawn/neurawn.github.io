---
title: "Lecture 4 Study Notes — Pruning"
collection: class
permalink: /class/lecture4-notes/
---

> NYU Efficient DNN & ML Systems | Midterm Coverage: P1–P65, P68–P75

---

## 1. Computational Cost Saving with Pruning

**Pruning** removes weights (or entire structures) from a network to reduce compute and memory, while maintaining acceptable accuracy.

### CNN Pruning
For a Conv2D layer with p% of weights pruned (set to zero) and q% of input activations zero:

**Number of nonzero MACs ≥ (1-p)(1-q) × M×C×R×S×E×F**

- p% weight sparsity: reduces the number of weight multiplications
- q% input sparsity: if input is also sparse (e.g. after ReLU), further reduces MACs
- Both together: `(1-p)(1-q)` fraction of MACs remain nonzero

For just weight pruning (q=0): MACs reduced by factor (1-p).

**Important**: Zero weights only save compute if the hardware/software can exploit sparsity. Dense hardware (standard GPUs) may not benefit unless sparsity is structured.

### Transformer Pruning
Pruning can be applied to:
- **Attention heads** (head pruning)
- **Individual tokens** in the sequence (token pruning)

---

## 2. Sparse Matrix Encoding

When a weight matrix is sparse, store only the non-zero elements efficiently.

### Bitmap Encoding
Store a binary mask (1=nonzero, 0=zero) alongside the nonzero values.
- Mask size: n bits for n weights
- Then store only the nonzero values consecutively

### Run-Length Encoding (RLE)
Encode consecutive zeros as a (count, value) pair:
- e.g. `[0, 0, 0, 5, 0, 3]` → `[(3, 0), (1, 5), (1, 0), (1, 3)]`
- Efficient when zeros form long runs (high structured sparsity)

### Coordinate Format (COO)
Store each nonzero as (row, col, value) triple:
- Flexible, works for any sparsity pattern
- Overhead: 3× the storage per nonzero entry compared to just the value
- Efficient when sparsity is very high (most values zero)

---

## 3. General Pruning Techniques

### Magnitude Pruning
Remove weights with the smallest absolute values:
```
prune weight w_i  if  |w_i| < threshold
```
Simple and effective. Intuition: small magnitude weights contribute little to the output.

**One-shot**: prune once after training, then fine-tune.
**Iterative**: prune a little → fine-tune → repeat (finds a better sparse solution).

### Gradient-Based Pruning
Use the gradient of the loss w.r.t. each weight as the importance score:
```
importance(w_i) = |∂L/∂w_i|  or  |w_i · ∂L/∂w_i|
```
Weights whose removal barely changes the loss are pruned first.

### Hessian-Based Pruning (OBD / OBS)
Uses the **second-order Taylor expansion** of the loss:
```
ΔL ≈ ½ · δwᵀ · H · δw
```
Where H is the Hessian (second derivative matrix). Prune the weight whose removal causes the smallest ΔL — more accurate than magnitude pruning but expensive to compute.

**OBD (Optimal Brain Damage)**: assumes H is diagonal (independent weights).
**OBS (Optimal Brain Surgeon)**: uses the full inverse Hessian; adjusts remaining weights to compensate.

### Lasso Regularization (L1)
Add an L1 penalty to encourage sparsity:
```
L_total = L_task + λ · Σ|w_i|
```
L1 promotes exact zeros (sparsity) unlike L2 (weight decay) which just shrinks values.

### Taxonomy of Pruning

| Dimension | Options |
|---|---|
| **Granularity** | Unstructured (individual weights) vs. Structured (filters, channels, layers) |
| **Timing** | Post-training vs. During training vs. Before training (lottery ticket) |
| **Criterion** | Magnitude, gradient, Hessian, learned mask |
| **Schedule** | One-shot vs. Iterative |

**Unstructured**: highest compression, but sparsity is irregular → hard to accelerate on standard hardware.
**Structured**: removes entire filters/channels/heads → regular structure → direct speedup on any hardware.

### Network Slimming
Uses **channel-level pruning** guided by BatchNorm scale parameters (γ):
- Add L1 penalty on γ: `L_total = L + λ · Σ|γ_i|`
- γ values near 0 → those channels are unimportant → prune them
- Clean structured pruning: remove entire channels, get a smaller dense network

### N:M Sparsity
For every M consecutive weights, exactly N are nonzero (the rest are zero):
- **2:4 sparsity** (NVIDIA Ampere): 2 nonzeros per 4 weights = 50% sparsity
- Hardware-friendly: GPU has dedicated sparse tensor cores for 2:4 patterns
- 2× throughput compared to dense on Ampere GPUs

### Cascade Effect of Pruning
Pruning in one layer affects the utility of weights in adjacent layers:
- If a filter in layer l is pruned → its output channels are gone → the corresponding input channels of layer l+1 become useless → can also be pruned
- This cascade allows aggressive compression when pruning is done jointly across layers

---

## 4. Transformer Pruning

### Token Pruning
Remove unimportant tokens from the sequence during processing:
- Early transformer layers identify "unimportant" tokens (low attention score to [CLS])
- Those tokens are dropped before being passed to deeper layers
- Reduces sequence length L → quadratically reduces attention cost O(L²)
- Final classification uses only the remaining tokens

Used in: **PoWER-BERT**, **SpAtten**, **TR-BERT**

### Head Pruning
Remove entire attention heads from Multi-Headed Attention:
- Score each head by its importance (e.g., gradient-based or attention entropy)
- Prune the least important heads
- Reduces compute and memory proportionally to number of heads removed
- Each head: (B, L, E/h) for Q, K, V — removing a head removes E/h dimensions

---

## Quick Reference

| Pruning Type | Granularity | Hardware Friendly? |
|---|---|---|
| Weight pruning (unstructured) | Individual weights | No (irregular) |
| Filter/channel pruning | Entire filters/channels | Yes (dense network) |
| N:M sparsity (2:4) | Fixed pattern blocks | Yes (NVIDIA Ampere) |
| Token pruning | Sequence positions | Yes (shorter sequence) |
| Head pruning | Attention heads | Yes (fewer heads) |

## Common Exam Traps

- Pruning p% of weights reduces MACs by **at most** p% — only if hardware exploits sparsity
- With both weight sparsity p and input sparsity q: MACs ≥ **(1-p)(1-q)** × full MACs
- Magnitude pruning removes **small** weights, not large ones
- Network Slimming pruning signal comes from **BN γ parameters**, not weight magnitudes
- N:M sparsity: 2:4 means **2 nonzeros** out of **4** weights (50% sparsity), NOT 2 zeros per 4
- Token pruning reduces cost quadratically (O(L²) attention); head pruning reduces linearly
