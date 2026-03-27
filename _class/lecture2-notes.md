---
title: "Lecture 2 Study Notes — CNNs"
collection: class
permalink: /class/lecture2-notes/
---

> NYU Efficient DNN & ML Systems | Midterm Coverage: P1–P69, P75–P78

---

## 1. Conv2D Operation

### How the Computation is Performed

A Conv2D slides a set of **M filters** (each of size K×K×C) over a (B,C,H,W) input, computing a dot product at each spatial location.

For each output pixel (b, m, e, f):
```
Y[b,m,e,f] = Σ_{c,i,j} X[b,c, e+i, f+j] × W[m,c,i,j]
```

Where the sum is over all input channels c and kernel positions (i,j).

### Dimension Notation

| Symbol | Meaning |
|---|---|
| B | Batch size |
| C | Number of input channels |
| H, W | Height, Width of input feature map |
| M | Number of output filters (output channels) |
| K | Kernel / weight size (K×K) |
| E, F | Height, Width of output feature map |

With padding=K//2 and stride=1: E=H, F=W.

**Output shape**: (B, M, E, F)

### Computational Cost

**MACs (Multiply-Accumulate Operations):**
```
MACs = B × M × K × K × C × E × F
```

**Storage cost (in bits, assuming 32-bit floats):**
```
Storage = 32 × (M×C×K×K + B×C×H×W + B×M×E×F)
           weights          input         output
```

---

## 2. BatchNorm — Batch Normalization

### What It Does
Normalizes each channel's activations **across the batch and spatial dimensions** (HW × B for each channel c), then applies a learnable scale and shift.

### Formula

For each channel c:
```
Y_c = α_c × (X_c - μ_c) / σ_c + β_c
```

Where:
- X_c has shape: (HW × B) — all spatial positions and batch samples for channel c
- μ_c, σ_c: mean and std computed over X_c (scalar per channel)
- α_c, β_c: **learnable** scale and shift (scalars per channel)
- Overall: α, β, μ, σ each have length C (one value per channel)

During inference: μ and σ are **fixed** — they are running statistics tracked during training, not recomputed from the batch.

### Parameter Folding During Inference

BatchNorm can be **fused into the preceding Conv2D layer** at inference time, eliminating the BN computation entirely:

Before folding: `Y = BN(Conv(X))` — two operations
After folding: `Y = Conv_folded(X)` — one operation

**How**:
```
Conv output:  Z = X * W + b
BN:           Y = α(Z - μ)/σ + β

Combine:      Y = X * (αW/σ) + (α(b-μ)/σ + β)
                       W'              b'
```

New fused weight: `W' = α·W/σ`
New fused bias: `b' = α(b-μ)/σ + β`

**Result**: No extra memory or compute for BN at inference. Only valid at inference (not training) since μ, σ are fixed.

**Layer Normalization**: normalizes over the feature/channel dimension per sample (not over batch). Cannot be folded in the same way — different normalization axis.

---

## 3. ResNet, MobileNet, ShuffleNet, SqueezeNet, DenseNet

### ResNet — Residual Networks
Key innovation: **skip connections** (identity shortcuts)
```
Y = F(X, {W}) + X
```
Allows gradients to flow directly through the skip path → enables very deep networks (50, 101, 152 layers) without vanishing gradients.

### MobileNet — Depthwise Separable Convolution

Standard Conv2D factored into two stages:

**Stage 1 — Depthwise Conv**: apply one K×K filter per input channel independently
```
Input: (B, C, H, W)
Depthwise filters: C filters, each (1, K, K)  → (B, C, E, F)
```

**Stage 2 — Pointwise Conv**: 1×1 conv to mix channels
```
Pointwise filters: M filters, each (C, 1, 1)  → (B, M, E, F)
```

**MACs:**
```
Depthwise + Pointwise = K×K×C×E×F + M×C×E×F
```

**Storage:**
```
32 × (C×H×W + C×K×K + C×E×F + M×C + M×E×F)
```

**vs. Standard Conv2D MACs**: M×K×K×C×E×F

**Reduction ratio**: `(K²×C×E×F + M×C×E×F) / (M×K²×C×E×F) ≈ 1/M + 1/K²`

For K=3, M=256: ≈ 8-9× fewer MACs.

### Groupwise Convolution

Divide the C input channels into G groups; each group is convolved independently with M/G filters.

```
Standard MAC: E×F×K×K×C×M
Group Conv MAC: E×F×K×K×C×M / G    (G times fewer MACs)
```

Each group of feature maps only sees C/G input channels, reducing both compute and memory.

Used in: **ShuffleNet** (group conv + channel shuffle to re-mix group features), **AlexNet** (original split across 2 GPUs).

### SqueezeNet
Uses **Fire modules**: squeeze (1×1 conv, reduces channels) → expand (mix of 1×1 and 3×3 convs). Achieves AlexNet-level accuracy with 50× fewer parameters.

### DenseNet — Densely Connected Networks
Each layer receives feature maps from **all preceding layers** (concatenation, not addition):
```
x^l = H_l([x⁰, x¹, ..., x^(l-1)])
```
Encourages feature reuse, strong gradient flow, fewer parameters per layer.

---

## 4. Other Tasks: Segmentation

**Semantic Segmentation**: assign a class label to every pixel.

**FCN (Fully Convolutional Network)**: replace FC layers with Conv layers; use transposed convolutions (deconv) to upsample feature maps back to input resolution.

**U-Net**: encoder (downsampling with pooling) + decoder (upsampling) with **skip connections** between corresponding encoder/decoder levels — preserves fine spatial detail.

Output shape: (B, num_classes, H, W) — one probability map per class per pixel.

---

## Quick Reference

| Operation | MACs | Notes |
|---|---|---|
| Standard Conv2D | B×M×K×K×C×E×F | Full cross-channel |
| Depthwise Sep Conv | K²×C×E×F + M×C×E×F | ~8-9× cheaper (K=3,M=256) |
| Group Conv (G groups) | M×K²×C×E×F / G | G× cheaper |
| BN folding | Free at inference | Fuse α,β,μ,σ into W,b |

## Common Exam Traps

- Conv2D output spatial size with padding=same, stride=1: **E=H, F=W**
- BN parameter folding is only valid at **inference** (fixed μ, σ)
- BN normalizes **per channel across the batch**; LayerNorm normalizes **per sample across channels**
- Depthwise conv does NOT mix channels — that's the pointwise step
- Group conv MAC formula: divide standard MAC by **G** (number of groups)
- Storage formula for standard Conv: weights are M×C×K×K (not M×C×H×W)
