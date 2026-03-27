---
title: "Lecture 6 Study Notes — Distillation, NAS, Low-Rank Decomposition, Dynamic Computing"
collection: class
permalink: /class/lecture6-notes/
---

> NYU Efficient DNN & ML Systems | Midterm Coverage: P1–P32, P50–P60, P63–P68

---

## 1. Knowledge Distillation

**Core idea**: Transfer knowledge from a large, accurate **teacher** model to a small, efficient **student** model that can be deployed in resource-constrained environments.

The student is trained not just on ground-truth labels (hard labels) but also on the teacher's output distribution (soft labels), which carries richer information about class relationships.

---

### Response-Based (Output) Knowledge Distillation

The student learns to match the teacher's **output probability distribution** (soft labels).

**Loss:**
```
L_total = c · L_CE(student_logits, hard_labels)
        + (1-c) · L_distill(student_softmax, teacher_softmax)
```

Where:
- `L_CE` = cross-entropy against ground truth (hard labels: one-hot)
- `L_distill` = KL divergence between teacher and student soft distributions
- `c` ≈ 0.9 — hard labels weight close to 1 (ground truth still important)
- Only **student weights** are updated during training; teacher is frozen

**Temperature scaling** (Hinton et al.): divide logits by temperature T before softmax to produce softer distributions:
```
soft_label_i = exp(z_i / T) / Σ_j exp(z_j / T)
```
Higher T → softer probabilities → more information transferred from teacher.

**KL Divergence** (distillation loss):
```
D_KL(P || Q) = Σ_x P(x) · log(P(x)/Q(x))
```
- P = teacher distribution, Q = student distribution
- D_KL = 0 if P = Q (student perfectly matches teacher)
- D_KL > 0 otherwise
- **Not symmetric**: D_KL(P||Q) ≠ D_KL(Q||P)

---

### Feature-Based Knowledge Distillation

Instead of (or in addition to) matching outputs, the student is trained to match the teacher's **intermediate feature maps** (activations from hidden layers).

```
L_feature = ||F_teacher(x) - transform(F_student(x))||²
```

A small "regressor" (1×1 conv or linear layer) adapts the student feature dimensions to match the teacher's, since the networks may differ in size.

**Why**: Intermediate features encode richer representations than just the final logits. The student learns *how* the teacher processes inputs, not just *what* it predicts.

---

### Online Distillation

Both teacher and student are trained simultaneously (no pre-trained teacher required):
- Multiple student models learn from each other's predictions
- Each model serves as a "peer teacher" for the others
- Cheaper setup — no separate teacher training phase

---

### Self-Distillation

A single model distills knowledge to **itself** across its own layers:
- Shallower layers of the network are trained using supervision from deeper layers
- The final layer acts as the teacher; earlier exit points act as students
- Improves intermediate representations and supports **early exiting**

---

### Multi-Teacher, Multi-Student, Cross-Modal Distillation

- **Multi-teacher**: Student learns from multiple specialized teachers (ensemble); soft labels are averaged or combined
- **Multi-student**: Multiple students share a teacher; can specialize for different hardware targets
- **Cross-modal**: Teacher and student process different modalities (e.g. teacher = image+text, student = image only); transfers multimodal knowledge to unimodal model

---

## 2. Once-for-All NAS (Neural Architecture Search)

**The problem with NAS**: Searching for optimal architectures typically requires training thousands of candidate networks — prohibitively expensive.

### Once-for-All (OFA) Approach

Train a single **supernet** that contains all candidate subnetworks simultaneously:

```
Supernet: large network with max depth, width, kernel size
Subnetworks: any valid subnet (smaller depth, fewer channels, etc.)
```

All subnetworks share weights with the supernet and are **trained jointly** through progressive shrinking:
1. Train the full supernet
2. Progressively subsample subnets of decreasing size during training
3. At deployment: extract a subnetwork matching hardware constraints → **no retraining needed**

**Benefits**:
- Train once → deploy anywhere (cloud, mobile, IoT/MCU)
- Eliminates repeated training per hardware target
- Decouples training cost from the number of deployment scenarios

---

## 3. Low-Rank Decomposition (SVD)

**Motivation**: Weight matrices in neural networks are often redundant — their effective rank is much lower than their full dimensions. Decomposing them into lower-rank factors reduces parameters and MACs.

### Singular Value Decomposition (SVD)

Any m×n matrix W can be decomposed:
```
W = U × R × Vᵀ = W₁ × W₂
```

Where:
- W: (m×n) — original weight matrix
- U: (m×r) — left singular vectors (orthonormal columns, eigenvectors of WWᵀ)
- R: (r×r) — diagonal matrix of singular values (σ₁ ≥ σ₂ ≥ ... ≥ σ_r)
- Vᵀ: (r×n) — right singular vectors (orthonormal rows, eigenvectors of WᵀW)
- r = rank of W

**Decomposed as two factors**:
```
W₁ = U·√R    (m×r)
W₂ = √R·Vᵀ  (r×n)
```

**Parameter count:**
- Before: m×n
- After: m×r + r×n = r(m+n)
- Savings when r < mn/(m+n) ≈ min(m,n)

**⚠ Important**: Without rank truncation (using full r = min(m,n)), the decomposition actually **increases** the number of parameters (r(m+n) > mn for full rank).

**Rank truncation**: Keep only top-k singular values (k < r), discarding the rest:
```
W ≈ W_k = U_k × R_k × Vᵀ_k
```
This introduces approximation error but achieves genuine compression (k(m+n) < mn when k is small).

### SVD Applied to LLM Attention (Lecture 7 preview)

SVD decomposition of W_q, W_k, W_v in attention layers:
- Replaces each (E×E) matrix with two smaller matrices (E×r) and (r×E)
- Saves MACs: from E² to 2Er per matrix
- Also reduces KV cache size (smaller K, V projections → smaller cached vectors)

---

## 4. Dynamic / Conditional Computing

**Motivation**: Not all inputs require the same amount of computation. Simple inputs can be processed with less compute; hard inputs need more.

### Early Exit
Multiple classification heads ("exits") at different depths:
```
Input → Block 1 → Exit 1 → (if confident: output)
                → Block 2 → Exit 2 → (if confident: output)
                ...
                → Block N → Final exit
```
Easy inputs exit early → save compute. Hard inputs go deeper.

**Confidence criterion**: if `max(softmax) > threshold` → exit here.

### Conditional Computation / Mixture of Experts (MoE)
Only activate a subset of parameters per input:
- A **router/gating network** selects which expert sub-networks process each token
- Only the selected experts are computed (sparse activation)
- Scales model capacity without proportional compute increase

Used in: **Switch Transformer**, **Mixtral**, **GPT-4** (rumored)

---

## Quick Reference

| Technique | Key Idea | When to Use |
|---|---|---|
| Response-based KD | Match output distributions | Standard compression |
| Feature-based KD | Match intermediate features | More accuracy, more complex |
| Online KD | Peers teach each other | No pre-trained teacher available |
| Self-distillation | Network teaches itself | Supports early exit |
| OFA-NAS | One supernet → many subnets | Multi-hardware deployment |
| SVD | Factorize W = W₁W₂ | FC/attention layers, low-rank assumption |
| Early exit | Output at intermediate layers | Inference-time dynamic compute |

## Common Exam Traps

- In response-based KD, **only the student** is updated; teacher is frozen
- KL divergence is **not symmetric**: D_KL(P||Q) ≠ D_KL(Q||P)
- SVD without rank truncation: parameters **increase** (r(m+n) vs. mn)
- With rank truncation r < min(m,n): parameters decrease only if `r < mn/(m+n)`
- OFA trains **one** supernet; subnetworks are **extracted at inference**, not retrained
- c in distillation loss (c ≈ 0.9): close to 1, meaning **hard labels dominate**
- Temperature T in distillation: higher T → **softer** probability distribution → more signal
