---
title: "Lecture 3 Study Notes — Transformers, ViT, LLMs, SSL"
collection: class
permalink: /class/lecture3-notes/
---

> NYU Efficient DNN & ML Systems | Midterm Coverage: P1–P69, P75–P78

---

## 1. Transformers

### How the Computation is Performed — and Why

The transformer replaces recurrence with **attention**: every token can directly attend to every other token in a single pass, enabling massive parallelism and capturing long-range dependencies that RNNs struggle with.

The core idea: for each token, compute a weighted sum over all other tokens' **values**, where the weights come from the compatibility (dot-product) between that token's **query** and every other token's **key**.

---

### Self-Attention Block (Step-by-Step)

Given input `x` of shape **(B, L, E)** — batch × sequence length × embedding dim:

**Step 1 — Create Q, K, V**
```
x: (B, L, E)  ×  W_Q/K/V: (E × E)  →  Q, K, V each (B, L, E)
```
Three separate learned linear projections produce the Query, Key, and Value matrices.

**Step 2 — Compute attention scores**
```
QKᵀ:  (B, L, E) × (B, E, L)  →  (B, L, L)
```
Each entry (i, j) is the raw compatibility score between token i's query and token j's key.

**Step 3 — Scale and normalize**
```
Softmax(QKᵀ / √E)  →  (B, L, L)
```
Dividing by √E prevents the dot products from growing too large (which would saturate softmax and kill gradients).

**Step 4 — Weighted sum over values**
```
Softmax(QKᵀ) × V:  (B, L, L) × (B, L, E)  →  (B, L, E)
```

**Step 5 — Output projection + residual**
```
Linear(attention_output) + x  →  (B, L, E)   [Y]
```

**Block diagram flow:**
```
x → Normalization → [linear_Q | linear_K | linear_V]
                        ↓         ↓         ↓
                        Q    QKᵀ  K         V
                             ↓
                           Scale
                             ↓
                          Softmax
                             ↓
                         (×) ──────────────┘
                             ↓
                           linear
                             ↓
                        (+) residual  →  Y
```

---

### Multi-Headed Attention (MHA)

Instead of one set of Q/K/V projections, run **h** independent attention heads in parallel, each with smaller dimension E/h. Concatenate their outputs and project back to E.

- **Why**: different heads can attend to different representation subspaces / positions simultaneously.
- Each head: `(B, L, E/h)` for Q, K, V
- Concat all heads → `(B, L, E)` → final linear

---

### Feed-Forward Network (FFN)

Applied independently to each token after self-attention:
```
FFN(x) = Linear₂(GeLU(Linear₁(x)))
```
- Linear₁: E → 4E (expansion)
- Linear₂: 4E → E (contraction)
- Acts as a per-token "memory" or lookup

---

### Normalization: LayerNorm vs. RMSNorm

| | LayerNorm | RMSNorm |
|---|---|---|
| Computes | mean + variance across features | only RMS (root-mean-square) |
| Formula | `(x − μ) / σ × γ + β` | `x / RMS(x) × γ` |
| Cost | higher (needs mean) | cheaper (skip mean subtraction) |
| Used in | original Transformer, BERT | LLaMA, modern LLMs |

Both normalize each token's embedding vector independently, stabilizing training.

---

### Activation: GeLU

**Gaussian Error Linear Unit** — smoother than ReLU, used in FFN:
```
GeLU(x) = x · Φ(x)   where Φ is the Gaussian CDF
≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```
Allows small negative values to pass (unlike hard-zero ReLU), improving gradient flow.

---

### Embeddings

**Word Embedding**
- Maps each token (integer ID) to a dense vector of size E
- Learned lookup table: `Vocab × E`
- Converts discrete tokens → continuous representations

**Positional Embedding**
- Transformers have no built-in sequence order (attention is permutation-invariant)
- Add a positional vector to each token embedding to inject position info
- Options:
  - **Sinusoidal** (fixed, original paper): `sin/cos` at different frequencies
  - **Learned** (BERT, GPT): trainable `(L × E)` table
  - **RoPE** (Rotary, LLaMA): rotates Q/K vectors by position angle — relative positions encoded implicitly

---

## 2. Vision Transformer (ViT)

**Key idea**: Convert an image into a sequence of patch tokens, then feed to a standard transformer.

### Image → Token Pipeline (3 parts)

**Part 1 — Patch Embedding**
```
Input:  (B, 3, 224, 224)    [3-channel RGB image]
Conv2D (stride=16, out=768): → (B, 768, 14, 14)   [14×14 = 196 patches]
Reshape:                     → (B, 196, 768)
Concat class token [CLS]:   → (B, 197, 768)   ← class token is nn.Parameter, added via +
```

- The Conv2D with kernel=16×16, stride=16 acts as a **patch extractor** — each output location corresponds to one non-overlapping 16×16 patch
- The class token `[CLS]` is a learnable vector prepended to the sequence; it aggregates global information

**Part 2 — Transformer Blocks**
```
(B, 197, 768) → Block → Block → … → (B, 197, 768)
```
Standard transformer encoder blocks (SA + FFN + LayerNorm + residuals), repeated N times (e.g. 12 for ViT-Base).

**Part 3 — Classification Head**
```
(B, 197, 768)
→ LayerNorm             → (B, 197, 768)
→ take [:, 0]           → (B, 768)       [take only the CLS token]
→ Linear(768 → 1000)    → (B, 1000)      [ImageNet logits]
```

### Why it works
The transformer sees all 196 patches simultaneously and can learn global spatial relationships without the locality bias of CNNs — but it needs more data to compensate.

---

## 3. LLM: Prefilling, Decoding, KV Cache

### Transformer as a Generative AI Tool

Decoder-only architecture (GPT-2, LLaMA):
```
Input tokens
→ Embedding
→ Decoder Block (SA + Normalization + FFN + Normalization) × N
→ Linear & Softmax
→ Next token probabilities
```

Key insight: during generation, **K and V vectors from past tokens must be buffered** for future steps — this is the KV cache.

---

### Prefilling Stage

During **prefilling**, the LLM processes the **entire prompt jointly** in one forward pass:

```
Prompt: "How are you"   [3 tokens]

For each token i, each layer j:
  → compute K_{i,j}  (shape: 1×E)
  → compute V_{i,j}  (shape: 1×E)
  → save both into KV cache
```

- All tokens attend to all previous tokens (causal mask)
- Computationally parallel — equivalent to a standard forward pass
- Output: KV cache populated with K/V for every prompt token × every layer

---

### Decoding Stage

After prefilling, the model **generates one token at a time**:

1. Take the last generated token as the new query token `q_L` (shape: `1 × E`)
2. Retrieve all cached K and V from previous tokens
3. Compute attention:
   ```
   q_L × Kᵀ  →  attention scores over all L previous tokens   (1 × L)
   softmax(scores) × V  →  context vector (1 × E)
   ```
4. Linear projection → next token logit → sample/argmax → new token
5. Append new token's K, V to cache; repeat

Each decoding step is **sequential** (can't parallelize across time) and processes only **1 new token**.

---

### Why KV Cache Saves Computation

**Without KV cache** (naive):
- At decoding step L, you must recompute K and V for all L−1 previous tokens from scratch
- Cost grows quadratically with sequence length: O(L²)

**With KV cache**:
- K and V for previous tokens are stored in memory after prefilling (and after each decode step)
- Only compute K, V for the **new token** at each step
- Retrieve cached K/V for attention → no recomputation
- Cost per step: O(L) instead of O(L²)

**Diagram intuition (decode step L):**
```
Q:  [q_L]           shape (1, E)     ← only 1 new query
Kᵀ: [k_1…k_L]      shape (E, L)     ← retrieved from cache
A = q_L × Kᵀ       shape (1, L)     ← attention weights
V:  [v_1…v_L]      shape (L, E)     ← retrieved from cache
y_L = A × V         shape (1, E)     ← output for new token
```

**Trade-off**: KV cache saves compute at the cost of **memory** — cache size = `2 × L × num_layers × E × sizeof(dtype)` bytes.

---

## 4. SSL — Self-Supervised Learning Basics

SSL trains on **unlabeled data** by constructing a supervisory signal from the data itself.

### Core Idea
Design a **pretext task** where the label is derived from the input — no human annotation needed. Learn representations that generalize to downstream tasks.

### Two Main Families

**Contrastive Learning** (e.g. SimCLR, MoCo)
- Create two augmented views of the same image → they should be similar (positive pair)
- Views from different images → they should be dissimilar (negative pair)
- Loss: pull positives together, push negatives apart in embedding space (e.g. InfoNCE loss)

**Masked Autoencoders / Predictive** (e.g. MAE, BERT, BEiT)
- Mask out a portion of input (image patches or text tokens)
- Train model to **reconstruct** the masked portions
- Forces model to learn semantic structure to predict missing parts

### Why SSL matters for Efficient DNN
- Pretrain a large model on unlabeled data (cheap) → fine-tune on small labeled set (few labels)
- Avoids expensive annotation
- Forms the backbone of modern foundation models (BERT, GPT, DINO, MAE)

### SSL with ViT (DINO / MAE)
- **MAE**: mask ~75% of ViT patches, reconstruct raw pixels of masked patches — very efficient
- **DINO**: student-teacher self-distillation, no labels, the [CLS] token learns strong features

---

## Quick Reference: Key Shapes

| Operation | Input | Output |
|---|---|---|
| Word embedding | token_id (B, L) | (B, L, E) |
| Q/K/V projection | (B, L, E) | (B, L, E) each |
| Attention scores QKᵀ | (B, L, E) × (B, E, L) | (B, L, L) |
| Attention output × V | (B, L, L) × (B, L, E) | (B, L, E) |
| ViT patch embed | (B, 3, 224, 224) | (B, 196, 768) |
| After CLS concat | — | (B, 197, 768) |
| ViT classifier output | (B, 768) [CLS only] | (B, 1000) |
| KV cache per token | — | K: (1×E), V: (1×E) per layer |
| Decode query | — | q_L: (1, E) |

---

## Common Exam Traps

- QKᵀ produces **(B, L, L)** — it's L×L not L×E
- Softmax is applied **per row** of the L×L attention matrix (each query attends to all keys)
- ViT patch count: 224/16 = 14 patches per side → 14×14 = **196** patches; with CLS → **197** tokens
- KV cache stores K and V **separately** for each layer; one entry per token per layer
- During decoding only **1 new K,V** is computed and added; all prior K,V are read from cache
- RMSNorm is cheaper than LayerNorm (no mean subtraction), used in LLaMA-family models
- GeLU (not ReLU) is used in transformer FFNs
