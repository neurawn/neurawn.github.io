---
title: "Lecture 8 Study Notes — Efficient Training"
collection: class
permalink: /class/lecture8-notes/
---

> NYU Efficient DNN & ML Systems | Midterm Coverage: P1–P50

---

## 1. Efficient Training of DNNs

Training large DNNs is expensive in both **compute** and **memory**. Efficient training techniques reduce one or both.

---

### Efficient Computing

#### Mixed-Precision Training
Use FP16 (or BF16) for most operations, but keep a **FP32 master copy** of weights for updates:

```
Forward pass:  FP16 weights → FP16 activations (faster, less memory)
Backward pass: FP16 gradients
Weight update: convert to FP32 → update → store FP32 master copy
                                         → copy back to FP16 for next forward
```

**Why keep FP32 master copy?**
- Weight updates are tiny (lr × gradient) — FP16 lacks precision to represent small increments without rounding to zero
- Accumulated gradients in FP32 prevent precision loss

**Loss scaling**: FP16 has limited dynamic range; small gradient values underflow to 0. Multiply loss by a large scalar S before backward → gradients are S× larger → no underflow. Divide by S before weight update.

#### Gradient Accumulation
When batch size is limited by GPU memory, simulate a larger batch by accumulating gradients over multiple micro-batches:
```
for i in range(N):          # N micro-batches
    loss = model(micro_batch[i]) / N
    loss.backward()         # accumulate gradients
optimizer.step()            # single update after N micro-batches
optimizer.zero_grad()
```
Effective batch size = N × micro_batch_size, without N× memory.

#### Tensor Parallelism
Split individual weight matrices across multiple GPUs:
```
W: (E × 4E)  →  split column-wise across 4 GPUs:
  GPU 0: W[:,   0:E]
  GPU 1: W[:, E:2E]
  GPU 2: W[:, 2E:3E]
  GPU 3: W[:, 3E:4E]
```
Each GPU computes a partial result; an AllReduce combines them.

---

### Efficient Storage

#### Gradient Checkpointing (Activation Recomputation)
Normal training stores all intermediate activations for the backward pass. With gradient checkpointing:
- During forward pass: **discard** most intermediate activations (only keep checkpoints at layer boundaries)
- During backward pass: **recompute** the discarded activations on-the-fly as needed

**Memory vs. Compute trade-off**:
- Memory: O(√n) instead of O(n) for n layers
- Compute: ~33% extra FLOPs (one extra forward pass per checkpoint segment)

#### ZeRO (Zero Redundancy Optimizer)
Across data-parallel training with N GPUs, ZeRO eliminates redundancy in storing:
- **ZeRO-1**: Partition optimizer states across GPUs (gradient momentum, variance)
- **ZeRO-2**: Also partition gradients
- **ZeRO-3**: Also partition model parameters

Each GPU holds 1/N of each quantity → memory scales as O(total/N) instead of O(total).

#### Quantized Training
Train with lower precision (INT8 or FP8) to reduce memory and increase throughput.
Challenge: gradients are noisy at low precision → use STE or stochastic rounding.

---

## 2. Parameter-Efficient Fine-Tuning (PEFT)

Fine-tuning all parameters of a large pretrained model is expensive. PEFT methods add or modify a **small number of parameters** while keeping the pretrained weights frozen.

### LoRA (Low-Rank Adaptation)

Freeze the original weight matrix W; add a low-rank decomposition as a trainable delta:
```
W_new = W + ΔW = W + A·B

W: (E × E) — frozen pretrained weight
A: (E × r) — trainable, initialized with random Gaussian
B: (r × E) — trainable, initialized to zero (so ΔW=0 at start)
r << E      — rank, typically r = 4, 8, 16
```

**Training**: only A and B are updated. Gradient flows through A·B.

**Inference merge**: `W_new = W + A·B` — compute once, deploy as a single weight matrix. **No inference overhead**.

**Parameter count**: 2Er vs. E² (original) → reduction factor E/(2r)
- E=4096, r=8: 4096²=16.7M → 2×4096×8=65K parameters (256× reduction)

LoRA is applied to Q, K, V (and sometimes FFN) projection matrices in each transformer layer.

### Adapter Layers
Insert small bottleneck modules inside each transformer layer:
```
x → [original layer] → Adapter(x) + x
                          ↓
                    Linear(d→r) → NonLinear → Linear(r→d)
```
Only the adapter parameters (small: d×r + r×d) are trained. Original weights frozen.
Slight inference overhead (extra computation per adapter).

### Prefix Tuning / Prompt Tuning
Prepend learnable "virtual tokens" (soft prompts) to the input sequence:
```
[prefix tokens (learnable)] + [actual input tokens]  →  transformer
```
Only the prefix embeddings are trained. The transformer itself is frozen.
No architectural changes; works with any frozen model.

### Comparison

| Method | Where added | Inference overhead | # trainable params |
|---|---|---|---|
| LoRA | Parallel to W_q,k,v | None (can merge) | ~0.1% of base model |
| Adapter | Sequential inside layer | Yes (extra layers) | ~0.5-3% |
| Prefix tuning | Input sequence | Longer context | ~0.01% |
| Full fine-tune | All parameters | None | 100% |

---

## 3. Basics of Speculative Decoding

### The Problem with Autoregressive Decoding
LLM decoding generates **one token at a time** — each step depends on the previous output. This is inherently sequential and GPU-underutilized (small batch of 1 token per step).

### Speculative Decoding

Use a small **draft model** to quickly generate K candidate tokens, then verify them in parallel with the large **target model**:

**Algorithm:**
1. **Draft**: small fast model generates K tokens speculatively:
   ```
   q₁, q₂, ..., q_K = DraftModel(x₁...x_t)  (fast, sequential)
   ```
2. **Verify**: large target model processes all K+1 positions in **one parallel forward pass**:
   ```
   p₁, p₂, ..., p_{K+1} = TargetModel(x₁...x_t, q₁...q_K)
   ```
3. **Accept/Reject**: compare target model's probabilities with draft tokens:
   - If target agrees with draft token q_i → **accept**, move to q_{i+1}
   - If target disagrees → **reject** q_i, sample from target's corrected distribution, discard q_{i+1...K}
4. Always produce at least 1 correct token per round (the rejected position or final position).

**Acceptance condition**:
```
Accept q_i  if  Uniform(0,1) ≤ p_target(q_i) / p_draft(q_i)
```
This is rejection sampling — produces exact samples from the target distribution.

**Why it's faster**:
- If the draft model is accurate (agrees with target often), most tokens are accepted → K tokens per verify step vs. 1 token per step normally
- The large model runs once per K tokens (parallel) instead of K times sequentially
- GPU utilization improves — verifying K tokens fills the GPU better than 1 token

**Requirements**:
- Draft model must be much **faster** than target model (typically 5-10× smaller)
- Draft model should be **accurate** — high acceptance rate
- No change in output quality — generates exact samples from target distribution

**Example**: Target = LLaMA-70B, Draft = LLaMA-7B. Draft generates 4 tokens; target verifies in one pass. If 3 accepted → 3× throughput improvement vs. pure autoregressive.

---

## Quick Reference

| Technique | Memory Impact | Compute Impact |
|---|---|---|
| Mixed-precision (FP16) | -50% activations | +throughput (tensor cores) |
| Gradient checkpointing | O(√n) → much lower | +33% FLOPs |
| ZeRO-3 | /N across N GPUs | +communication |
| LoRA | +2Er per layer (tiny) | None at inference (merge) |
| Speculative decoding | Same | +throughput (K tokens/verify) |

## Common Exam Traps

- Mixed-precision: keep **FP32 master copy** for weight updates — FP16 lacks precision for tiny updates
- Loss scaling: multiply **loss** (not gradients directly) by S before backward; divide by S before update
- Gradient checkpointing trade-off: **less memory, more compute** (~33% extra FLOPs)
- LoRA: B is initialized to **zero** (so ΔW=0 at initialization) — full model starts unchanged
- LoRA at inference: **merge** A·B into W — zero overhead; no extra computation
- Speculative decoding output is **identical** to pure autoregressive (exact same distribution)
- Speculative decoding: at least **1 token** is accepted per round (the re-sampled one at rejection point)
- ZeRO reduces **redundancy** across data-parallel replicas — different from model parallelism
