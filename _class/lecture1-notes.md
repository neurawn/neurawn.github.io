---
title: "Lecture 1 Study Notes — Neural Network Basics"
collection: class
permalink: /class/lecture1-notes/
---

> NYU Efficient DNN & ML Systems | Midterm Coverage: P65–P98

---

## 1. MLP — Multi-Layer Perceptron

A fully-connected feedforward network. Each layer applies a linear transformation followed by a nonlinearity:

```
z = Wx + b        (linear)
a = σ(z)          (activation: ReLU, GeLU, etc.)
```

**Layer shapes (batch B, input n, output m):**
```
x: (B, n)
W: (n, m)
b: (m,)
z = xW + b: (B, m)
```

**MACs per FC layer**: B × n × m

**Typical architecture**: Input → [Linear → ReLU] × L → Linear → Softmax → output

---

## 2. Forward Propagation

Pass input through the network layer by layer, computing activations at each layer:

```
a⁰ = x                                   (input)
z^l = W^l · a^(l-1) + b^l               (pre-activation)
a^l = σ(z^l)                             (post-activation)
ŷ   = a^L                                (output)
```

**Loss**: Cross-entropy for classification:
```
L = -Σ y_i · log(ŷ_i)
```

---

## 3. Backward Propagation

Compute gradients of the loss w.r.t. all parameters using the **chain rule**, flowing gradients from output back to input.

**Key steps:**
```
δ^L = ∂L/∂z^L                           (output layer gradient)
δ^l = (W^(l+1))ᵀ · δ^(l+1) ⊙ σ'(z^l)  (hidden layer gradient)
∂L/∂W^l = (a^(l-1))ᵀ · δ^l            (weight gradient)
∂L/∂b^l = Σ δ^l                         (bias gradient)
```

**Why it works**: Each gradient is the product of all upstream gradients — this is automatic differentiation through dynamic programming.

**Vanishing gradient**: If σ'(z) < 1 at many layers, the gradient signal shrinks exponentially deep into the network → use ReLU, residual connections, normalization.

---

## 4. Regularization: Weight Decay & Dropout

### Weight Decay (L2 Regularization)
Add a penalty on large weights to the loss:
```
L_total = L_task + λ/2 · ||W||²
```
Gradient update becomes: `W ← W - η(∂L/∂W + λW)`

Effect: shrinks weights toward zero, discourages overfitting, equivalent to a Gaussian prior on weights.

### Dropout
During training: randomly zero out each activation with probability p (typically p=0.5).
During inference: **scale activations by (1-p)** (or equivalently, scale up by 1/(1-p) during training — inverted dropout).

```python
mask = Bernoulli(1 - p)    # 1 with prob (1-p), 0 with prob p
a_dropped = a * mask / (1 - p)
```

**Why it works**: Forces the network to learn redundant, independent features; prevents co-adaptation. Acts like training an ensemble of exponentially many sub-networks.

---

## 5. Optimizers: SGD, RMSProp, Adam

### SGD (Stochastic Gradient Descent)
```
W ← W - η · ∂L/∂W
```
With **momentum** (standard practice):
```
v ← β·v + ∂L/∂W
W ← W - η·v
```
Simple, but sensitive to learning rate; noisy gradient estimates.

### RMSProp
Adapts per-parameter learning rates using exponential moving average of squared gradients:
```
v ← β·v + (1-β)·(∂L/∂W)²
W ← W - η · (∂L/∂W) / (√v + ε)
```
Prevents the learning rate from decaying too fast in dimensions with frequent large gradients.

### Adam (Adaptive Moment Estimation)
Combines momentum (1st moment) and RMSProp (2nd moment):
```
m ← β₁·m + (1-β₁)·g          (1st moment — mean)
v ← β₂·v + (1-β₂)·g²         (2nd moment — variance)
m̂ = m / (1-β₁ᵗ)              (bias correction)
v̂ = v / (1-β₂ᵗ)
W ← W - η · m̂ / (√v̂ + ε)
```
Default: β₁=0.9, β₂=0.999, ε=1e-8. Most widely used optimizer in deep learning.

| Optimizer | Adapts LR? | Momentum? | Notes |
|---|---|---|---|
| SGD | No | Optional | Simple, requires tuning |
| RMSProp | Yes (2nd moment) | No | Good for non-stationary |
| Adam | Yes (1st + 2nd) | Yes | Default choice |

---

## 6. Multistage Learning Rate Scheduler

The learning rate η changes during training to improve convergence:

**Step Decay**: reduce by factor γ every N epochs
```
η_t = η₀ × γ^⌊t/N⌋
```

**Cosine Annealing**: smooth decay following cosine curve
```
η_t = η_min + ½(η_max - η_min)(1 + cos(πt/T))
```

**Warmup + Decay** (common in transformers): linearly increase η for first W steps, then decay:
```
step ≤ W: η = η_max × (step/W)
step > W: η = η_max × decay_schedule
```
Warmup prevents instability at the start when gradients are large and noisy.

**Multistage**: combine phases — e.g. warmup → cosine decay → fine-tuning at low LR.

---

## 7. Initialization

Bad initialization → vanishing or exploding gradients before training even starts.

### Xavier / Glorot Initialization
For tanh/sigmoid activations:
```
W ~ Uniform(-√(6/(n_in + n_out)), +√(6/(n_in + n_out)))
```
Keeps variance of activations and gradients approximately equal across layers.

### He / Kaiming Initialization
For ReLU activations (accounts for the fact that ReLU zeros half the values):
```
W ~ Normal(0, √(2/n_in))
```

**Why it matters**: At layer l, if `Var(a^l) ≠ Var(a^(l-1))`, gradients either vanish (< 1) or explode (> 1) exponentially. Initialization sets the right scale so variance ≈ 1 at every layer at the start of training.

---

## Quick Reference

| Concept | Key Formula / Rule |
|---|---|
| FC layer MACs | B × n_in × n_out |
| Weight decay update | W ← W - η(g + λW) |
| Dropout at inference | multiply activations by (1-p) |
| Adam bias correction | divide m by (1-β₁ᵗ), v by (1-β₂ᵗ) |
| He init (ReLU) | std = √(2/n_in) |
| Xavier init (tanh) | std = √(2/(n_in+n_out)) |

## Common Exam Traps

- Adam has **bias correction** — without it, early steps would be biased toward zero
- Dropout is **disabled at inference**; activations are scaled instead
- Weight decay adds `λW` to the gradient (not the loss directly in the update rule)
- He init uses factor **2** (for ReLU); Xavier uses **1** (for tanh)
- Momentum in SGD accumulates past gradients; it does **not** adapt the learning rate per-parameter
