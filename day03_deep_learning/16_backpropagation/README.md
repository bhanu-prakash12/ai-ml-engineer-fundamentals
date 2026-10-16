# 16 — Backpropagation

> Backpropagation is just the chain rule, applied layer by layer and cached — derive it by hand once and autograd stops being magic.

| | |
|---|---|
| **Day** | 3 — Deep Learning + PyTorch |
| **Topic** | Backpropagation by hand, gradient checking |
| **Notebooks** | 1 (`01_backprop_by_hand.ipynb`) |
| **Builds on** | [`15_deep_learning`](../15_deep_learning/README.md), Day 1 `05_ml_mathematics` |
| **Leads to** | [`17_deep_learning_architectures`](../17_deep_learning_architectures/README.md), [`19_pytorch_autograd`](../19_pytorch_autograd/README.md) |

## 📖 Short forms used in this topic

| Term | Full form | What it is |
|---|---|---|
| MLP | Multi-Layer Perceptron | Fully-connected network with non-linear activations |
| ReLU | Rectified Linear Unit | Activation `max(0, x)`; its derivative is `1` for `x > 0`, else `0` |
| SGD | Stochastic Gradient Descent | Update `w ← w - lr * grad` using mini-batches |
| `∂L/∂W` | Partial derivative of loss L w.r.t. W | How much the loss changes per small change in W |
| `dW`, `db`, `dZ`, `dA` | Gradient of the loss w.r.t. W, b, Z, A | Shorthand for the corresponding `∂L/∂...` |
| `Z`, `A` | Pre-activation, activation | `Z = XW + b`, `A = f(Z)` |
| NumPy | Numerical Python | Array library used for the from-scratch code |

## 🧠 In plain words

Training needs to know, for every weight, "if I nudge you up a bit, does the loss go up or down, and by how much?" Backpropagation answers that for all weights at once by walking from the loss back toward the input, reusing the result from the layer just ahead at each step. The optimizer then uses those answers to change the weights.

## 🎒 Analogy to remember

A **restaurant gets a bad review** (the loss). The owner traces the blame backward: the waiter, then the kitchen, then the supplier, each time reusing what was already learned about the later stage ("the dish was cold, so the kitchen share is X"). Nobody re-investigates from scratch for each person. *Where it breaks down:* the math assigns exact numeric credit (derivatives), not blame, and every "person" is a differentiable function.

## Learning objectives

By the end of this topic you can:
- Trace the full training step: forward pass → prediction → loss → gradients → backward pass → optimizer → updated weights.
- Compute every gradient for a tiny 2-layer network by hand, matching a numerical gradient check.
- Explain why backpropagation is "just" the chain rule applied layer by layer, cached efficiently.

## Why it matters

This is the single most important mechanism in all of deep learning, and the part most
engineers treat as a black box. Doing it by hand once, on a tiny example, removes the mystery
before PyTorch's `autograd` (`19_pytorch_autograd`) does it automatically — you'll be verifying
autograd's output against your own hand-derived gradients, not trusting it blindly.

## Prerequisites

[`15_deep_learning`](../15_deep_learning/README.md) (the forward pass, losses) and Day 1's
[`05_ml_mathematics`](../../day01_python_data_math/05_ml_mathematics/README.md) (the chain rule).

## Core concepts

```text
forward pass
→ prediction
→ loss
→ gradients (backward pass)
→ optimizer
→ updated weights
```

- **Forward pass**: compute every layer's output, *keeping* the intermediate values (`Z1`, `A1`, `Z2`, ...) — they're needed again during the backward pass.
- **Loss**: a single scalar, as in `15_deep_learning`.
- **Backward pass**: starting from the loss, apply the chain rule layer by layer, *backward* through the network, computing `∂Loss/∂W` for every weight matrix and `∂Loss/∂b` for every bias.
- **The key reuse**: each layer's backward computation reuses the gradient computed one layer downstream (`∂Loss/∂A_next`) — this reuse, not any new math, is why backprop is efficient compared to recomputing each weight's gradient independently from scratch.
- **Optimizer step**: once every gradient is known, any optimizer from Day 2's `10_ml_optimization` (SGD, Adam, ...) can use them to update weights — backprop only computes the gradients, the optimizer decides how to use them.

```mermaid
flowchart LR
  X["Input X"] --> Z1["Z1 = XW1 + b1"] --> A1["A1 = relu of Z1"] --> Z2["Z2 = A1W2 + b2"] --> Yhat["yhat = softmax of Z2"]
  Yhat --> L["Loss"]
  L -->|"dL/dZ2"| dZ2["dZ2"]
  dZ2 -->|"dL/dW2, dL/db2"| W2grad["dW2, db2"]
  dZ2 -->|"dL/dA1 via W2 transposed"| dA1["dA1"]
  dA1 -->|"dL/dZ1 via ReLU derivative"| dZ1["dZ1"]
  dZ1 -->|"dL/dW1, dL/db1"| W1grad["dW1, db1"]
  classDef fwd fill:#DBEAFE,stroke:#2563EB,color:#1a1a1a;
  classDef loss fill:#FEF3C7,stroke:#D97706,color:#1a1a1a;
  classDef bwd fill:#FEE2E2,stroke:#DC2626,color:#1a1a1a;
  classDef grad fill:#DCFCE7,stroke:#16A34A,color:#1a1a1a;
  class X,Z1,A1,Z2,Yhat fwd;
  class L loss;
  class dZ2,dA1,dZ1 bwd;
  class W2grad,W1grad grad;
```

## 🔍 Concepts explained

> [!NOTE]
> Chips: 🟦 forward values · 🟥 backward gradients · 🟩 weight updates · 🟨 verification.

### 🟦 Cached forward pass
- **Definition**: compute the forward pass but keep `X, Z1, A1, Z2, ŷ`.
- **Analogy**: keeping receipts so you can audit later.
- **Code idea**: store them in a `cache` dict returned by `forward()`.
- **Use when / watch out**: the backward pass needs these (e.g. `A1` for `dW2`, `Z1` for ReLU'); recomputing wastes work.

### 🟥 Backward pass (chain rule)
- **Definition**: starting at `∂L/∂ŷ`, multiply local derivatives layer by layer toward the input.
- **Analogy**: passing blame back down the chain of command.
- **Code idea** (softmax + cross-entropy, batch size `N`; MSE would differ):
  ```text
  dZ2 = (ŷ - y_onehot) / N
  dW2 = A1ᵀ @ dZ2        db2 = sum(dZ2, axis=0)
  dA1 = dZ2 @ W2ᵀ
  dZ1 = dA1 * (Z1 > 0)   # ReLU'
  dW1 = Xᵀ @ dZ1         db1 = sum(dZ1, axis=0)
  ```
- **Use when / watch out**: the transposes are exactly where shape-valid-but-wrong bugs hide.

### 🟨 Numerical gradient check
- **Definition**: estimate each gradient by finite differences and compare with the analytic one.
- **Analogy**: double-checking a calculator result by estimating on paper.
- **Code idea**: `(L(w + ε) - L(w - ε)) / (2ε)` per parameter; compare via relative error.
- **Use when / watch out**: use for debugging only (it costs two forward passes per parameter); use float64 and a small `ε` around `1e-5`.

### 🟩 Optimizer step
- **Definition**: uses gradients to update weights (`w ← w - lr * dW` for SGD).
- **Analogy**: the map (gradients) vs. the driver (optimizer).
- **Use when / watch out**: backprop computes gradients only; the choice of SGD/Adam is separate.

### Differences at a glance

| | Aspect | Backprop (analytic) | Numerical gradient |
|---|---|---|---|
| 🟥 | Cost | One backward pass for all weights | Two forward passes per weight |
| 🟥 | Accuracy | Exact (to float precision) | Approximate (truncation and rounding error) |
| 🟨 | Role | Used in training | Used to verify the analytic code |

> [!WARNING]
> A transpose mistake can still produce a valid shape (e.g. square matrices) yet wrong gradients. Only a numerical check reliably catches it.

> [!TIP]
> A gradient disagreement of about a factor of 2 often points to a missing or extra `2` in MSE, or a missing `1/N` batch average.

### 🗺️ Concept map

```mermaid
flowchart LR
  subgraph FWD["Forward and cache"]
    F1["Forward pass"] --> F2["Cache Z1, A1, Z2"]
  end
  subgraph BWD["Backward"]
    B1["Chain rule per layer"] --> B2["Reuse downstream gradient"]
  end
  subgraph CHK["Verify"]
    C1["Finite differences"] --> C2["Compare with analytic"]
  end
  F2 --> B1
  B2 --> O["Optimizer step"]
  B2 --> C1
  classDef fwd fill:#DBEAFE,stroke:#2563EB,color:#1a1a1a;
  classDef bwd fill:#FEE2E2,stroke:#DC2626,color:#1a1a1a;
  classDef chk fill:#FEF3C7,stroke:#D97706,color:#1a1a1a;
  classDef upd fill:#DCFCE7,stroke:#16A34A,color:#1a1a1a;
  class F1,F2 fwd;
  class B1,B2 bwd;
  class C1,C2 chk;
  class O upd;
```

## Important APIs / functions

Pure NumPy manual backward-pass implementation; `numpy`-based numerical gradient checking
(finite differences, reused from Day 1's `05_ml_mathematics`) to verify every analytic gradient.

## Notebooks

| Notebook | Purpose | Expected output |
|---|---|---|
| `01_backprop_by_hand.ipynb` | A tiny manual numerical example first (2-3 scalar values, full chain-rule derivation by hand), then backpropagation implemented from scratch for the 2-layer MLP from `15_deep_learning`, with every analytic gradient checked against a numerical (finite-difference) gradient | Every analytic gradient (`dW1`, `db1`, `dW2`, `db2`) matching its numerical counterpart within tolerance; one full training step shown to reduce the loss |

## Key takeaways

- Backpropagation is the chain rule, applied once per layer, with intermediate results cached and reused — it is not a separate algorithm from calculus, it's calculus organized efficiently.
- Every gradient computed by hand here is exactly what PyTorch's `autograd` computes automatically in `19_pytorch_autograd` — the same math, just tracked by a computational graph instead of hand-derived.
- Numerical gradient checking (finite differences) is the standard way to verify a hand-written or newly-implemented backward pass is correct.

## Common mistakes

- Forgetting to cache forward-pass intermediate values, forcing a wasteful full forward recomputation during the backward pass.
- Getting a transpose wrong in a backward-pass matrix multiplication (a very common bug — shapes will often still "work" by accident in one direction but be mathematically wrong).
- Not checking gradients numerically after implementing a backward pass by hand, shipping a silently-wrong gradient.

## Interview questions

**Conceptual**
1. Why is backpropagation described as "the chain rule, applied efficiently" rather than a separate algorithm?
2. What intermediate values from the forward pass does the backward pass need, and why?
3. Why does a numerical gradient check matter even when you're confident in your analytic derivation?
4. What's the relationship between backpropagation and the optimizer (e.g. SGD, Adam) — which one computes what?
5. Why would a transposed-matrix bug in a backward pass often not raise a shape error, yet still produce wrong results?

**Coding**
1. Implement the backward pass for a single linear layer (`Z = XW + b`) and verify `dW`, `db`, and `dX` against numerical gradients.
2. Implement ReLU's backward pass (`dZ = dA * (Z > 0)`) and verify it against a numerical gradient.
3. Chain the full 2-layer MLP's forward and backward pass together and run 50 steps of gradient descent, verifying the loss decreases.

**Debugging**
1. Your backward pass's `dW1` has the wrong shape — what's the most common cause (hint: check which matrix needs transposing)?
2. Your numerical gradient check consistently disagrees with your analytic gradient by roughly a factor of 2 — what's a common cause of exactly this kind of discrepancy?

**Real-world engineering**
1. You're debugging a custom loss function in a framework that doesn't support autograd for your use case — how would you verify your hand-written gradient is correct before trusting it in training?
2. Training loss decreases initially then suddenly becomes `nan` — how might a backward-pass bug (vs a learning-rate issue) cause this, and how would you distinguish the two?

## Exercises

1. By hand, derive `∂Loss/∂w` for a single-weight, single-input linear regression with MSE loss, and verify it matches the formula used in Day 2's `07_regression` gradient descent.
2. Implement numerical gradient checking as a reusable function and apply it to every weight matrix in the 2-layer MLP backward pass.
3. Deliberately introduce a transpose bug in the backward pass and observe how the numerical gradient check catches it.

## References

- [Stanford CS231n — Backpropagation notes](https://cs231n.github.io/optimization-2/)
- [3Blue1Brown — Backpropagation calculus](https://www.3blue1brown.com/lessons/backpropagation-calculus)
- [Rumelhart, Hinton, Williams (1986) — Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0)
