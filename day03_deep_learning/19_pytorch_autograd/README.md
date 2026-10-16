# 19 — PyTorch Autograd

> Autograd records every tensor operation in a graph, then runs the chain rule backward for you.

| | |
|---|---|
| **Day** | 3 — Deep Learning + PyTorch |
| **Topic** | `requires_grad`, `.backward()`, `.grad`, `torch.no_grad()` |
| **Notebooks** | 1 (`01_autograd_basics.ipynb`) |
| **Builds on** | [`18_pytorch_basics`](../18_pytorch_basics/README.md), [`16_backpropagation`](../16_backpropagation/README.md) |
| **Leads to** | [`20_pytorch_models`](../20_pytorch_models/README.md), [`22_pytorch_training`](../22_pytorch_training/README.md) |

## 📖 Short forms used in this topic

| Term | Full form | What it is |
|---|---|---|
| autograd | Automatic differentiation (PyTorch's engine) | Records operations and computes gradients via reverse-mode differentiation |
| MLP | Multi-Layer Perceptron | The 2-layer network rebuilt in PyTorch here |
| `grad` | Gradient | Tensor attribute holding `∂output/∂tensor` after `.backward()` |
| `W`, `b`, `Z`, `A` | Weights, bias, pre-activation, activation | Same names as in `16_backpropagation` |
| CPU / GPU | Central / Graphics Processing Unit | Processors a tensor can live on |
| CUDA | Compute Unified Device Architecture | NVIDIA GPU platform |
| MPS | Metal Performance Shaders | Apple Silicon GPU backend |
| NVIDIA | (company name) | GPU maker behind CUDA |
| NumPy | Numerical Python | Used for the hand-derived gradients being compared |

## 🧠 In plain words

When you compute with tensors that have `requires_grad=True`, PyTorch quietly writes down every step. Calling `.backward()` on the final scalar walks that record in reverse and fills in `.grad` on every input — the same chain-rule math you did by hand, with no manual derivation.

## 🎒 Analogy to remember

Autograd is a **flight data recorder**: during the flight (forward pass) it logs every operation; after the incident (loss), `.backward()` replays the log in reverse to find each component's contribution. *Where it breaks down:* a recorder only reads; autograd's graph is rebuilt on each forward pass and freed after `.backward()` by default.

## Learning objectives

By the end of this topic you can:
- Explain the computational graph PyTorch builds when `requires_grad=True`.
- Use `.backward()` and `.grad` to compute gradients automatically, and verify them against `16_backpropagation`'s hand-derived gradients.
- Explain `torch.no_grad()` and when/why to use it.

## Why it matters

This is the payoff for `16_backpropagation`. Every gradient computed by hand there is computed
automatically here — the exact same math, just tracked by a graph instead of derived on paper.
Trusting (and knowing how to debug) autograd is foundational to every PyTorch training loop
from here on.

## Prerequisites

[`18_pytorch_basics`](../18_pytorch_basics/README.md) (tensors) and
[`16_backpropagation`](../16_backpropagation/README.md) (the chain rule, done by hand — this
topic automates exactly that).

## Core concepts

- **Computational graph**: as operations are performed on tensors with `requires_grad=True`, PyTorch records them in a graph connecting inputs to outputs — this graph is what `.backward()` walks in reverse.
- **`requires_grad`**: marks a tensor as one PyTorch should track operations on, for later gradient computation. Model parameters have this set automatically; raw input data usually doesn't need it.
- **`.grad`**: after `.backward()`, each leaf tensor with `requires_grad=True` has its computed gradient accumulated here — note "accumulated," not "overwritten" (`.grad.zero_()` between steps matters, previewed here, used for real in `22_pytorch_training`).
- **`.backward()`**: triggers the automatic backward pass over the computational graph, computing `∂output/∂input` for every tracked tensor.
- **Chain rule (recap)**: autograd's internal mechanism is exactly `16_backpropagation`'s chain rule, applied automatically to an arbitrary graph of operations instead of a fixed 2-layer MLP derived by hand.
- **`torch.no_grad()`**: a context manager that disables gradient tracking — used during inference/evaluation, where gradients are never needed and tracking them wastes memory and compute.

```mermaid
flowchart LR
  A["Tensors with requires_grad=True"] --> B["Operations build a computational graph"]
  B --> C["Forward pass: compute output or loss"]
  C --> D["Call backward"]
  D --> E["Autograd walks the graph in reverse, applying the chain rule"]
  E --> F["grad populated on every leaf tensor"]
  G["torch.no_grad context"] -.->|"disables graph building"| B
  classDef data fill:#DBEAFE,stroke:#2563EB,color:#1a1a1a;
  classDef fwd fill:#DCFCE7,stroke:#16A34A,color:#1a1a1a;
  classDef back fill:#FEE2E2,stroke:#DC2626,color:#1a1a1a;
  classDef off fill:#FEF3C7,stroke:#D97706,color:#1a1a1a;
  class A data;
  class B,C fwd;
  class D,E,F back;
  class G off;
```

## 🔍 Concepts explained

> [!NOTE]
> Chips: 🟦 graph building · 🟥 backward · 🟨 accumulation · 🟩 turning tracking off.

### 🟦 Computational graph and `requires_grad`
- **Definition**: operations on tensors with `requires_grad=True` are recorded as a graph from inputs to output.
- **Analogy**: the flight recorder's log.
- **Code idea**: `w = torch.tensor(2.0, requires_grad=True); y = w * 3 + 1`.
- **Use when / watch out**: parameters need it; ordinary input data does not.

### 🟥 `.backward()` and `.grad`
- **Definition**: `.backward()` applies the chain rule from a scalar back through the graph; results land in `.grad` of leaf tensors.
- **Analogy**: replaying the log in reverse.
- **Code idea**: `y.backward(); w.grad  # tensor(3.)`.
- **Use when / watch out**: `.backward()` needs a scalar output (or an explicit gradient argument).

### 🟨 Gradient accumulation gotcha
- **Definition**: each `.backward()` **adds** into `.grad`; it does not overwrite.
- **Analogy**: a tally counter that keeps counting until you reset it.
- **Code idea**: `w.grad.zero_()` (or `optimizer.zero_grad()`) before each new backward.
- **Use when / watch out**: forgetting this silently doubles or inflates gradients across calls.

### 🟩 `torch.no_grad()` and `.detach()`
- **Definition**: `no_grad()` disables graph recording inside its block; `.detach()` returns a tensor cut from the graph.
- **Analogy**: switching the recorder off when you only want to fly, not log.
- **Use when / watch out**: use for inference and evaluation to save memory and compute; do not use it around code whose gradients you need.

### Differences at a glance

| | Tool | Effect | Use when |
|---|---|---|---|
| 🟦 | `requires_grad=True` | Start tracking this tensor | Learnable parameters |
| 🟥 | `.backward()` | Compute gradients into `.grad` | After computing a scalar loss |
| 🟨 | `.grad.zero_()` | Clear accumulated gradients | Before each new backward |
| 🟩 | `torch.no_grad()` | No graph for ops in the block | Inference, evaluation |
| 🟩 | `.detach()` | New tensor without graph history | Stop gradients flowing to a value |

> [!WARNING]
> Gradients accumulate. Without zeroing them between steps, every update uses the sum of all past gradients.

> [!TIP]
> Verify autograd against your hand-derived gradients from `16_backpropagation` with `torch.allclose`, as the notebook does.

### 🗺️ Concept map

```mermaid
flowchart LR
  subgraph BUILD["Forward: build graph"]
    RG["requires_grad=True"] --> OP["Operations recorded"] --> LS["Scalar loss"]
  end
  subgraph BACK["Backward"]
    BW[".backward()"] --> GR[".grad on leaves"]
  end
  subgraph CTRL["Control"]
    Z["grad.zero_()"]
    NG["no_grad / detach"]
  end
  LS --> BW
  GR --> Z
  NG -.->|"skips recording"| OP
  classDef build fill:#DBEAFE,stroke:#2563EB,color:#1a1a1a;
  classDef back fill:#FEE2E2,stroke:#DC2626,color:#1a1a1a;
  classDef ctrl fill:#FEF3C7,stroke:#D97706,color:#1a1a1a;
  class RG,OP,LS build;
  class BW,GR back;
  class Z,NG ctrl;
```

## Important APIs / functions

`torch.tensor(..., requires_grad=True)`, `.backward()`, `.grad`, `.grad.zero_()`,
`torch.no_grad()`, `.detach()`.

## Notebooks

| Notebook | Purpose | Expected output |
|---|---|---|
| `01_autograd_basics.ipynb` | A tiny scalar example (matching `16_backpropagation`'s hand-derived `a*b+1` example) verified via autograd; the 2-layer MLP from `15_deep_learning`/`16_backpropagation` rebuilt in PyTorch, with `.backward()`'s gradients compared directly against the hand-derived NumPy gradients; a `.grad` accumulation gotcha demonstrated and fixed; `torch.no_grad()` used for inference | Autograd's gradients matching the hand-derived gradients from `16_backpropagation` to floating-point precision; a clear demonstration of `.grad` silently accumulating across calls without `zero_()`, and the fix |

## Key takeaways

- Autograd computes nothing new mathematically — it automates exactly the chain-rule bookkeeping done by hand in `16_backpropagation`.
- `.grad` accumulates by default; forgetting to zero it between training steps is one of the most common PyTorch training bugs (a silent one — training often still "sort of works" but incorrectly).
- `torch.no_grad()` isn't just a performance optimization — using it during evaluation prevents accidentally building up a graph you'll never need, and signals intent clearly.

## Common mistakes

- Forgetting `optimizer.zero_grad()` (or manually `.grad.zero_()`) between training steps, silently accumulating gradients across batches.
- Calling `.numpy()` on a tensor that still requires grad (PyTorch will raise an error — `.detach()` first).
- Doing unnecessary gradient tracking during evaluation/inference, wasting memory.
- Calling `.backward()` twice on the same graph without `retain_graph=True`, when that's not actually needed (usually a sign of a structural bug elsewhere).

## Interview questions

**Conceptual**
1. What does `requires_grad=True` actually do, mechanically?
2. Why does `.grad` accumulate instead of overwrite by default, and why does that matter?
3. What's the difference between `.detach()` and `torch.no_grad()`?
4. Why is `.backward()` able to compute gradients for an arbitrary chain of operations, not just a fixed architecture?
5. What happens if you try to call `.backward()` a second time on the same graph without `retain_graph=True`?

**Coding**
1. Build the `a*b+1` scalar example from `16_backpropagation` in PyTorch with `requires_grad=True`, call `.backward()`, and verify `.grad` matches the hand-derived gradient.
2. Rebuild the 2-layer MLP forward pass from `15_deep_learning` in PyTorch and verify `.backward()`'s gradients match `16_backpropagation`'s hand-derived NumPy gradients.
3. Demonstrate the `.grad` accumulation bug by calling `.backward()` twice without zeroing gradients, and show the fix.

**Debugging**
1. You call `.numpy()` on a tensor and get `RuntimeError: Can't call numpy() on Tensor that requires grad` — what's the fix, and why does PyTorch enforce this?
2. Your training loss looks like it's training correctly but convergence is oddly slow/unstable — you suspect a missing `zero_grad()` — how would you verify this?

**Real-world engineering**
1. You're running inference on a trained model in production — why wrap the forward pass in `torch.no_grad()`?
2. You're implementing a custom operation that autograd can't differentiate through automatically — what PyTorch mechanism would you reach for (conceptually)?

## Exercises

1. Build a small computational graph with at least 4 chained operations, and manually verify `.backward()`'s gradient against a hand-derived chain-rule calculation.
2. Reproduce the `.grad` accumulation bug intentionally, observe the incorrect (doubled) gradient, then fix it with `.grad.zero_()`.
3. Time a forward+backward pass with and without `torch.no_grad()` (forward only, since `no_grad` disables backward) and compare memory/time.

## References

- [PyTorch docs — Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [PyTorch docs — `torch.autograd`](https://pytorch.org/docs/stable/autograd.html)
- [PyTorch tutorials — Autograd](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
