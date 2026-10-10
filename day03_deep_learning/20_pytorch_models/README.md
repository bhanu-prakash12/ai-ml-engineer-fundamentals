# 20 — PyTorch Models: `nn.Module`

## Learning objectives

By the end of this topic you can:
- Build a model by subclassing `nn.Module`, defining layers in `__init__` and the forward pass in `forward()`.
- Use `nn.Linear`, `nn.Embedding`, `nn.ReLU`/`nn.GELU`, `nn.LayerNorm`, `nn.Dropout`, `nn.Softmax` correctly.
- Rebuild the 2-layer MLP and the Transformer block from `15-17` as proper `nn.Module`s.

## Why it matters

`nn.Module` is the organizing unit of every PyTorch model, from a 2-layer MLP to a full LLM.
It automatically registers parameters (so an optimizer can find them) and gives every model a
consistent `forward()` interface — this is the structure `22_pytorch_training`'s training loop
expects.

## Prerequisites

[`19_pytorch_autograd`](../19_pytorch_autograd/README.md) (the gradients this wraps) and
[`17_deep_learning_architectures`](../17_deep_learning_architectures/README.md) (the
architecture being rebuilt as a module).

## Core concepts

- **`nn.Module`**: the base class for all PyTorch models/layers. Subclass it, define sub-layers in `__init__`, and implement `forward()` — calling the module (`model(x)`) invokes `forward()` automatically and handles bookkeeping.
- **`nn.Linear`**: a fully-connected layer (`Y = XW^T + b`) — replaces manually tracking `W`/`b` tensors from `19_pytorch_autograd`, and registers its weights as parameters automatically.
- **`nn.Embedding`**: a lookup table mapping integer token IDs to dense vectors — the first layer of almost every NLP/LLM model.
- **`nn.ReLU` / `nn.GELU`**: activation functions as modules (vs. the functional `torch.relu` used directly in `19_pytorch_autograd`) — useful for putting activations inside `nn.Sequential`.
- **`nn.LayerNorm`**: the production version of `17_deep_learning_architectures`'s from-scratch `layer_norm` function.
- **`nn.Dropout`**: randomly zeroes elements during training (recap from Day 2's `11_generalization`) — automatically disabled during `.eval()` mode.
- **`nn.Softmax`**: softmax as a module, for use inside a `forward()` method or `nn.Sequential`.
- **Parameters**: every `nn.Linear`/`nn.Embedding`/etc. registers its weights as `nn.Parameter`s automatically — `model.parameters()` yields every learnable tensor in the whole module tree, which is exactly what an optimizer needs.

```mermaid
flowchart LR
  A["class MyModel(nn.Module)"] --> B["__init__: define sub-layers (nn.Linear, nn.Embedding, ...)"]
  A --> C["forward(x): wire sub-layers together"]
  B --> D[Parameters auto-registered]
  D --> E[model.parameters() for the optimizer]
  C --> F[model(x) calls forward() automatically]
```

## Important APIs / functions

`nn.Module`, `nn.Linear`, `nn.Embedding`, `nn.ReLU`, `nn.GELU`, `nn.LayerNorm`, `nn.Dropout`,
`nn.Softmax`, `nn.Sequential`, `model.parameters()`, `model.train()`/`model.eval()`.

## Notebooks

| Notebook | Purpose | Expected output |
|---|---|---|
| `01_nn_module_basics.ipynb` | The 2-layer MLP from `19_pytorch_autograd` rewritten as a proper `nn.Module`; each layer type (`nn.Linear`/`nn.Embedding`/`nn.LayerNorm`/`nn.Dropout`) demonstrated individually; `model.train()` vs `model.eval()`'s effect on dropout; a from-scratch-equivalent Transformer block rebuilt using `nn.MultiheadAttention` + `nn.LayerNorm` + `nn.Linear` | The `nn.Module`-based MLP producing numerically identical output to the manual-tensor version from `19_pytorch_autograd`, given the same weights; dropout demonstrably active in `.train()` mode and inactive in `.eval()` mode |

## Key takeaways

- `nn.Module` is purely organizational — it doesn't change the underlying math, just makes parameter management and composition automatic (no more manually tracking a list of weight tensors).
- `model.parameters()` is what connects a model to an optimizer (`22_pytorch_training`) — it recursively finds every registered parameter in every sub-module.
- Dropout and some other layers behave differently in train vs eval mode — always call `.eval()` before inference and `.train()` before resuming training.

## Common mistakes

- Forgetting to call `model.eval()` before inference, leaving dropout active and getting non-deterministic predictions.
- Defining a layer in `forward()` instead of `__init__` — this creates a new, untrained layer on every call instead of reusing registered parameters.
- Not calling the parent `nn.Module.__init__()` (via `super().__init__()`) before registering sub-layers, which breaks parameter registration silently.

## Interview questions

**Conceptual**
1. What does subclassing `nn.Module` give you that manually managing tensors (as in `19_pytorch_autograd`) doesn't?
2. Why must sub-layers be defined in `__init__`, not `forward()`?
3. What's the practical difference between `model.train()` and `model.eval()`, concretely for dropout?
4. Why does `model.parameters()` need to recurse into sub-modules?
5. What does `nn.Embedding` do differently from `nn.Linear`?

**Coding**
1. Rewrite the manual-tensor 2-layer MLP from `19_pytorch_autograd` as an `nn.Module` subclass and verify identical output given the same weights.
2. Build an `nn.Sequential` stack equivalent to a hand-defined `nn.Module` with the same layers, and verify both produce the same output.
3. Demonstrate `nn.Dropout`'s different behavior in `.train()` vs `.eval()` mode on the same input.

**Debugging**
1. Your model's parameters aren't updating during training — you forgot `super().__init__()` in your `nn.Module` subclass — what symptom would you observe, and why?
2. Your model gives different outputs on identical inputs at inference time — what mode check would you add first?

**Real-world engineering**
1. You're deploying a trained model for inference-only serving — what two things should you make sure happen before serving predictions (hint: mode and gradient tracking)?
2. You're building a large model with many repeated sub-blocks — how would `nn.Module` composition help you avoid duplicating code?

## Exercises

1. Build a 3-layer MLP as an `nn.Module`, print `model.parameters()`'s shapes, and verify the total parameter count by hand.
2. Build an `nn.Embedding` layer, look up a batch of token IDs, and verify the output shape is `(batch, sequence_length, embedding_dim)`.
3. Rebuild `17_deep_learning_architectures`'s from-scratch Transformer block using `nn.MultiheadAttention`, `nn.LayerNorm`, and `nn.Linear`, and verify the output shape matches the from-scratch version.

## References

- [PyTorch docs — `nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [PyTorch tutorials — Build the Neural Network](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- [PyTorch docs — `nn.MultiheadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
