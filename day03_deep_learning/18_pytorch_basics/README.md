# 18 — PyTorch Basics: Tensors, Devices, Shapes

## Learning objectives

By the end of this topic you can:
- Create and inspect PyTorch tensors (shape, dtype, device) and convert between NumPy and PyTorch seamlessly.
- Move tensors between CPU and GPU (CUDA/MPS) and explain why device mismatches cause errors.
- Index, reshape/view, transpose, broadcast, and matrix-multiply PyTorch tensors — the exact same operations from Day 1's `02_numpy`, now on a framework built for autograd and GPU execution.

## Why it matters

Everything from here to the end of the curriculum (and into `llm-fine-tuning/`) is PyTorch.
The good news: if Day 1's `02_numpy` is solid, this topic is almost entirely new vocabulary
for familiar operations — the shape-reasoning skill transfers directly.

## Prerequisites

Day 1's [`02_numpy`](../../day01_python_data_math/02_numpy/README.md) (this topic is the same
material, ported) and [`17_deep_learning_architectures`](../17_deep_learning_architectures/README.md)
(so there's a reason to want GPU tensors — attention/matmul at scale).

## Core concepts

- **Tensor**: PyTorch's `ndarray` equivalent — the difference is tensors can live on a GPU and track gradients (next topic, `19_pytorch_autograd`).
- **Shape / dtype / device**: a PyTorch tensor is defined by its shape (same as NumPy), dtype (`torch.float32` is the deep-learning default, not `float64`), and device (`cpu`, `cuda`, or `mps` on Apple Silicon) — an operation between tensors on different devices raises an error.
- **CPU vs GPU, CUDA**: GPUs parallelize the massive matrix multiplications deep learning relies on; `tensor.to(device)` moves data between CPU and GPU memory — a real cost, so minimizing unnecessary transfers matters for performance (previewed here, detailed in `23_pytorch_performance`).
- **Indexing / reshape / view / transpose**: nearly identical syntax to NumPy, with one crucial difference — `.view()` requires the underlying memory to be contiguous, while `.reshape()` will copy if needed (NumPy's `.reshape()` quietly does the same thing internally).
- **Broadcasting**: identical rules to NumPy (Day 1's `02_numpy`).
- **Matrix multiplication**: `@` and `torch.matmul` — same semantics as NumPy, now potentially running on a GPU.

```mermaid
flowchart LR
  A[NumPy ndarray] -->|torch.from_numpy| B[PyTorch tensor, CPU]
  B -->|.to('cuda') / .to('mps')| C[PyTorch tensor, GPU]
  C -->|.cpu().numpy()| A
  B --> D[Same ops as NumPy: index/reshape/transpose/broadcast/matmul]
  D --> E[19_pytorch_autograd: gradients tracked automatically]
```

## Important APIs / functions

`torch.tensor`, `.shape`/`.dtype`/`.device`, `torch.from_numpy`/`.numpy()`, `.to(device)`,
`torch.cuda.is_available()`, `torch.backends.mps.is_available()`, `.view`/`.reshape`/`.T`,
`torch.matmul`/`@`.

## Notebooks

| Notebook | Purpose | Expected output |
|---|---|---|
| `01_tensors_devices_shapes.ipynb` | Tensor creation, dtype/device inspection, NumPy↔PyTorch conversion, device detection and movement (CPU/CUDA/MPS, whichever is available), indexing/reshape/view/transpose/broadcasting/matmul, each mirrored against Day 1's NumPy equivalent | Every operation's PyTorch result verified numerically identical to its NumPy equivalent; a device-mismatch error deliberately triggered and explained; the fastest available device on the current machine detected and used for a matmul timing comparison |

## Key takeaways

- PyTorch tensors and NumPy arrays share a mental model (shape/dtype/broadcasting) almost completely — the new concepts are device placement and (next topic) gradient tracking.
- `float32` is the deep learning default; using `float64` by accident (e.g. from an unconverted NumPy array) wastes memory and can silently slow down GPU code.
- A device mismatch (one tensor on CPU, another on GPU) is one of the most common first errors — always check `.device` when debugging a shape/type error that doesn't make sense.

## Common mistakes

- Trying to operate on two tensors on different devices without moving one first.
- Converting a NumPy `float64` array to a tensor without explicitly casting to `float32`.
- Calling `.view()` on a non-contiguous tensor (e.g. after `.transpose()`) and hitting a `RuntimeError` — `.reshape()` or `.contiguous().view()` are the fixes.
- Forgetting that moving data to/from a GPU has real latency cost, and doing it unnecessarily inside a hot loop.

## Interview questions

**Conceptual**
1. What three things define a PyTorch tensor, beyond just its values?
2. Why does PyTorch default to `float32` instead of `float64` for most deep learning work?
3. Why does an operation between a CPU tensor and a GPU tensor raise an error instead of silently working?
4. What's the difference between `.view()` and `.reshape()`?
5. Why would you want to minimize the number of `.to(device)` calls in a training loop?

**Coding**
1. Create a tensor from a NumPy array, move it to the best available device, and move it back, verifying the values are unchanged.
2. Reproduce 5 operations from Day 1's `02_numpy` notebook (indexing, broadcasting, matmul) in PyTorch and assert the results match NumPy's.
3. Deliberately create a device mismatch error, catch it, and print a clear diagnostic message.

**Debugging**
1. You get `RuntimeError: view size is not compatible with input tensor's size and stride` — what's the likely cause, and what's the fix?
2. A GPU training loop runs far slower than expected, and you suspect excessive CPU↔GPU transfers — what would you check first?

**Real-world engineering**
1. Your code needs to run identically on a machine with a GPU and one without — how do you write device-agnostic code in PyTorch?
2. You're debugging a shape mismatch deep inside a model's forward pass — what's your systematic approach using `.shape` prints?

## Exercises

1. Benchmark the same large matmul on CPU vs the best available accelerated device (CUDA or MPS) and report the speedup.
2. Write a `get_device()` helper that returns `cuda` if available, else `mps`, else `cpu`, and use it throughout the rest of Day 3's notebooks.
3. Convert a non-contiguous tensor (after `.transpose()`) and show both the `.view()` failure and the `.reshape()`/`.contiguous().view()` fix.

## References

- [PyTorch docs — Tensors tutorial](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [PyTorch docs — `torch.Tensor`](https://pytorch.org/docs/stable/tensors.html)
- [PyTorch docs — CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [PyTorch docs — MPS backend](https://pytorch.org/docs/stable/notes/mps.html)
