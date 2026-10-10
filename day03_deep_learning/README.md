# Day 3 — Deep Learning + PyTorch

Builds on Day 1's math (`05_ml_mathematics`) and Day 2's optimization/generalization topics.
Covers neural network fundamentals, backpropagation, architectures (CNN/RNN/Transformer), and
the full PyTorch stack (tensors → autograd → `nn.Module` → data → training → performance),
closing with an introduction to Hugging Face. Built with the same create → execute → validate
→ fix method as Days 1 and 2.

## Topics

1. [`15_deep_learning`](15_deep_learning/README.md) — perceptron, MLP forward pass, activations, losses
2. [`16_backpropagation`](16_backpropagation/README.md) — the chain rule, by hand, verified numerically
3. [`17_deep_learning_architectures`](17_deep_learning_architectures/README.md) — CNN, RNN/LSTM/GRU, attention, Transformer block
4. [`18_pytorch_basics`](18_pytorch_basics/README.md) — tensors, devices, shapes
5. [`19_pytorch_autograd`](19_pytorch_autograd/README.md) — `.backward()`, `.grad`, `torch.no_grad()`
6. [`20_pytorch_models`](20_pytorch_models/README.md) — `nn.Module`, `nn.Linear`/`Embedding`/`LayerNorm`/`Dropout`
7. [`21_pytorch_data`](21_pytorch_data/README.md) — `Dataset`, `DataLoader`, `collate_fn`, `num_workers`
8. [`22_pytorch_training`](22_pytorch_training/README.md) — the complete training pipeline, checkpointing, early stopping
9. [`23_pytorch_performance`](23_pytorch_performance/README.md) — mixed precision, gradient accumulation/clipping, profiling
10. [`24_huggingface_foundations`](24_huggingface_foundations/README.md) — tokenizers, `AutoModel`, `generate()`, LoRA concept

See the repo root `ROADMAP.md` for how this connects to Days 1-2 and the capstone, and this
workspace's `llm-fine-tuning/` curriculum for what comes after.
