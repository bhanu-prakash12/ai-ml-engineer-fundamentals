# Day 3 — Deep Learning + PyTorch

> From a single neuron to a Transformer, then the PyTorch stack that trains it and the Hugging Face on-ramp to LLMs.

| | |
|---|---|
| **Day** | 3 |
| **Topic** | Deep learning + PyTorch |
| **Notebooks** | 12 across 10 topics (15 to 24) |
| **Builds on** | Day 1 `05_ml_mathematics`; Day 2 `10_ml_optimization`, `11_generalization` |
| **Leads to** | Capstone and the `llm-fine-tuning/` curriculum |

## 📖 Short forms used in this topic

| Term | Full form | What it is |
|---|---|---|
| MLP | Multi-Layer Perceptron | Fully-connected network with non-linear activations |
| ReLU / GELU | Rectified / Gaussian Error Linear Unit | Common activation functions |
| MSE / BCE / CE | Mean Squared Error / Binary Cross-Entropy / Cross-Entropy | Standard losses for regression, binary and multiclass tasks |
| CNN | Convolutional Neural Network | Shared-kernel network for grid data such as images |
| RNN / LSTM / GRU | Recurrent NN / Long Short-Term Memory / Gated Recurrent Unit | Sequential models; LSTM and GRU add gates |
| Q / K / V | Query / Key / Value | Attention projections |
| FFN | Feed-Forward Network | Per-position MLP inside a Transformer block |
| LayerNorm | Layer Normalization | Per-sample feature normalization |
| CPU / GPU | Central / Graphics Processing Unit | Processors tensors run on |
| CUDA / MPS | Compute Unified Device Architecture / Metal Performance Shaders | NVIDIA and Apple GPU backends |
| FP32 / FP16 / BF16 | 32-bit / 16-bit / Brain Float 16 | Floating-point formats for precision and speed trade-offs |
| LR | Learning Rate | Step size of weight updates |
| GIL | Global Interpreter Lock | Why `DataLoader` workers are processes |
| LLM / GPT | Large Language Model / Generative Pre-trained Transformer | Large causal Transformer language models |
| PEFT / LoRA / QLoRA | Parameter-Efficient Fine-Tuning / Low-Rank Adaptation / Quantized LoRA | Cheap ways to fine-tune big models |
| HF | Hugging Face | Library ecosystem used in topic 24 |

## 🧠 In plain words

Deep learning is training stacked layers of simple math to map inputs to outputs, using gradients from backpropagation to improve them. Day 3 first builds that by hand in NumPy, then repeats it with PyTorch's tensors, autograd, modules, data loaders and training loops, and finishes by loading a pretrained Transformer from Hugging Face.

## 🎒 Analogy to remember

Learning to cook: first make a dish **from scratch by hand** (topics 15 to 17), then **learn the professional kitchen** (PyTorch, 18 to 23) where equipment does the tedious parts, and finally **order from a catering supplier** (Hugging Face, 24) and adjust a ready-made dish. *Where it breaks down:* in deep learning the "recipe" is learned from data, not written by a chef.

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

## 🔍 Concepts explained

> [!NOTE]
> Colour key used in all Day 3 topics: 🟦 foundations (15 to 17) · 🟩 PyTorch core (18 to 21) · 🟧 pipeline and speed (22 to 23) · 🟪 ecosystem (24).

| | Topic | Core idea | Key takeaway |
|---|---|---|---|
| 🟦 | [15 Deep learning](15_deep_learning/README.md) | Layers, activations, losses | Non-linearity is what makes depth useful |
| 🟦 | [16 Backpropagation](16_backpropagation/README.md) | Chain rule with cached values | Check gradients numerically |
| 🟦 | [17 Architectures](17_deep_learning_architectures/README.md) | CNN, RNN, attention, Transformer | Attention parallelizes; residuals and norms make depth trainable |
| 🟩 | [18 PyTorch basics](18_pytorch_basics/README.md) | Tensor: shape, dtype, device | Same device for all operands |
| 🟩 | [19 Autograd](19_pytorch_autograd/README.md) | Graph plus `.backward()` | Gradients accumulate; zero them |
| 🟩 | [20 Models](20_pytorch_models/README.md) | `nn.Module`, layers | `eval()` for inference |
| 🟩 | [21 Data](21_pytorch_data/README.md) | `Dataset`, `DataLoader` | Workers are processes; `collate_fn` for variable length |
| 🟧 | [22 Training](22_pytorch_training/README.md) | Train, validate, checkpoint, stop | Keep the best checkpoint; seed for reproducibility |
| 🟧 | [23 Performance](23_pytorch_performance/README.md) | Mixed precision, accumulation, clipping | Profile before optimizing |
| 🟪 | [24 Hugging Face](24_huggingface_foundations/README.md) | Tokenizer, model, `generate()`, LoRA | Attention mask on padded batches |

> [!TIP]
> Study order matters: each topic's gradients, shapes, or layers reappear in the next. The hand-derived gradients in 16 are exactly what autograd verifies in 19.

### 🗺️ Day 3 topic flow

```mermaid
flowchart LR
  subgraph FOUND["Foundations by hand"]
    T15["15 Deep learning"] --> T16["16 Backprop"] --> T17["17 Architectures"]
  end
  subgraph CORE["PyTorch core"]
    T18["18 Tensors"] --> T19["19 Autograd"] --> T20["20 nn.Module"] --> T21["21 Data"]
  end
  subgraph PIPE["Pipeline and speed"]
    T22["22 Training"] --> T23["23 Performance"]
  end
  subgraph ECO["Ecosystem"]
    T24["24 Hugging Face"]
  end
  T17 --> T18
  T21 --> T22
  T23 --> T24
  T24 --> LLM["llm-fine-tuning curriculum"]
  classDef found fill:#DBEAFE,stroke:#2563EB,color:#1a1a1a;
  classDef core fill:#DCFCE7,stroke:#16A34A,color:#1a1a1a;
  classDef pipe fill:#FEF3C7,stroke:#D97706,color:#1a1a1a;
  classDef eco fill:#EDE9FE,stroke:#7C3AED,color:#1a1a1a;
  class T15,T16,T17 found;
  class T18,T19,T20,T21 core;
  class T22,T23 pipe;
  class T24,LLM eco;
```

See the repo root `ROADMAP.md` for how this connects to Days 1-2 and the capstone, and this
workspace's `llm-fine-tuning/` curriculum for what comes after.
