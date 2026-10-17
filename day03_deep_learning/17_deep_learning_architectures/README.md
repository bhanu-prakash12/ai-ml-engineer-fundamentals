# 17 — Deep Learning Architectures: CNN, RNN/LSTM/GRU, Attention, Transformer

> Three ways to exploit structure in data: convolution for space, recurrence for order, attention for everything-to-everything — and the Transformer is the one that scaled.

| | |
|---|---|
| **Day** | 3 — Deep Learning + PyTorch |
| **Topic** | CNN, RNN/LSTM/GRU, attention, Transformer block |
| **Notebooks** | 3 (`01_cnn_from_scratch`, `02_rnn_lstm_gru_concept`, `03_attention_and_transformer_block`) |
| **Builds on** | [`16_backpropagation`](../16_backpropagation/README.md), Day 1 `02_numpy` |
| **Leads to** | [`18_pytorch_basics`](../18_pytorch_basics/README.md), [`20_pytorch_models`](../20_pytorch_models/README.md), `llm-fine-tuning/` |

## 📖 Short forms used in this topic

| Term | Full form | What it is |
|---|---|---|
| CNN | Convolutional Neural Network | Network that slides shared small filters over grid-like data (images) |
| RNN | Recurrent Neural Network | Processes a sequence step by step, carrying a hidden state |
| LSTM | Long Short-Term Memory | RNN variant with forget/input/output gates and a cell state |
| GRU | Gated Recurrent Unit | Simpler gated RNN with reset and update gates |
| Q / K / V | Query / Key / Value | The three projections of each token used by attention |
| `d_k` | Key dimension | Size of key vectors; attention scores are divided by `√d_k` |
| FFN | Feed-Forward Network | Per-position MLP inside a Transformer block |
| MLP | Multi-Layer Perceptron | Fully-connected layers with non-linearities |
| LayerNorm | Layer Normalization | Normalizes the features of each sample independently |
| ReLU | Rectified Linear Unit | Activation `max(0, x)` |
| GPT | Generative Pre-trained Transformer | Decoder-only (causal) Transformer language model family |
| LLM | Large Language Model | Very large Transformer trained on text |
| NumPy | Numerical Python | Array library used for the from-scratch code |

## 🧠 In plain words

Different data has different structure, and each architecture bakes in an assumption about it. CNNs assume nearby pixels matter together and patterns repeat anywhere. RNNs assume order matters and read one step at a time. Transformers let every token look directly at every other token and learn which ones matter, which trains in parallel and scales to LLMs.

## 🎒 Analogy to remember

Reading a document: a **CNN** is a **magnifying glass** you slide across the page looking for one kind of pattern. An **RNN** is reading word by word while keeping **notes in your head** that fade over long passages. **Attention** is having the whole page spread on a table with a **highlighter**: for each word you highlight the other words that help explain it. *Where it breaks down:* the highlighter is soft (weights sum to 1, not on/off) and each attention head highlights differently.

## Learning objectives

By the end of this topic you can:
- Explain convolution, kernels, stride, padding, and pooling, and implement a 2D convolution from scratch.
- Explain RNN/LSTM/GRU and why vanilla RNNs struggle with long sequences.
- Explain query/key/value, self-attention, cross-attention, and multi-head attention — and implement scaled dot-product attention from scratch.
- Describe the full Transformer block: embeddings, positional encoding, multi-head attention, FFN, residual connections, normalization, encoder/decoder, and causal masking.

## Why it matters

This topic is the direct on-ramp to everything in this workspace's `llm-fine-tuning/`
curriculum — every LLM is a stack of Transformer blocks. Understanding *why* attention
replaced RNNs (parallelizable, no vanishing-gradient-over-time problem) and what each piece of
a Transformer block does is the difference between using Hugging Face models as a black box and
actually understanding what you're fine-tuning.

## Prerequisites

[`16_backpropagation`](../16_backpropagation/README.md) (gradients through a network);
Day 1's [`02_numpy`](../../day01_python_data_math/02_numpy/README.md) (shape reasoning, now
applied to much higher-dimensional tensors).

## Core concepts

### CNNs
- **Convolution**: slide a small learned filter (**kernel**) across the input, computing a dot product at each position — detects local patterns (edges, textures) regardless of position.
- **Stride**: how far the kernel moves per step — larger stride = smaller output, less overlap.
- **Padding**: adding border pixels (usually zeros) so the output size can match the input size ("same" padding) instead of always shrinking ("valid" padding).
- **Pooling**: downsampling (e.g. max pooling) — reduces spatial size and adds translation invariance.

### RNNs
- **RNN**: processes a sequence step by step, carrying a hidden state forward — in principle captures arbitrary-length context, but **vanishing/exploding gradients** over many steps make long-range dependencies hard to learn.
- **LSTM / GRU**: add gating mechanisms (forget/input/output gates for LSTM; reset/update gates for GRU) specifically to control what the hidden state remembers vs forgets, mitigating vanishing gradients — GRU is a simpler, often similarly effective variant of LSTM.

### Attention and Transformers
- **Query/Key/Value**: for each position, compute a query ("what am I looking for"), compared against every position's key ("what do I offer") to get attention weights, which weight every position's value ("what information do I carry").
- **Self-attention**: queries, keys, and values all come from the *same* sequence — every token can directly attend to every other token, in one step (unlike an RNN's step-by-step propagation).
- **Cross-attention**: queries come from one sequence, keys/values from another (e.g. decoder attending to encoder outputs in a translation model).
- **Multi-head attention**: run several attention computations ("heads") in parallel, each potentially learning a different kind of relationship, then combine the results.
- **Transformer block**: embeddings → (+ positional encoding, since attention has no inherent sense of order) → multi-head self-attention → residual connection + normalization → feed-forward network (FFN) → another residual connection + normalization.
- **Encoder/decoder**: an encoder processes the full input at once (bidirectional attention); a decoder generates output step by step and uses **causal masking** to prevent attending to future tokens it hasn't generated yet.
- **Residual connections**: `output = x + f(x)` — lets gradients flow directly through many stacked layers, critical for training very deep networks.
- **Normalization (LayerNorm)**: normalizes activations within each sample, stabilizing training in deep stacks of layers.

```mermaid
flowchart TD
  A["Tokens"] --> B["Embeddings + Positional Encoding"]
  B --> C["Multi-Head Self-Attention"]
  C --> D["Residual Add + LayerNorm"]
  D --> E["Feed-Forward Network"]
  E --> F["Residual Add + LayerNorm"]
  F -->|"stack N times"| C
  F --> G["Output: next-token logits or encoded representation"]
  H["Causal mask"] -.->|"restricts attention to the past"| C
  classDef inp fill:#DBEAFE,stroke:#2563EB,color:#1a1a1a;
  classDef attn fill:#EDE9FE,stroke:#7C3AED,color:#1a1a1a;
  classDef norm fill:#FEF3C7,stroke:#D97706,color:#1a1a1a;
  classDef ffn fill:#DCFCE7,stroke:#16A34A,color:#1a1a1a;
  classDef mask fill:#FEE2E2,stroke:#DC2626,color:#1a1a1a;
  class A,B,G inp;
  class C attn;
  class D,F norm;
  class E ffn;
  class H mask;
```

## 🔍 Concepts explained

> [!NOTE]
> Chips: 🟦 CNN · 🟩 RNN family · 🟪 attention and Transformer · 🟨 stabilizers (residual, normalization).

### 🟦 Convolution, stride, padding, pooling
- **Definition**: a small shared kernel slides over the input, taking a dot product at each position; stride sets the step, padding adds a border, pooling downsamples.
- **Analogy**: a stamp-sized magnifying glass swept across a photo.
- **Formula**: output size `= floor((H - K + 2P) / S) + 1` (with `P = 0, S = 1`: `H - K + 1`).
- **Use when / watch out**: images and other grids; padding and stride change output size, so track shapes.

### 🟩 RNN, LSTM, GRU
- **Definition**: an RNN updates a hidden state `h_t = f(x_t, h_{t-1})`; LSTM/GRU add gates controlling what to keep or forget.
- **Analogy**: a note-taker whose notes get fainter with every page; gates are an editor deciding what to keep.
- **Use when / watch out**: small sequential problems; gradients vanish (or explode) over long sequences, and steps cannot be parallelized in training.

### 🟪 Attention (Q/K/V, self, cross, multi-head)
- **Definition**: each token's query is compared to all keys; the softmax-ed scores weight the values.
- **Analogy**: a search engine: query = your search, keys = page titles, values = page contents.
- **Formula**: `Attention(Q,K,V) = softmax(QKᵀ / √d_k) V`
- **Use when / watch out**: skip the `√d_k` scaling and softmax saturates, giving peaky weights and unstable gradients.

### 🟪 Transformer block
- **Definition**: attention then FFN, each wrapped with a residual connection and normalization; token embeddings plus positional encoding go in.
- **Analogy**: a meeting round (attention: tokens talk to each other) followed by individual thinking time (FFN).
- **Code idea**: `x = LN(x + MHA(x)); x = LN(x + FFN(x))` (post-norm, as in the original paper; many modern models use pre-norm).
- **Use when / watch out**: output shape must equal input shape so blocks stack.

### 🟪 Causal masking
- **Definition**: set scores for future positions to `-inf` before softmax, so each token sees only itself and the past.
- **Analogy**: an exam where answers for later questions are covered up.
- **Use when / watch out**: required for decoder or autoregressive training; forgetting it leaks the answer.

### 🟨 Residual connection and LayerNorm
- **Definition**: `x + f(x)` gives gradients a direct path; LayerNorm normalizes each sample's features.
- **Analogy**: a highway bypass beside a winding road; LayerNorm keeps every car at a similar speed.
- **Use when / watch out**: needed for deep stacks.

### Differences at a glance

| | Architecture | Exploits | Parallel across time/space? | Long-range context | Typical pitfall |
|---|---|---|---|---|---|
| 🟦 | CNN | Spatial locality, weight sharing | Yes | Limited by receptive field | Shape errors from stride/padding |
| 🟩 | RNN | Order, via hidden state | No (sequential) | Poor (vanishing gradients) | Gradients vanish or explode |
| 🟩 | LSTM / GRU | Order, with gates | No (sequential) | Better, but still limited | Assuming gates fully fix vanishing |
| 🟪 | Transformer | All-pairs interaction | Yes | Direct, any distance | Attention cost grows with length² |

| | Attention type | Q from | K, V from | Typical use |
|---|---|---|---|---|
| 🟪 | Self-attention | Sequence X | Same sequence X | Encoder, decoder-only LMs |
| 🟪 | Causal self-attention | Sequence X | Same X, future masked | GPT-style generation |
| 🟪 | Cross-attention | Decoder sequence | Encoder outputs | Translation (encoder-decoder) |
| 🟪 | Multi-head | Several parallel heads of any above | | Learn different relations at once |

> [!IMPORTANT]
> Attention has no built-in notion of order. Positional encoding is what tells a Transformer which token is where; an RNN gets order for free from its step-by-step processing.

> [!WARNING]
> LSTM/GRU mitigate vanishing gradients but do not eliminate them for very long sequences. Missing `√d_k` scaling or a missing causal mask are the two classic attention bugs.

### 🗺️ Concept map

```mermaid
flowchart TD
  D["Data type"] --> G{"What structure?"}
  G -->|"grid: images"| CN["CNN"]
  G -->|"sequence, small or streaming"| RN["RNN / LSTM / GRU"]
  G -->|"sequence, long and parallel"| TR["Transformer"]
  subgraph CNNG["CNN parts"]
    K["Kernel"] --> ST["Stride and padding"] --> PL["Pooling"]
  end
  subgraph RNNG["Recurrent parts"]
    H["Hidden state"] --> GT["Gates: forget, input, output"]
  end
  subgraph TRG["Transformer parts"]
    AT["Q, K, V attention"] --> MH["Multi-head"] --> BL["Residual + LayerNorm + FFN"]
  end
  CN --> K
  RN --> H
  TR --> AT
  classDef start fill:#F3F4F6,stroke:#6B7280,color:#1a1a1a;
  classDef cnn fill:#DBEAFE,stroke:#2563EB,color:#1a1a1a;
  classDef rnn fill:#DCFCE7,stroke:#16A34A,color:#1a1a1a;
  classDef tr fill:#EDE9FE,stroke:#7C3AED,color:#1a1a1a;
  class D,G start;
  class CN,K,ST,PL cnn;
  class RN,H,GT rnn;
  class TR,AT,MH,BL tr;
```

## Important APIs / functions

Pure NumPy for this topic (a from-scratch 2D convolution, a from-scratch RNN cell, from-scratch
scaled dot-product attention) — PyTorch's `nn.Conv2d`/`nn.RNN`/`nn.MultiheadAttention`
equivalents are introduced for real in `20_pytorch_models`.

## Notebooks

| Notebook | Purpose | Expected output |
|---|---|---|
| `01_cnn_from_scratch.ipynb` | 2D convolution, stride, padding, and max pooling implemented from scratch on a small synthetic "image" | A kernel detecting a specific pattern (e.g. a vertical edge) in a toy grid, with output shapes verified for different stride/padding combinations |
| `02_rnn_lstm_gru_concept.ipynb` | A vanilla RNN cell from scratch, a demonstration of vanishing gradients over a long sequence, and a conceptual (not fully from-scratch) walkthrough of LSTM/GRU gating | A plot showing gradient magnitude shrinking exponentially with sequence length in a vanilla RNN |
| `03_attention_and_transformer_block.ipynb` | Scaled dot-product attention from scratch (Q/K/V, multi-head, causal masking), then a full from-scratch Transformer block (embeddings + positional encoding + multi-head attention + residual + LayerNorm + FFN) | Verified attention weight matrices summing to 1 per query; a causal mask visibly zeroing out attention to future positions; a full Transformer block's output shape matching its input shape (so blocks can stack) |

## Key takeaways

- Convolution exploits spatial locality and parameter sharing (the same small kernel is reused across the whole image) — a huge parameter-efficiency win over a fully-connected layer on image data.
- RNNs process sequentially and can't parallelize across time steps during training; attention computes all pairwise relationships in a sequence in one parallelizable step, which is a large part of why Transformers scale better.
- Residual connections and normalization aren't optional extras — very deep networks (CNNs, Transformers alike) are difficult or impossible to train well without them.
- Causal masking is what makes a decoder "autoregressive" — it can only attend to positions it has already generated or been given.

## Common mistakes

- Forgetting padding changes output spatial size, leading to shape-mismatch errors deeper in a CNN.
- Assuming LSTMs fully solve vanishing gradients — they mitigate it substantially but very long sequences are still hard.
- Implementing attention without scaling by `√d_k`, causing extremely peaked (low-entropy) attention distributions and unstable gradients.
- Forgetting the causal mask in a decoder, letting it "cheat" by attending to future tokens during training.

## Interview questions

**Conceptual**
1. Why does convolution use far fewer parameters than a fully-connected layer for the same input size?
2. Why do vanilla RNNs struggle with long-range dependencies, and how do LSTM/GRU gates help?
3. What problem does self-attention solve that made it preferable to RNNs for sequence modeling?
4. Why is positional encoding necessary for a Transformer but not for an RNN?
5. What does causal masking prevent, and why is it necessary for autoregressive generation?

**Coding**
1. Implement 2D convolution from scratch (a single kernel, no padding) and verify the output shape formula `(H - K + 1, W - K + 1)`.
2. Implement scaled dot-product attention (`softmax(QKᵗ/√d_k)V`) from scratch and verify attention weights sum to 1 per query.
3. Implement a causal mask and verify it correctly zeros out (via `-inf` before softmax) attention to future positions.

**Debugging**
1. Your from-scratch convolution's output is a different shape than expected — what are the two most likely causes (stride, padding)?
2. Your attention implementation produces `nan` losses during training — what's a likely missing step (hint: scaling)?

**Real-world engineering**
1. You're choosing between a CNN and a Transformer for a long-sequence time-series task — what tradeoffs would you weigh regarding sequence length and parallelism?
2. You're implementing a decoder-only language model — which masking strategy do you need, and why?

## Exercises

1. Implement max pooling from scratch and verify it reduces a 4x4 grid to 2x2 with a 2x2 pooling window.
2. Implement multi-head attention (multiple parallel single-head attentions, concatenated) and verify the output shape matches a single-head attention's output shape.
3. Stack two from-scratch Transformer blocks and verify the output shape is preserved end to end.

## References

- [Stanford CS231n — Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)
- [Stanford CS224n — Transformers and Self-Attention](https://web.stanford.edu/class/cs224n/)
- [Vaswani et al. (2017) — Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
- [The Illustrated Self-Attention (Raimi Karim)](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a)
