# 10 — ML Optimization

## Learning objectives

By the end of this topic you can:
- Explain the relationship between a loss, an objective, and a cost function.
- Implement and compare gradient descent variants: batch, SGD, mini-batch, momentum, AdaGrad, RMSProp, Adam, AdamW.
- Explain learning-rate schedules, warmup, and decay, and why they matter in practice.

## Why it matters

Every model in this curriculum — from linear regression to the Transformers in Day 3 — is fit by
some variant of gradient descent. Knowing why Adam usually beats plain SGD, and why a learning
rate schedule matters, is the difference between a training run that converges in an hour and
one that never converges at all.

## Prerequisites

[`07_regression`](../07_regression/README.md) (gradient descent from scratch) and Day 1's
[`05_ml_mathematics`](../../day01_python_data_math/05_ml_mathematics/README.md) (the chain rule/gradient).

## Core concepts

- **Loss / objective / cost**: a loss is computed per-example; cost (or objective) is typically the average loss over a batch/dataset — terms are often used interchangeably in practice, but the per-example vs aggregate distinction matters when reasoning about gradients.
- **(Batch) Gradient descent**: compute the gradient over the *entire* dataset before each update — accurate but slow per step, and may not fit in memory.
- **SGD (Stochastic Gradient Descent)**: update using the gradient from a *single* example — noisy but fast, and the noise can help escape shallow local minima.
- **Mini-batch SGD**: the practical middle ground — update using a small batch (e.g. 32-256 examples); the de facto standard.
- **Momentum**: accumulates a moving average of past gradients, smoothing out noisy updates and accelerating through consistent directions.
- **AdaGrad**: adapts the learning rate per-parameter based on the sum of squared past gradients — shrinks the learning rate for frequently-updated parameters.
- **RMSProp**: fixes AdaGrad's ever-shrinking learning rate by using an exponential moving average of squared gradients instead of a running sum.
- **Adam**: combines momentum (first moment) and RMSProp-style adaptive per-parameter learning rates (second moment) — the most common default optimizer for deep learning.
- **AdamW**: Adam with weight decay decoupled from the gradient update (rather than added into the gradient like L2 regularization) — generally preferred over plain Adam for training large models.
- **Learning-rate schedules, warmup, decay**: starting with a small LR and ramping up (warmup) avoids early instability; decaying the LR over training helps fine-grained convergence near the end.

```mermaid
flowchart LR
  A[Full-batch GD: accurate, slow, memory-heavy] --> D[Mini-batch SGD: the practical default]
  B[Pure SGD: noisy, fast per-step] --> D
  D --> E[+ Momentum: smooths direction]
  E --> F[+ Adaptive LR: AdaGrad -> RMSProp]
  F --> G[Adam: momentum + adaptive LR]
  G --> H[AdamW: decoupled weight decay]
  H --> I[+ LR schedule: warmup then decay]
```

## Important APIs / functions

From-scratch NumPy implementations of each optimizer update rule; conceptually mirrored by
`torch.optim.SGD`, `torch.optim.Adam`, `torch.optim.AdamW`, and
`torch.optim.lr_scheduler.*` (used for real in Day 3).

## Notebooks

| Notebook | Purpose | Expected output |
|---|---|---|
| `01_optimizers_from_scratch.ipynb` | Implement batch GD, SGD, mini-batch SGD, momentum, AdaGrad, RMSProp, and Adam from scratch on the same loss surface; implement a linear warmup + decay schedule | A single plot comparing all optimizers' convergence curves on the same function; a visualization of how a learning-rate schedule changes the effective step size over training |

## Key takeaways

- Mini-batch SGD is a biased-free, lower-variance *estimate* of the full-batch gradient (connecting back to Day 1's sampling/expectation concepts) — this is why it works despite being "noisy."
- Adam's popularity comes from needing little tuning: it adapts per-parameter learning rates automatically, which matters enormously once a model has millions of parameters with very different gradient scales.
- A learning-rate schedule isn't optional polish for large models — warmup prevents early divergence, and decay is often necessary for the loss to settle near its minimum.

## Common mistakes

- Using a single fixed learning rate for the entire training run without ever checking if decay would help.
- Forgetting that increasing batch size changes the effective noise in SGD, which often means the learning rate needs retuning too.
- Confusing AdaGrad's ever-shrinking learning rate (a known limitation) with RMSProp/Adam's fix for it.
- Treating Adam as a magic "set and forget" optimizer — it still has hyperparameters (especially the learning rate) that matter.

## Interview questions

**Conceptual**
1. Why does mini-batch SGD work at all, given it uses a noisy, incomplete estimate of the true gradient?
2. What specific problem does momentum solve that plain SGD has?
3. What's AdaGrad's main weakness, and how does RMSProp fix it?
4. What does Adam combine, conceptually, from momentum and RMSProp?
5. Why is AdamW generally preferred over Adam with naive L2 regularization added to the loss?

**Coding**
1. Implement momentum-based gradient descent from scratch and compare its convergence speed to plain gradient descent on a function with a narrow, elongated minimum (e.g. a Rosenbrock-like function).
2. Implement Adam from scratch (first and second moment estimates with bias correction) and verify it converges on a simple quadratic.
3. Implement a linear warmup + cosine decay learning-rate schedule and plot the resulting learning rate over training steps.

**Debugging**
1. Your training loss oscillates wildly and occasionally spikes — what hyperparameter would you check first, and why?
2. Switching from SGD to Adam at the same learning rate causes immediate divergence — why might that happen, and what should you do?

**Real-world engineering**
1. You're training on a single GPU with limited memory — how does batch size interact with your choice of optimizer and learning rate?
2. Your model trains fine for small datasets but diverges early on a much larger one — how might a warmup schedule help, and why?

## Exercises

1. Compare batch GD, SGD, and mini-batch SGD (batch size 16) on the same quadratic loss and plot all three convergence curves together.
2. Implement AdaGrad and demonstrate its learning rate shrinking over time by plotting the effective per-parameter step size.
3. Implement a step-decay schedule (halve the LR every N steps) and compare final convergence to a constant learning rate.

## References

- [Sebastian Ruder — An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/)
- [Kingma & Ba (2014) — Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Loshchilov & Hutter (2017) — Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)
- [PyTorch docs — `torch.optim`](https://pytorch.org/docs/stable/optim.html)
