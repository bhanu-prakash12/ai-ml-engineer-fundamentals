# 22 — PyTorch Training: The Complete Pipeline

## Learning objectives

By the end of this topic you can:
- Build the complete pipeline: `Dataset` → `DataLoader` → model → forward → loss → backward → optimizer → scheduler → validation → checkpoint.
- Implement a training loop and a validation loop correctly (including `model.train()`/`model.eval()` and `torch.no_grad()`).
- Save and load checkpoints, implement early stopping, and set seeds for reproducibility.

## Why it matters

This topic assembles every piece from `15` through `21` into the thing you'll actually run:
a real training loop. From here, training any model in this workspace's `llm-fine-tuning/`
curriculum is the same pipeline at a larger scale, not a new concept.

## Prerequisites

Every prior Day 3 topic — this is the integration point for
[`19_pytorch_autograd`](../19_pytorch_autograd/README.md),
[`20_pytorch_models`](../20_pytorch_models/README.md), and
[`21_pytorch_data`](../21_pytorch_data/README.md), plus Day 2's
[`10_ml_optimization`](../../day02_machine_learning/10_ml_optimization/README.md) (optimizers)
and [`11_generalization`](../../day02_machine_learning/11_generalization/README.md) (early
stopping).

## Core concepts

```text
Dataset
→ DataLoader
→ Model
→ Forward
→ Loss
→ Backward
→ Optimizer
→ Scheduler
→ Validation
→ Checkpoint
```

- **Training loop**: for each epoch, for each batch from the `DataLoader`: zero gradients, forward pass, compute loss, `.backward()`, optimizer step.
- **Validation loop**: after each epoch (or periodically), evaluate on held-out data with `model.eval()` and `torch.no_grad()` — no gradient tracking, no dropout.
- **Checkpointing**: saving `model.state_dict()` (and optimizer state, epoch number) periodically or whenever validation improves, so training can resume or the best model can be reloaded.
- **Early stopping**: recap from Day 2's `11_generalization`, now applied with real PyTorch validation loss tracking.
- **Reproducibility**: setting `torch.manual_seed()` (and Python's/NumPy's seeds) makes runs comparable — critical for debugging and for fair comparisons between hyperparameter settings (Day 2's `14_hyperparameter_tuning`).
- **Scheduler**: a learning-rate schedule (recap from Day 2's `10_ml_optimization`) applied via `torch.optim.lr_scheduler`, stepped once per epoch or per batch depending on the schedule.

```mermaid
flowchart TD
  A[Dataset] --> B[DataLoader]
  B --> C[for each epoch]
  C --> D[for each batch]
  D --> E[optimizer.zero_grad]
  E --> F[forward pass]
  F --> G[compute loss]
  G --> H[loss.backward]
  H --> I[optimizer.step]
  I --> D
  D -->|epoch done| J[scheduler.step]
  J --> K[validation: model.eval + no_grad]
  K --> L{improved?}
  L -->|yes| M[save checkpoint]
  L -->|no, patience exceeded| N[early stop]
  L -->|no, patience remains| C
```

## Important APIs / functions

`optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`,
`torch.optim.lr_scheduler.*`, `model.train()`/`model.eval()`, `torch.save`/`torch.load`,
`model.state_dict()`/`load_state_dict()`, `torch.manual_seed`.

## Notebooks

| Notebook | Purpose | Expected output |
|---|---|---|
| `01_full_training_pipeline.ipynb` | The complete pipeline on the `SyntheticClassificationDataset` from `21_pytorch_data` and the `TwoLayerMLP` from `20_pytorch_models`: training loop, validation loop, an LR scheduler, checkpointing (save best model), early stopping, and a reproducibility check (same seed → same results) | A training/validation loss curve showing convergence; the correct checkpoint reloaded and verified to reproduce the best validation score; two runs with the same seed producing identical results |

## Key takeaways

- Every piece in this pipeline was already built in isolation across `15`-`21` — this topic is integration, not new math.
- `model.eval()` + `torch.no_grad()` together are what make validation correct and efficient — forgetting either is a common bug (dropout active during validation, or wasted memory tracking unneeded gradients).
- Checkpointing on best validation score (not best training score, and not just the last epoch) is what lets you recover the actual best model after training completes or overfits.

## Common mistakes

- Forgetting `optimizer.zero_grad()` at the start of each batch, silently accumulating gradients (the exact bug from `19_pytorch_autograd`, now in a full loop).
- Computing validation loss without `model.eval()`, leaving dropout active and producing noisy, misleading validation numbers.
- Saving only the final model instead of checkpointing on best validation performance, missing the best point before overfitting set in.
- Not setting seeds, making two "identical" runs produce different results and debugging much harder.

## Interview questions

**Conceptual**
1. Walk through the complete training step, from `optimizer.zero_grad()` to `optimizer.step()`.
2. Why must `model.eval()` and `torch.no_grad()` both be used for validation, and what does each one specifically prevent?
3. Why checkpoint on best validation score rather than just saving the final epoch's model?
4. What's the difference between stepping a scheduler once per epoch vs once per batch, and why does it matter which you choose?
5. Why does setting a random seed matter for comparing two training runs?

**Coding**
1. Implement a full training loop (train + validate per epoch) for the `TwoLayerMLP` on `SyntheticClassificationDataset`.
2. Add checkpointing that saves the model only when validation loss improves, and reload the best checkpoint at the end.
3. Implement early stopping (recap from Day 2's `11_generalization`) integrated into this training loop, using validation loss as the signal.

**Debugging**
1. Your training loss decreases smoothly but validation loss is identical to training loss every epoch — what mode-related bug would you check first?
2. Two runs with the same seed produce different results — what's commonly still a source of non-determinism even with seeds set (hint: data loading order, certain GPU operations)?

**Real-world engineering**
1. You're training a model that takes 10 hours — what checkpointing strategy would let you resume training after a crash without losing progress?
2. Your validation loss has been flat for 10 epochs while training loss keeps dropping — what would you do, concretely, using tools from this topic?

## Exercises

1. Build the full pipeline end to end and train for enough epochs to see validation loss plateau, with early stopping triggering correctly.
2. Add a learning-rate scheduler (e.g. `torch.optim.lr_scheduler.StepLR`) and plot the learning rate alongside the loss curve.
3. Save a checkpoint, intentionally modify the model's weights, reload the checkpoint, and verify the original weights are restored exactly.

## References

- [PyTorch tutorials — Optimization loop](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [PyTorch docs — Saving and loading models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [PyTorch docs — `torch.optim.lr_scheduler`](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
