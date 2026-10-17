# 11 — Generalization

> Training error is easy; the real goal is low error on data the model has never seen.

| At a glance | |
|---|---|
| **Day** | 2 — Machine Learning |
| **Topic** | Overfitting, underfitting, bias-variance, regularization, early stopping |
| **Notebooks** | 1 (`01_regularization_and_generalization.ipynb`) |
| **Builds on** | [`10_ml_optimization`](../10_ml_optimization/README.md), [`06_ml_fundamentals`](../06_ml_fundamentals/README.md) |
| **Leads to** | [`12_ml_evaluation`](../12_ml_evaluation/README.md), [`14_hyperparameter_tuning`](../14_hyperparameter_tuning/README.md) |

## 📖 Short forms used in this topic

| Term | Full form | What it is |
|---|---|---|
| **L1** | L1 norm / Lasso penalty | Penalty on the sum of absolute weights; drives some weights to exactly zero. |
| **L2** | L2 norm / Ridge penalty | Penalty on the sum of squared weights; shrinks weights smoothly. |
| **Lasso** | Least Absolute Shrinkage and Selection Operator | Linear regression with an L1 penalty. |
| **Ridge** | Ridge regression | Linear regression with an L2 penalty. |
| **MSE** | Mean Squared Error | Average of squared prediction errors; the standard regression loss. |
| **λ / alpha** | Regularization strength | Weight of the penalty term (`alpha` in scikit-learn). |

## 🧠 In plain words

Generalization is how well a model does on new data. A model that memorizes its training set (overfitting) or is too simple to learn the pattern (underfitting) both generalize badly. Regularization, early stopping and more varied data are the tools that keep a model from memorizing noise.

## 🎒 Analogy to remember

Studying for an exam. Memorizing last year's answer key (overfitting) gets full marks on the practice paper and fails the real one; skimming only the chapter titles (underfitting) fails both. Regularization is a rule like "answers must be short", which stops you from memorizing every detail. Early stopping is putting the book down once mock-exam scores stop improving. Where it breaks: a student chooses to cram, whereas a model overfits automatically.

## Learning objectives

By the end of this topic you can:
- Formally define overfitting, underfitting, and the bias-variance tradeoff.
- Apply L1, L2, and Elastic Net regularization and explain what each does differently.
- Explain dropout and early stopping as regularization techniques, and why data augmentation helps.

## Why it matters

`06_ml_fundamentals` introduced the underfit/overfit demo informally; this topic formalizes it
and gives you the actual tools (regularization, early stopping) used to fix it in practice —
tools you'll reuse directly on neural networks in Day 3.

## Prerequisites

[`10_ml_optimization`](../10_ml_optimization/README.md); the underfit/overfit demo in
[`06_ml_fundamentals`](../06_ml_fundamentals/README.md).

## Core concepts

- **Overfitting**: the model fits training data (including its noise) so closely that it generalizes poorly to new data — low training error, high validation/test error.
- **Underfitting**: the model is too simple to capture the underlying pattern — high error on both training and validation/test data.
- **Bias-variance tradeoff**: bias = error from overly simplistic assumptions (underfitting); variance = error from sensitivity to the specific training sample (overfitting). Total expected error ≈ bias² + variance + irreducible noise — you can't drive both to zero simultaneously by adjusting capacity alone.
- **L1 regularization (Lasso)**: adds `λΣ|w|` to the loss — pushes some weights exactly to zero, giving automatic feature selection.
- **L2 regularization (Ridge)**: adds `λΣw²` to the loss — shrinks all weights smoothly toward zero without necessarily zeroing any out.
- **Elastic Net**: a weighted combination of L1 and L2 — gets some sparsity (from L1) while keeping L2's smoother shrinkage.
- **Dropout (concept)**: randomly zeroes out a fraction of neurons during training, forcing the network to not rely on any single neuron — a regularization technique specific to neural networks (hands-on in Day 3).
- **Early stopping**: stop training when validation loss stops improving, even if training loss keeps dropping — directly targets the overfitting point.
- **Data augmentation (concept)**: artificially expanding the effective training set (e.g. image flips/rotations, text paraphrasing) so the model sees more variation without needing more raw labeled data.

```mermaid
flowchart LR
  subgraph P["Problems"]
    A["Model capacity too low"] --> B["Underfitting: high bias"]
    C["Capacity too high, no regularization"] --> D["Overfitting: high variance"]
  end
  subgraph T["Remedies for variance"]
    E["L1, L2, Elastic Net"]
    G["Early stopping"]
    H["Data augmentation"]
    J["Dropout"]
  end
  E --> F["Shrink or zero weights, reduce variance"]
  G --> F
  H --> F
  J --> F
  B -.->|"opposite failure mode of"| D
  D --> E
  F --> I["Better generalization on validation and test data"]
  classDef bad fill:#FEE2E2,stroke:#DC2626,color:#1a1a1a;
  classDef warn fill:#FEF3C7,stroke:#D97706,color:#1a1a1a;
  classDef fix fill:#DBEAFE,stroke:#2563EB,color:#1a1a1a;
  classDef good fill:#DCFCE7,stroke:#16A34A,color:#1a1a1a;
  class B,D bad;
  class A,C warn;
  class E,G,H,J fix;
  class F,I good;
```

## 🔍 Concepts explained

Chips: 🟥 failure modes, 🟦 penalty-based regularizers, 🟩 training-procedure / data regularizers.

### 🟥 Underfitting vs overfitting (bias vs variance)
- **Definition**: underfit = high error on train and validation (high bias); overfit = low train error, high validation error (high variance).
- **Analogy**: skimming titles vs memorizing the answer key.
- **Formula**: expected error ≈ bias² + variance + irreducible noise.
- **Use when / watch out**: compare train vs validation error to diagnose; the notebook estimates bias² and variance over 200 resampled datasets for polynomial degrees 1, 4, 8.

### 🟦 L1 (Lasso)
- **Definition**: adds `λΣ|w|` to the loss; some weights become exactly zero.
- **Analogy**: a strict packing limit that forces you to leave items out entirely.
- **Code idea**: `sklearn.linear_model.Lasso(alpha=...)`.
- **Use when / watch out**: many features, few relevant; unstable among correlated features.

### 🟦 L2 (Ridge)
- **Definition**: adds `λΣw²`; shrinks all weights smoothly.
- **Analogy**: making every item in the bag a bit lighter instead of dropping any.
- **Code idea**: `sklearn.linear_model.Ridge(alpha=...)`.
- **Use when / watch out**: most features are somewhat useful; no feature selection.

### 🟦 Elastic Net
- **Definition**: weighted mix of L1 and L2 penalties.
- **Analogy**: a packing limit plus lighter items.
- **Code idea**: `sklearn.linear_model.ElasticNet(alpha=..., l1_ratio=...)`.
- **Use when / watch out**: correlated features where you still want sparsity; two knobs to tune.

### 🟩 Early stopping
- **Definition**: stop iterative training when validation loss has not improved for a patience window.
- **Analogy**: closing the textbook when mock scores plateau.
- **Code idea**: track train and validation loss each epoch; keep the best.
- **Use when / watch out**: watch validation loss, not training loss; too small a patience stops too early.

### 🟩 Dropout and data augmentation (concepts)
- **Definition**: dropout randomly zeroes neurons during training; augmentation creates varied copies of training data.
- **Analogy**: practicing with random teammates absent; practicing the same play from many angles.
- **Use when / watch out**: neural networks (Day 3); augmentation must preserve the label.

### Differences at a glance

| Technique | Chip | Changes loss? | Effect on weights | Best when | Pitfall |
|---|---|---|---|---|---|
| L1 / Lasso | 🟦 | Yes | Exact zeros | Few relevant features | Picks arbitrarily among correlated features |
| L2 / Ridge | 🟦 | Yes | Smooth shrinkage | All features matter a bit | Keeps every feature |
| Elastic Net | 🟦 | Yes | Sparse and smooth | Correlated features | Two hyperparameters |
| Early stopping | 🟩 | No | Implicit limit | Iterative fits | Needs a validation set |
| Dropout | 🟩 | No | Reduces co-adaptation | Neural nets | Not for plain linear models |
| Augmentation | 🟩 | No | Indirect | Images, text | Label-changing transforms |

> [!NOTE]
> You cannot drive bias and variance to zero together by changing capacity; regularization trades a little bias for lower variance.

> [!WARNING]
> Choosing `alpha` on the test set leaks information. Tune it on a validation set (see `14_hyperparameter_tuning`).

> [!TIP]
> If raising L2 strength makes both train and validation error worse, you were already underfitting.

## Important APIs / functions

`sklearn.linear_model.Ridge`, `sklearn.linear_model.Lasso`, `sklearn.linear_model.ElasticNet`,
manually tracking train vs validation loss per epoch for early stopping.

## Notebooks

| Notebook | Purpose | Expected output |
|---|---|---|
| `01_regularization_and_generalization.ipynb` | Formal bias-variance demo (repeated resampling to show the tradeoff numerically); L1 vs L2 vs Elastic Net on the same overfitting-prone dataset, showing L1's sparsity vs L2's shrinkage; an early-stopping demo on an iteratively-fit model | A plot showing L1 driving several coefficients exactly to zero while L2 shrinks all of them smoothly; a train-vs-validation loss curve with the early-stopping point marked |

## Key takeaways

- Bias and variance trade off against each other as capacity changes — regularization explicitly buys down variance at the cost of a little bias.
- L1's sparsity (exact zeros) makes it useful for feature selection; L2's smooth shrinkage is usually better when all features are at least somewhat relevant.
- Early stopping is "free" regularization — it costs nothing but a validation set and tracking a metric per epoch.

## Common mistakes

- Choosing the regularization strength (`λ`/`alpha`) without a validation set — this itself needs tuning (previewed here, formalized in `14_hyperparameter_tuning`).
- Assuming L1 and L2 are interchangeable — they produce qualitatively different weight patterns (sparse vs smoothly shrunk).
- Stopping training the moment training loss plateaus, rather than watching validation loss (training loss plateauing is normal and doesn't imply overfitting has started).

## Interview questions

**Conceptual**
1. State the bias-variance tradeoff in your own words and connect it to under/overfitting.
2. Why does L1 regularization produce exactly-zero weights while L2 doesn't?
3. Why is early stopping considered a regularization technique, even though it doesn't change the loss function?
4. How does data augmentation reduce overfitting without collecting more labeled data?
5. Why can't you simultaneously minimize both bias and variance just by tuning model capacity?

**Coding**
1. Fit `Ridge`, `Lasso`, and `ElasticNet` on the same dataset with increasing regularization strength and plot how coefficients shrink/zero out.
2. Implement early stopping manually: track validation loss per epoch and stop when it hasn't improved for N epochs.
3. Numerically demonstrate the bias-variance tradeoff by fitting many models on different resamples of the same data-generating process and decomposing the average error.

**Debugging**
1. Increasing L2 regularization strength makes both training AND validation error worse — what does this suggest about your starting regularization strength?
2. Your early-stopping criterion triggers after just 2 epochs, far too early — what hyperparameter of the early-stopping logic itself might need adjusting?

**Real-world engineering**
1. You have 1,000 features but suspect only ~20 are actually relevant — would you reach for L1 or L2, and why?
2. Your model overfits badly and you can't collect more data — list three techniques from this topic you'd try, in order.

## Exercises

1. Generate an overfitting-prone dataset (many features, few samples) and compare unregularized, L1, L2, and Elastic Net linear models' test performance.
2. Plot coefficient magnitude vs regularization strength (`alpha`) for Ridge and Lasso side by side.
3. Simulate the bias-variance tradeoff: for polynomial models of increasing degree, repeatedly resample the training data, fit, and plot average bias² and variance separately.

## References

- [scikit-learn — Linear Models: Ridge, Lasso, Elastic Net](https://scikit-learn.org/stable/modules/linear_model.html)
- [Stanford CS229 — Bias-Variance notes](https://cs229.stanford.edu/notes2022fall/)
- [Srivastava et al. (2014) — Dropout](https://jmlr.org/papers/v15/srivastava14a.html)
