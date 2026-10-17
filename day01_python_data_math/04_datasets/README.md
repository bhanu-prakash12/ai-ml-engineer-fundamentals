# 04 — Datasets: Design Principles + Hugging Face `datasets`

> A model is only as good as its data: split correctly, prevent leakage, and use Hugging Face `datasets` to process data at scale.

| | |
|---|---|
| **Day** | 1 — Python + Data + Math |
| **Topic** | 04 — Dataset design + Hugging Face `datasets` |
| **Notebooks** | 1 (`01_dataset_design_and_hf_datasets`) |
| **Builds on** | [`03_data_libraries`](../03_data_libraries/README.md) |
| **Leads to** | [`05_ml_mathematics`](../05_ml_mathematics/README.md), Day 2 evaluation and feature engineering, PyTorch `Dataset` (Day 3) |

## 📖 Short forms used in this topic

| Term | Full form | What it is |
|---|---|---|
| **ML** | Machine Learning | Building models that learn from data. |
| **HF** | Hugging Face | Hub and libraries (`datasets`, `transformers`) for sharing data and models. |
| **API** | Application Programming Interface | The functions/classes a library exposes. |
| **LLM** | Large Language Model | Big neural language model; HF `datasets` is used to prepare its fine-tuning data. |
| **RAM** | Random-Access Memory | Main memory; streaming avoids loading everything into it. |
| **GB** | Gigabyte | Unit of data size (500GB example). |
| **NaN / null** | Not a Number / null | Marker for a missing value. |
| **z-score** | Standard score | `(x - mean) / std`; rescales a feature to mean 0, std 1. |
| **min-max** | Min-max scaling | `(x - min) / (max - min)`; rescales to `[0, 1]`. |
| **std** | Standard deviation | Typical spread of values around the mean. |
| **seed** | Random seed | Integer that fixes a random generator so shuffles/splits are reproducible. |
| **Dataset / DatasetDict** | - | HF classes: one table of examples / a dict of splits (`train`, `test`, ...). |
| **IterableDataset** | Iterable dataset | HF streaming dataset that yields examples lazily. |
| **Arrow** | Apache Arrow | Columnar memory format that backs HF datasets. |
| **One-hot** | One-hot encoding | Category becomes a vector with a single 1. |

## 🧠 In plain words

A dataset is a collection of examples (each with input features and, for supervised learning, a label). Dataset design is about making sure what you measure in testing honestly predicts real-world behaviour: split the data properly, clean it, scale it without peeking at the test set, and watch for imbalance. Hugging Face `datasets` is a library for loading and transforming such data efficiently.

## 🎒 Analogy to remember

Training is **studying**, validation is a **practice exam** you use to adjust your study plan, and the test set is the **final exam** you sit once. Leakage is **seeing the exam questions while studying** - your score looks great and means nothing.
*Where it breaks down:* a student cannot unsee the questions, but you can (and must) re-split and retrain when you find leakage.

## Learning objectives

By the end of this topic you can:
- Explain samples/features/labels and why train/validation/test splits exist.
- Identify and fix common dataset problems: missing values, outliers, duplicates, imbalance, and leakage.
- Normalize/standardize numeric features and encode categorical ones.
- Use Hugging Face `datasets`' `Dataset`/`DatasetDict`, `map`/`filter`/`select`/`shuffle`/`train_test_split`, batched processing, and streaming.

## Why it matters

Model quality is bounded by dataset quality far more often than by architecture choice. Leakage
(information from validation/test leaking into training) silently inflates reported accuracy and
is one of the most common real-world ML bugs. Hugging Face `datasets` is the standard interface
for dataset manipulation you'll reuse directly when fine-tuning LLMs later in this workspace's
`llm-fine-tuning/` curriculum.

## Prerequisites

[`03_data_libraries`](../03_data_libraries/README.md) — `datasets` is built on Arrow, covered there.

## Core concepts

- **Samples / features / labels**: a sample is one row/example; features are its inputs; a label is what you're predicting (supervised learning only).
- **Train / validation / test split**: train to fit, validation to tune/select, test to report a final, unbiased estimate — never tune based on test performance.
- **Structured vs unstructured data**: tabular/structured (rows+columns) vs text/image/audio (unstructured, needs featurization).
- **Cleaning**: missing values (drop vs impute), outliers (cap/remove/investigate), duplicates (can cause train/test leakage if not removed before splitting).
- **Imbalance**: when one class dominates — accuracy becomes misleading; needs resampling, class weights, or different metrics (see Day 2's evaluation topic).
- **Leakage**: any case where information not available at prediction time (or from the test set) influences training — e.g. normalizing using statistics computed over the *entire* dataset before splitting, or duplicate rows split across train and test.
- **Normalization/standardization**: scaling numeric features (min-max to `[0,1]`, or z-score to mean 0/std 1) — fit the scaler on train only, then apply to val/test.
- **Encoding**: turning categories into numbers (one-hot, label/ordinal encoding) since models need numeric input.

### Hugging Face `datasets`
- `load_dataset` — loads a dataset (from the Hub or local files) as Arrow-backed `Dataset`/`DatasetDict`.
- `.map()` — apply a function to every example (optionally `batched=True` for speed).
- `.filter()` — keep only examples matching a predicate.
- `.select()` — take a specific set of row indices.
- `.shuffle()` / `.train_test_split()` — randomize order / split into train+test with reproducible seeds.
- `.remove_columns()` — drop columns no longer needed after preprocessing.
- **Streaming**: `load_dataset(..., streaming=True)` yields an `IterableDataset` that never loads the whole thing into memory — the dataset-level analog of the generators from `01_python`.

```mermaid
flowchart LR
  subgraph Prep["Prepare"]
    A["Raw data"] --> B["Clean: missing values, outliers, duplicates"]
    B --> C["Split: train / validation / test"]
  end
  subgraph Fit["Leakage-safe transforms"]
    D["Fit scalers and encoders on TRAIN ONLY"] --> E["Apply to val and test"]
  end
  subgraph HF["Hugging Face datasets"]
    F["Dataset: map, filter, select, shuffle"] --> G["Batched or streaming processing"]
  end
  C --> D
  E --> F
  G --> H["Ready for a model DataLoader"]
  classDef prep fill:#DBEAFE,stroke:#2563EB,color:#1a1a1a;
  classDef safe fill:#DCFCE7,stroke:#16A34A,color:#1a1a1a;
  classDef hf fill:#FEF3C7,stroke:#D97706,color:#1a1a1a;
  classDef goal fill:#EDE9FE,stroke:#7C3AED,color:#1a1a1a;
  class A,B,C prep;
  class D,E safe;
  class F,G hf;
  class H goal;
```

## 🔍 Concepts explained

**Category chips:** 🟦 Splits · 🟥 Data problems · 🟩 Transforms · 🟪 HF `datasets`

### 🟦 Train, validation, test

| Split | Purpose | Touch it when | Pitfall |
|---|---|---|---|
| 🟦 Train | fit parameters (and fit scalers/encoders) | always | none, but it must be clean |
| 🟨 Validation | tune hyperparameters, select models | during development | over-tuning slowly overfits it |
| 🟥 Test | one final unbiased estimate | once, at the end | tuning on it destroys its value |

### 🟥 Data problems

| Problem | What it looks like | Typical fix | Why it hurts |
|---|---|---|---|
| Missing values | NaN/null | drop, or impute (mean/median/constant) | models cannot ingest NaN; dropping can bias |
| Outliers | extreme values | investigate, cap (winsorize) or remove | distort mean/scale and some models |
| Duplicates | repeated rows | `drop_duplicates` **before** splitting | copies in train and test leak |
| Imbalance | one class dominates | resampling, class weights, better metrics | accuracy misleads |
| Leakage | information from test/future in training | split first, fit transforms on train only | inflated scores that collapse in production |

> [!WARNING]
> Leakage patterns: (1) scaler/vocabulary fit on the **whole** dataset, (2) duplicates across splits, (3) a feature that encodes the label or only exists after the outcome.

### 🟩 Scaling and encoding

| Technique | Formula / idea | Use when | Pitfall |
|---|---|---|---|
| Min-max | `(x - min) / (max - min)` | bounded range wanted | sensitive to outliers |
| Standardization (z-score) | `(x - mean) / std` | most gradient-based models | needs train-only statistics |
| One-hot | one 0/1 column per category | nominal categories | many categories means wide data |
| Label / ordinal | integer per category | ordered categories, tree models | implies false order otherwise |

> [!IMPORTANT]
> Fit on train only, then apply unchanged to validation and test: `scaler.fit(train)`; `scaler.transform(val)`.

**Imbalance**
- **Definition:** classes appear in very different proportions.
- **Analogy:** a smoke alarm that never rings is "99.9% accurate" in a house that rarely burns.
- **Use when / watch out:** use precision, recall, F1, PR-AUC (Day 2); accuracy alone hides a majority-class predictor.

### 🟪 Hugging Face `datasets`

| Method | Does | Notes |
|---|---|---|
| `load_dataset` | load from Hub or local files | returns `Dataset` / `DatasetDict` |
| `.map(fn, batched=True)` | transform examples | `batched=True` passes lists of examples, far faster |
| `.filter(fn)` | keep matching rows | predicate returns bool |
| `.select(indices)` | pick rows by index | |
| `.shuffle(seed=...)` | random order | set a seed |
| `.train_test_split(test_size, seed)` | split | reproducible with seed |
| `.remove_columns` | drop columns | after deriving new ones |
| `streaming=True` | `IterableDataset` | no full download or load into RAM |

```mermaid
flowchart TD
  D["Raw data"] --> C["Deduplicate and clean"]
  C --> S["Split: train / validation / test"]
  S --> F["Fit scaler and encoder on TRAIN only"]
  F --> T["Transform train, val, test"]
  T --> H["HF Dataset: map / filter / batched / streaming"]
  X["Fit on all data before splitting"] -.->|"LEAKAGE"| S
  classDef ok fill:#DCFCE7,stroke:#16A34A,color:#1a1a1a;
  classDef step fill:#DBEAFE,stroke:#2563EB,color:#1a1a1a;
  classDef bad fill:#FEE2E2,stroke:#DC2626,color:#1a1a1a;
  class D,C,S step;
  class F,T,H ok;
  class X bad;
```

> [!TIP]
> Quick leakage check: if validation accuracy is suspiciously high, look for duplicates across splits and for statistics computed before the split.

## Important APIs / functions

`datasets.load_dataset`, `datasets.Dataset`, `datasets.DatasetDict`, `.map`, `.filter`,
`.select`, `.shuffle`, `.train_test_split`, `.remove_columns`, `batched=True`,
`streaming=True`.

## Notebooks

| Notebook | Purpose | Expected output |
|---|---|---|
| `01_dataset_design_and_hf_datasets.ipynb` | Dataset design principles (split, leakage, imbalance, scaling fit-on-train-only) using a small synthetic tabular dataset, then the same data wrapped as a Hugging Face `Dataset` demonstrating map/filter/select/shuffle/train_test_split/batched/streaming | A deliberately-leaky split shown to inflate "accuracy," contrasted with a correct split; an HF `Dataset` built from a Pandas DataFrame, transformed via `.map(batched=True)`, and split via `.train_test_split()` |

## Key takeaways

- Fit any data-dependent transform (scaler, encoder, vocabulary) on the training split only, then apply it unchanged to validation/test.
- Deduplicate *before* splitting, or duplicates can land in both train and test, leaking information.
- Class imbalance doesn't go away by ignoring it — accuracy on an imbalanced dataset can look great while the model learns to always predict the majority class.
- `datasets.Dataset` is Arrow-backed, so `.map(batched=True)` is fast and streaming works on data too large for memory — the same ideas as `03_data_libraries`.

## Common mistakes

- Normalizing/standardizing using statistics from the *whole* dataset before splitting (leakage).
- Shuffling before splitting without a fixed seed, making results non-reproducible.
- Treating accuracy as sufficient on an imbalanced dataset.
- Calling `.map()` without `batched=True` on a large dataset when the function could easily be vectorized over a batch.

## Interview questions

**Conceptual**
1. Why must you fit a scaler on the training set only, not the full dataset?
2. Give a concrete example of data leakage that wouldn't show up as a bug until deployment.
3. Why is accuracy a poor metric on an imbalanced dataset?
4. What's the difference between `Dataset.map()` with and without `batched=True`?
5. When would you use streaming mode (`streaming=True`) instead of loading a dataset fully?

**Coding**
1. Given a Pandas DataFrame, build a Hugging Face `Dataset` and use `.train_test_split(test_size=0.2, seed=42)`.
2. Write a `.map(batched=True)` function that lowercases a text column across the whole dataset.
3. Use `.filter()` to keep only rows where a `label` column is not null.

**Debugging**
1. A model gets 99% validation accuracy but performs terribly in production — what dataset issue would you check first?
2. After splitting, you realize duplicate rows exist in both train and test — what's the fix, and does it require re-splitting?

**Real-world engineering**
1. You're building a fraud-detection model where fraud is 0.1% of transactions — how does this change your evaluation strategy, independent of the model itself?
2. Your dataset is 500GB and doesn't fit in memory — how does Hugging Face `datasets`' Arrow-backed storage and streaming mode help?

## Exercises

1. Build a synthetic dataset, deliberately leak test statistics into a "normalized" feature, and measure the artificially inflated validation score vs a correct, leakage-free pipeline.
2. Convert a Pandas DataFrame to a Hugging Face `Dataset`, apply `.map(batched=True)` to add a derived column, then `.remove_columns()` the original.
3. Load a small dataset with `streaming=True` and iterate the first 5 examples without ever materializing the full dataset.

## References

- [Hugging Face `datasets` docs](https://huggingface.co/docs/datasets/index)
- [Hugging Face `datasets` — Process](https://huggingface.co/docs/datasets/process)
- [Hugging Face `datasets` — Streaming](https://huggingface.co/docs/datasets/stream)
- [scikit-learn — Data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
