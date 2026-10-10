# Build Report — Full Curriculum (Days 1-3 + Capstone)

Scope: this report covers the **complete** 3-day AI/ML Engineer Fundamentals curriculum —
Day 1 (Python+Data+Math), Day 2 (Machine Learning), Day 3 (Deep Learning+PyTorch), and the
capstone project — built across three incremental sessions, each created → executed →
validated → fixed before moving on. **The curriculum is now complete: 38/38 notebooks pass.**

## Files created (cumulative)

```
ai-ml-engineer-fundamentals/
├── README.md, ROADMAP.md, STUDY_PLAN.md
├── requirements.txt, requirements-colab.txt
├── day01_python_data_math/
│   ├── 01_python/            README.md + 3 notebooks
│   ├── 02_numpy/             README.md + 1 notebook
│   ├── 03_data_libraries/    README.md + 3 notebooks + generate_dataset.py + data/
│   ├── 04_datasets/          README.md + 1 notebook
│   └── 05_ml_mathematics/    README.md + 3 notebooks
├── day02_machine_learning/
│   ├── README.md             (topic index)
│   ├── 06_ml_fundamentals/        README.md + 1 notebook
│   ├── 07_regression/             README.md + 2 notebooks
│   ├── 08_classification/         README.md + 2 notebooks
│   ├── 09_unsupervised_learning/  README.md + 2 notebooks
│   ├── 10_ml_optimization/        README.md + 1 notebook
│   ├── 11_generalization/         README.md + 1 notebook
│   ├── 12_ml_evaluation/          README.md + 3 notebooks
│   ├── 13_feature_engineering/    README.md + 1 notebook
│   └── 14_hyperparameter_tuning/  README.md + 1 notebook
├── day03_deep_learning/
│   ├── README.md             (topic index)
│   ├── 15_deep_learning/              README.md + 1 notebook
│   ├── 16_backpropagation/            README.md + 1 notebook
│   ├── 17_deep_learning_architectures/ README.md + 3 notebooks
│   ├── 18_pytorch_basics/             README.md + 1 notebook
│   ├── 19_pytorch_autograd/           README.md + 1 notebook
│   ├── 20_pytorch_models/             README.md + 1 notebook
│   ├── 21_pytorch_data/               README.md + 1 notebook
│   ├── 22_pytorch_training/           README.md + 1 notebook
│   ├── 23_pytorch_performance/        README.md + 1 notebook
│   └── 24_huggingface_foundations/    README.md + 1 notebook
├── projects/project_end_to_end_ai_foundation/  README.md + 1 notebook
├── scripts/validate_notebooks.py (multi-day + capstone aware), scripts/run_notebooks.py
└── reports/notebook_validation.md
```

## Notebooks created

**Day 1: 11 notebooks**

| Topic | Notebook | What it covers |
|---|---|---|
| 01_python | `01_core_language.ipynb` | collections, comprehensions, functions, OOP, dataclasses, magic methods |
| 01_python | `02_generators_decorators_contextmanagers.ipynb` | generators/`yield`, decorators, context managers, exceptions, typing |
| 01_python | `03_stdlib_and_concurrency.ipynb` | json/csv/pathlib/regex/logging, asyncio vs threading vs multiprocessing, the GIL |
| 02_numpy | `01_numpy_fundamentals.ipynb` | ndarray, shape/dtype, indexing/broadcasting, vectorization, dot/matmul, tensor-shape reasoning |
| 03_data_libraries | `01_pandas.ipynb` | Series/DataFrame, cleaning, groupby, merge/concat, pivot/melt |
| 03_data_libraries | `02_polars.ipynb` | expression API, lazy execution, joins, aggregation |
| 03_data_libraries | `03_pyarrow.ipynb` | Arrow columnar format, Parquet round-trip, zero-copy interop |
| 04_datasets | `01_dataset_design_and_hf_datasets.ipynb` | splits, leakage, imbalance, Hugging Face `datasets` (map/filter/select/shuffle/train_test_split/streaming) |
| 05_ml_mathematics | `01_linear_algebra.ipynb` | dot product, matmul, norms, eigendecomposition, PCA-by-hand |
| 05_ml_mathematics | `02_probability_statistics.ipynb` | Bayes' theorem, distributions, expected value/variance/covariance/correlation, sampling |
| 05_ml_mathematics | `03_calculus_and_gradient_descent.ipynb` | derivative/gradient/chain rule, gradient descent from scratch (quadratic + MSE regression) |

**Day 2: 14 notebooks**

| Topic | Notebook | What it covers |
|---|---|---|
| 06_ml_fundamentals | `01_ml_fundamentals.ipynb` | paradigms (supervised/unsupervised/semi/self-supervised/RL), parameters vs hyperparameters, training vs inference, underfit/good-fit/overfit |
| 07_regression | `01_linear_regression_numpy.ipynb` | closed-form least squares + gradient descent from scratch, MSE/MAE/RMSE/R² |
| 07_regression | `02_linear_regression_sklearn.ipynb` | same data via scikit-learn, residual analysis |
| 08_classification | `01_classification_fundamentals.ipynb` | binary/multiclass/multilabel, logistic regression, k-NN, decision tree, random forest, gradient boosting, SVM, Naive Bayes, decision boundaries |
| 08_classification | `02_xgboost_lightgbm.ipynb` | XGBoost/LightGBM vs sklearn's GradientBoosting, with graceful environment-dependent fallback |
| 09_unsupervised_learning | `01_clustering.ipynb` | K-means, hierarchical/dendrogram, DBSCAN, K-means failure case vs DBSCAN success |
| 09_unsupervised_learning | `02_dimensionality_reduction.ipynb` | PCA/SVD, t-SNE, UMAP on the `digits` dataset |
| 10_ml_optimization | `01_optimizers_from_scratch.ipynb` | batch GD, SGD, momentum, AdaGrad, RMSProp, Adam, warmup+cosine-decay schedule, all from scratch |
| 11_generalization | `01_regularization_and_generalization.ipynb` | numeric bias-variance decomposition, L1/L2/Elastic Net, manual early stopping |
| 12_ml_evaluation | `01_classification_metrics.ipynb` | confusion matrix, precision/recall/F1/specificity, ROC-AUC vs PR-AUC under imbalance |
| 12_ml_evaluation | `02_ranking_metrics.ipynb` | Precision@K, Recall@K, MRR, MAP, NDCG from scratch |
| 12_ml_evaluation | `03_cross_validation.ipynb` | single split vs K-fold variance, stratified vs plain K-fold, leakage demo + `Pipeline` fix |
| 13_feature_engineering | `01_feature_engineering_pipeline.ipynb` | `ColumnTransformer` + `Pipeline` (encoding/scaling/imputation/outliers/selection), leakage-safe cross-validation |
| 14_hyperparameter_tuning | `01_hyperparameter_search.ipynb` | `GridSearchCV` vs `RandomizedSearchCV`, a simplified Gaussian-process Bayesian optimization demo, rigorous score comparison |

**Day 3: 12 notebooks**

| Topic | Notebook | What it covers |
|---|---|---|
| 15_deep_learning | `01_perceptron_and_mlp_forward_pass.ipynb` | perceptron, 2-layer MLP forward pass, sigmoid/tanh/ReLU/Leaky ReLU/GELU/softmax, MSE/BCE/cross-entropy |
| 16_backpropagation | `01_backprop_by_hand.ipynb` | tiny manual chain-rule example, full MLP backward pass from scratch, numerical gradient checking, one training step |
| 17_deep_learning_architectures | `01_cnn_from_scratch.ipynb` | 2D convolution, stride/padding, edge detection, max pooling |
| 17_deep_learning_architectures | `02_rnn_lstm_gru_concept.ipynb` | vanilla RNN cell, vanishing gradients demonstrated, LSTM gating forward pass, GRU concept |
| 17_deep_learning_architectures | `03_attention_and_transformer_block.ipynb` | scaled dot-product attention, multi-head attention, causal masking, full Transformer block |
| 18_pytorch_basics | `01_tensors_devices_shapes.ipynb` | tensor creation/dtype/device, NumPy↔PyTorch conversion, device-agnostic code, `.view()` vs `.reshape()` |
| 19_pytorch_autograd | `01_autograd_basics.ipynb` | `.backward()` verified against `16_backpropagation`'s hand-derived gradients, `.grad` accumulation gotcha, `torch.no_grad()` |
| 20_pytorch_models | `01_nn_module_basics.ipynb` | `nn.Module`, `nn.Linear`/`Embedding`/`LayerNorm`/`Dropout`, a Transformer block via `nn.MultiheadAttention` |
| 21_pytorch_data | `01_dataset_and_dataloader.ipynb` | custom `Dataset`, `DataLoader` batching/shuffling, custom `collate_fn`, `IterableDataset`, `num_workers` |
| 22_pytorch_training | `01_full_training_pipeline.ipynb` | full training+validation loop, LR scheduler, checkpointing on best val score, early stopping, reproducibility |
| 23_pytorch_performance | `01_performance_techniques.ipynb` | FP32/FP16/BF16, `torch.autocast`, gradient accumulation, gradient clipping, data-loading profiling |
| 24_huggingface_foundations | `01_tokenizers_and_generation.ipynb` | `AutoTokenizer`/`AutoModelForCausalLM`, attention masks, `generate()`, model config, `.map()` preprocessing, LoRA parameter-count concept |

**Capstone: 1 notebook**

| Topic | Notebook | What it covers |
|---|---|---|
| project_end_to_end_ai_foundation | `01_end_to_end_pipeline.ipynb` | full pipeline on `load_breast_cancer`: Pandas/NumPy → leakage-safe split → train-only scaling → PyTorch `Dataset`/`DataLoader` → `nn.Module` MLP → training+checkpointing → precision/recall/F1/ROC-AUC evaluation → inference |

## Datasets used (Day 3 + Capstone)

All synthetic (NumPy-generated) or tiny sklearn/Hugging Face built-ins: `load_breast_cancer`
(569×30, capstone), `sshleifer/tiny-gpt2` (a ~500KB, 102K-parameter random-weight test model
from the Hugging Face Hub, used in `24_huggingface_foundations` purely to exercise tokenizer/
model/generate() mechanics — its output is intentionally nonsensical since its weights are
random). No large downloads anywhere in Day 3 or the capstone.

## Datasets used (Day 2)

All synthetic (`sklearn.datasets.make_classification`/`make_blobs`/`make_moons`) or small
sklearn built-ins (`load_digits`, 1797×64, ~0.5MB in memory, no download). No external network
dependency anywhere in Day 2 (unlike Day 1's optional Hub dataset load).

## Dependencies added (Day 1 recap)

Installed into `.venv312`: `nbclient`, `polars`, `pyarrow`, `scikit-learn`, `matplotlib`,
`datasets`. Recorded in `requirements.txt` / `requirements-colab.txt`.

## Dependencies added (Day 2)

Installed into `.venv312`: `umap-learn` (installed cleanly). Attempted `xgboost` and `lightgbm`
— both installed via `pip` but fail to **import** at runtime because their native libraries
require the `libomp` OpenMP runtime, which is not present on this machine. A `brew install
libomp` was started but did not complete in this session (see Known Limitations). The
`08_classification/02_xgboost_lightgbm.ipynb` notebook handles this with a broad
`except Exception` guard around each import/fit, printing a clear skip message and still
validating successfully end-to-end — on Colab (which ships OpenMP) or after `libomp` finishes
installing, both cells will run for real instead of skipping.

## Dependencies added (Day 3 + Capstone)

`torch`, `transformers`, and `datasets` were already installed from prior sessions (`.venv312`
came with `torch` 2.14.1 from the start; `transformers` 4.57.2 and `datasets` were already
present or installed in Day 1). No new packages were required for Day 3 or the capstone beyond
what Days 1-2 already installed.

## Validation results

`scripts/validate_notebooks.py` was generalized further to also discover notebooks under
`projects/` (previously only `day0*` directories) and re-run across **the entire curriculum**.
**All 38 notebooks passed** (11 Day 1 + 14 Day 2 + 12 Day 3 + 1 capstone) — see
`reports/notebook_validation.md`.

## Failures encountered and fixed during the Day 1 build (recap)

1. **`asyncio.run()` inside a Jupyter kernel** — fixed with top-level `await`.
2. **`multiprocessing` pickling failure under macOS's `spawn` start method** — fixed by writing
   the worker function to an importable module file on disk.
3. **Complex eigenvalues from `np.linalg.eig` on a symmetric matrix** — fixed by switching to
   `np.linalg.eigh`.

## Failures encountered and fixed during the Day 2 build

1. **`make_classification` infeasible parameter combination** (`08_classification`) —
   `n_informative + n_redundant` exceeded `n_features` for the multiclass example. Fixed by
   explicitly setting `n_redundant=0`.
2. **XGBoost's native-library load failure not caught by `except (ImportError, OSError)`**
   (`08_classification/02_xgboost_lightgbm.ipynb`) — XGBoost raises its own `XGBoostError`
   subclass when `libomp` is missing, not a standard `ImportError`/`OSError`. Fixed by widening
   to `except Exception`, with an inline comment explaining why the broad catch is intentional
   here.
3. **Complex eigenvalues / numerical instability in the bias-variance demo**
   (`11_generalization/01_regularization_and_generalization.ipynb`) — raw (unscaled) high-degree
   polynomial features caused numerically unstable closed-form fits, making "bias" blow up
   non-monotonically instead of cleanly illustrating the tradeoff. Fixed by standardizing
   polynomial features before fitting and reducing the highest tested degree from 15 to 8 (still
   clearly overfit, but numerically stable).
4. **Early-stopping demo never actually triggered early stopping**
   (`11_generalization/01_regularization_and_generalization.ipynb`) — the original setup (20
   features, 140 training samples, convex MSE loss) had no real capacity-vs-data mismatch, so
   validation loss never turned upward within the loop and the "stopped before the end" assertion
   failed. Fixed by moving the setup much closer to the interpolation threshold (28 training
   samples vs 20 features, heavier label noise), which produces genuine overfitting and a visible
   early-stopping point.

## Failures encountered and fixed during the Day 3 + Capstone build

1. **`torch.linalg.eig`-style complex-eigenvalue issue reappeared as an RNN gradient-explosion
   bug** (`17_deep_learning_architectures/02_rnn_lstm_gru_concept.ipynb`) — the vanishing-
   gradient demo used the default (Frobenius) matrix norm to check whether `W_hh`'s effective
   scale was below 1, but Frobenius norm can overstate the true growth factor by up to
   `sqrt(rank)` relative to the spectral norm that actually governs repeated multiplication —
   so the gradient *exploded* instead of vanishing. Fixed by using the spectral norm
   (`ord=2`) for both rescaling `W_hh` and measuring the per-step Jacobian.
2. **`num_workers` timing assertion was environment-fragile**
   (`21_pytorch_data/01_dataset_and_dataloader.ipynb`) — each worker process re-imports `torch`
   from scratch, and in this sandboxed environment that fixed startup cost outweighed the
   parallelism benefit for a small artificial workload, making `num_workers=2` *slower* than
   `num_workers=0`. Rather than force a misleading "speedup" via an arbitrarily large workload,
   the notebook now reports whichever outcome actually occurs and explains both possibilities,
   with the correctness assertion (every sample loaded exactly once, regardless of
   `num_workers`) kept as the real pass/fail check instead of a timing comparison.
3. **GPT-2 tokenizer has no default pad token** (`24_huggingface_foundations/01_tokenizers_and_generation.ipynb`)
   — batched padding raised `ValueError: Asking to pad but the tokenizer does not have a
   padding token`. Fixed with the standard GPT-2-family convention: `tokenizer.pad_token =
   tokenizer.eos_token` (safe because the attention mask excludes padding positions from
   attention regardless).

Each fix was applied and the affected notebook re-validated before moving to the next topic.

## Known limitations

- **XGBoost and LightGBM do not actually run in this local environment** because of the missing
  `libomp` OpenMP runtime. A `brew install libomp` was attempted twice across the Day 2 and
  Day 3 sessions and never completed — its sandboxed extraction step appeared to hang
  indefinitely in this environment, and the stuck processes were eventually killed rather than
  left running indefinitely. The relevant notebook cells (`08_classification/02_xgboost_lightgbm.ipynb`)
  skip gracefully with a clear explanatory message rather than failing, and the code is correct
  and will run on Colab (which ships OpenMP) or on a machine with `libomp` successfully
  installed — but their actual output in this repo's committed notebook is the "skipped" path,
  not a real XGBoost/LightGBM run.
- Validation was run locally on one machine/Python version (3.12.5), with PyTorch running on
  CPU (no CUDA available; MPS — Apple Silicon's GPU backend — is available but the notebooks
  are written device-agnostically and were validated on whichever device `get_device()`
  selects, not forced onto MPS specifically). Colab compatibility is designed-in (small data,
  deterministic seeds, install cells, no hard CUDA requirement) but not separately executed on
  Colab's infrastructure in this session.
- `24_huggingface_foundations` uses `sshleifer/tiny-gpt2`, a random-weight test model, so its
  `generate()` output is intentionally nonsensical — this is called out explicitly in the
  notebook; swapping in a real small model (e.g. `distilgpt2`) would require a larger (~350MB)
  download.
- (Carried over from Day 1) The `rotten_tomatoes` Hub dataset load in `04_datasets` depends on
  network access at notebook run time; a synthetic fallback exists but wasn't forced/tested in
  isolation.

## Curriculum status: complete

All 3 days and the capstone are built and validated. The natural next step is this workspace's
`llm-fine-tuning/` curriculum (Transformers → LLMs → RAG → fine-tuning → inference
engineering), which `24_huggingface_foundations` and the capstone both explicitly hand off to.
