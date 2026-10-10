# Study Plan

A suggested pacing for Day 1 (roughly 6-8 focused hours, splittable across more days if needed).

| Time | Topic | Goal |
|---|---|---|
| ~90 min | `01_python` | Be fluent in the language features that show up constantly in ML code: comprehensions, generators, decorators, dataclasses, context managers, typing. |
| ~60 min | `02_numpy` | Think in shapes. Be able to predict the output shape of any broadcast/reduction/matmul before running it. |
| ~90 min | `03_data_libraries` | Load, clean, and reshape tabular data three ways (Pandas for ergonomics, Polars for speed/lazy execution, PyArrow for columnar interchange). |
| ~60 min | `04_datasets` | Reason about train/val/test splits, leakage, and imbalance; use Hugging Face `datasets` for batched/streaming processing. |
| ~90 min | `05_ml_mathematics` | Connect derivative → gradient → gradient descent by hand, and know the linear-algebra/probability vocabulary used in every ML paper. |

## How to work through each topic

1. Read the topic `README.md` first (objectives, diagrams, common mistakes).
2. Open the notebook(s) in Colab or locally; run every cell, read the interpretation under each output.
3. Do the exercises at the end before looking at any hints.
4. Answer the interview questions out loud, then check your answer against the README.
5. Move to the next topic only once `scripts/validate_notebooks.py` passes for the current one.

## Day 2 pacing (roughly 8-10 focused hours)

| Time | Topic | Goal |
|---|---|---|
| ~45 min | `06_ml_fundamentals` | Nail the vocabulary: paradigms, parameters vs hyperparameters, training vs inference, capacity. |
| ~60 min | `07_regression` | Derive and implement linear regression two ways (closed-form, gradient descent) and connect MSE/MAE/RMSE/R². |
| ~90 min | `08_classification` | Fit and compare 7+ classifiers; understand *when* to reach for each one, not just how to call `.fit()`. |
| ~75 min | `09_unsupervised_learning` | See K-means fail where DBSCAN succeeds; use PCA safely vs t-SNE for visualization only. |
| ~75 min | `10_ml_optimization` | Implement every optimizer variant from scratch; understand why Adam is the default. |
| ~60 min | `11_generalization` | Numerically see the bias-variance tradeoff; apply L1/L2/Elastic Net and early stopping. |
| ~90 min | `12_ml_evaluation` | Internalize why accuracy lies under imbalance; use ranking metrics and leakage-safe cross-validation. |
| ~60 min | `13_feature_engineering` | Build a `ColumnTransformer` + `Pipeline` that's leakage-safe by construction. |
| ~60 min | `14_hyperparameter_tuning` | Compare grid vs random search vs a taste of Bayesian optimization, rigorously. |

## Day 3 pacing (roughly 9-11 focused hours)

| Time | Topic | Goal |
|---|---|---|
| ~60 min | `15_deep_learning` | Implement a forward pass, every activation function, and MSE/BCE/cross-entropy from scratch. |
| ~75 min | `16_backpropagation` | Derive gradients by hand for a tiny example, then a full 2-layer MLP, verified numerically. |
| ~120 min | `17_deep_learning_architectures` | Implement convolution, an RNN cell (and see vanishing gradients), and attention/a Transformer block, all from scratch. |
| ~45 min | `18_pytorch_basics` | Port Day 1's NumPy shape-reasoning to PyTorch tensors, devices, and dtypes. |
| ~45 min | `19_pytorch_autograd` | Verify `.backward()` against `16_backpropagation`'s hand-derived gradients. |
| ~45 min | `20_pytorch_models` | Rebuild the MLP and the Transformer block as proper `nn.Module`s. |
| ~60 min | `21_pytorch_data` | Build a `Dataset`/`DataLoader`, a custom `collate_fn`, and understand `num_workers`. |
| ~60 min | `22_pytorch_training` | Assemble the complete pipeline: training, validation, scheduling, checkpointing, early stopping. |
| ~60 min | `23_pytorch_performance` | Understand mixed precision, gradient accumulation/clipping, and profiling before optimizing. |
| ~45 min | `24_huggingface_foundations` | Get fluent with tokenizers, `AutoModel`, `generate()`, and the LoRA concept. |

## Capstone pacing (roughly 60-90 min)

Run `projects/project_end_to_end_ai_foundation/01_end_to_end_pipeline.ipynb` top to bottom —
it's a synthesis exercise, not new material. If any step feels unfamiliar, that's a signal to
revisit the specific Day 1/2/3 topic it came from (the notebook links back to each one).

## After the capstone

Continue to this workspace's `llm-fine-tuning/` curriculum: Transformers → LLMs → RAG →
fine-tuning → inference engineering — all built directly on the foundations from these three
days.
