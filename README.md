# AI/ML Engineer Fundamentals — 3-Day Bootcamp

A connected, hands-on path through the fundamentals a software engineer needs before tackling
LLMs, RAG, fine-tuning, and inference engineering (see the companion
[`llm-fine-tuning/`](../llm-fine-tuning/fine_tuning_roadmap.md) curriculum in this workspace, which assumes
everything taught here).

> Understand concepts → implement them → connect them → explain them in interviews.

## Status

| Day | Topic | Status |
|---|---|---|
| 1 | Python + Data + Mathematics | ✅ Complete |
| 2 | Machine Learning | ✅ Complete |
| 3 | Deep Learning + PyTorch | ✅ Complete (this build) |
| — | Capstone project | ✅ Complete (this build) |

Built incrementally: each day is created → executed → validated → fixed before moving to the next
(see `reports/notebook_validation.md` for proof every notebook — all 3 days plus the capstone —
actually runs end to end).

## How to use this repo

**Option A — Google Colab (recommended, zero setup):** open any `.ipynb` directly in Colab. The
first cell of each notebook installs anything Colab doesn't already ship (see
`requirements-colab.txt`).

**Option B — Local:**

```bash
cd ai-ml-engineer-fundamentals
python -m venv .venv && source .venv/bin/activate   # or reuse an existing venv
pip install -r requirements.txt
jupyter lab
```

## Structure

```text
ai-ml-engineer-fundamentals/
├── day01_python_data_math/   # Python, NumPy, Pandas/Polars/PyArrow, Datasets, ML Math
├── day02_machine_learning/   # ML fundamentals through hyperparameter tuning
├── day03_deep_learning/      # Deep learning, backprop, architectures, PyTorch, Hugging Face
├── projects/                 # end-to-end capstone pipeline
├── scripts/                  # validate_notebooks.py, run_notebooks.py
├── reports/                  # notebook_validation.md
└── references/
```

Each topic folder contains a `README.md` (objectives, concepts, diagrams, interview questions,
exercises) and one or more Colab-ready `.ipynb` notebooks.

## Day 1 topics

1. [`01_python`](day01_python_data_math/01_python/README.md) — core language, OOP, iterators/generators, decorators, concurrency basics
2. [`02_numpy`](day01_python_data_math/02_numpy/README.md) — ndarrays, broadcasting, vectorization, tensor-shape reasoning
3. [`03_data_libraries`](day01_python_data_math/03_data_libraries/README.md) — Pandas, Polars, PyArrow
4. [`04_datasets`](day01_python_data_math/04_datasets/README.md) — dataset design + Hugging Face `datasets`
5. [`05_ml_mathematics`](day01_python_data_math/05_ml_mathematics/README.md) — linear algebra, probability/statistics, calculus for ML

## Day 2 topics

6. [`06_ml_fundamentals`](day02_machine_learning/06_ml_fundamentals/README.md) — learning paradigms, parameters vs hyperparameters, capacity
7. [`07_regression`](day02_machine_learning/07_regression/README.md) — linear regression from scratch and with scikit-learn
8. [`08_classification`](day02_machine_learning/08_classification/README.md) — logistic regression, k-NN, trees, forests, boosting, XGBoost/LightGBM, SVM, Naive Bayes
9. [`09_unsupervised_learning`](day02_machine_learning/09_unsupervised_learning/README.md) — K-means, hierarchical, DBSCAN, PCA/SVD, t-SNE/UMAP
10. [`10_ml_optimization`](day02_machine_learning/10_ml_optimization/README.md) — SGD, momentum, AdaGrad, RMSProp, Adam, LR schedules
11. [`11_generalization`](day02_machine_learning/11_generalization/README.md) — bias-variance, L1/L2/Elastic Net, early stopping
12. [`12_ml_evaluation`](day02_machine_learning/12_ml_evaluation/README.md) — confusion matrix metrics, ROC/PR-AUC, ranking metrics, cross-validation
13. [`13_feature_engineering`](day02_machine_learning/13_feature_engineering/README.md) — encoding, scaling, imputation, selection, pipelines
14. [`14_hyperparameter_tuning`](day02_machine_learning/14_hyperparameter_tuning/README.md) — grid/random search, Bayesian optimization

## Day 3 topics

15. [`15_deep_learning`](day03_deep_learning/15_deep_learning/README.md) — perceptron, MLP forward pass, activations, losses
16. [`16_backpropagation`](day03_deep_learning/16_backpropagation/README.md) — the chain rule, by hand, verified numerically
17. [`17_deep_learning_architectures`](day03_deep_learning/17_deep_learning_architectures/README.md) — CNN, RNN/LSTM/GRU, attention, Transformer block
18. [`18_pytorch_basics`](day03_deep_learning/18_pytorch_basics/README.md) — tensors, devices, shapes
19. [`19_pytorch_autograd`](day03_deep_learning/19_pytorch_autograd/README.md) — `.backward()`, `.grad`, `torch.no_grad()`
20. [`20_pytorch_models`](day03_deep_learning/20_pytorch_models/README.md) — `nn.Module`, `nn.Linear`/`Embedding`/`LayerNorm`/`Dropout`
21. [`21_pytorch_data`](day03_deep_learning/21_pytorch_data/README.md) — `Dataset`, `DataLoader`, `collate_fn`, `num_workers`
22. [`22_pytorch_training`](day03_deep_learning/22_pytorch_training/README.md) — the complete training pipeline, checkpointing, early stopping
23. [`23_pytorch_performance`](day03_deep_learning/23_pytorch_performance/README.md) — mixed precision, gradient accumulation/clipping, profiling
24. [`24_huggingface_foundations`](day03_deep_learning/24_huggingface_foundations/README.md) — tokenizers, `AutoModel`, `generate()`, LoRA concept

## Capstone

[`projects/project_end_to_end_ai_foundation`](projects/project_end_to_end_ai_foundation/README.md)
— the complete pipeline (raw data → preprocessing → PyTorch `Dataset`/`DataLoader` → model →
training → evaluation → inference) on a real dataset, tying together all 3 days.

See `ROADMAP.md` for the full 3-day map and `STUDY_PLAN.md` for a suggested daily schedule.

## Validating notebooks

```bash
python scripts/validate_notebooks.py        # execute every notebook, assert outputs, write report
python scripts/run_notebooks.py <path.ipynb> # re-run and save outputs for a single notebook
```
