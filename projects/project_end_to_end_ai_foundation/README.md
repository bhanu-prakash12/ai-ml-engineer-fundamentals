# Capstone — End-to-End AI Foundation Project

## Objective

Run the complete pipeline this curriculum has built, topic by topic, start to finish on one
real dataset:

```text
Raw data
→ Python/Pandas/NumPy preprocessing   (Day 1: 01_python, 02_numpy, 03_data_libraries)
→ train/val/test split, leakage-safe  (Day 1: 04_datasets, Day 2: 12_ml_evaluation)
→ feature scaling (fit on train only) (Day 2: 13_feature_engineering)
→ PyTorch Dataset/DataLoader          (Day 3: 21_pytorch_data)
→ nn.Module model                     (Day 3: 20_pytorch_models)
→ training loop + checkpointing       (Day 3: 22_pytorch_training)
→ evaluation (precision/recall/F1/ROC-AUC) (Day 2: 12_ml_evaluation)
→ inference on new data               (Day 3: 19_pytorch_autograd's torch.no_grad + 20's .eval())
```

Nothing here is a new concept — every step is a direct reuse of something already built and
validated earlier in the curriculum. The point of the capstone is seeing the whole thing work
together, end to end, once.

## Dataset

[`sklearn.datasets.load_breast_cancer`](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset)
— 569 samples, 30 numeric features (cell nucleus measurements from breast tissue biopsies), a
binary target (malignant/benign). Built into scikit-learn, no download, no license concerns,
small enough to run entirely on CPU in seconds.

- **Source**: UCI Machine Learning Repository, via `scikit-learn`.
- **Purpose**: binary classification — is a tumor malignant or benign, from measured features.
- **Schema**: 30 continuous numeric features (e.g. `mean radius`, `mean texture`, `worst area`); target: `0` = malignant, `1` = benign.
- **Samples**: 569 total (212 malignant, 357 benign — a mild, realistic class imbalance).
- **License**: public domain / freely redistributable via scikit-learn.

## Notebook

| Notebook | Purpose | Expected output |
|---|---|---|
| `01_end_to_end_pipeline.ipynb` | The complete pipeline above, on the breast cancer dataset | A trained PyTorch MLP achieving reasonable held-out test performance (precision/recall/F1/ROC-AUC all reported, not just accuracy); the best checkpoint reloaded and used for inference on a few held-out samples with predictions compared to ground truth |

## How to run

Same as every other notebook in this repo — open in Colab or run locally after
`pip install -r ../../requirements.txt`.

## Key takeaways

- Every stage of this pipeline was already built and unit-validated earlier in the curriculum — the capstone's job is wiring, not invention.
- Fitting the scaler on the training split only (not full data) and evaluating with precision/recall/F1/ROC-AUC (not just accuracy) are both habits from Day 2 that matter just as much with a real PyTorch model as they did with scikit-learn models.
- `model.eval()` + `torch.no_grad()` at inference time (Day 3) is the same discipline used throughout `22_pytorch_training`.

## Exercises

1. Swap the MLP for a deeper network (3+ hidden layers) and compare validation performance.
2. Add L2 weight decay (Day 2's `11_generalization`, now via PyTorch's optimizer `weight_decay` argument) and observe its effect on the train/validation loss gap.
3. Repeat the whole pipeline with `sklearn.datasets.load_wine` (a multiclass problem) and adapt the model's output layer and loss function accordingly.

## References

- [scikit-learn — Breast cancer Wisconsin dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset)
- Every Day 1/2/3 topic README referenced above.
