# 07 — Regression

## Learning objectives

By the end of this topic you can:
- Derive and implement linear and multiple linear regression from scratch with NumPy (least squares and gradient descent).
- Fit the same models with scikit-learn and get matching results.
- Compute and interpret MSE, MAE, RMSE, and R².

## Why it matters

Linear regression is the simplest supervised model and the cleanest way to see the full
ML pipeline (fit → predict → evaluate) without an opaque algorithm in the middle. Every
regression metric here (MSE/MAE/RMSE/R²) reappears throughout Day 2 and in real evaluation
dashboards.

## Prerequisites

[`06_ml_fundamentals`](../06_ml_fundamentals/README.md), Day 1's
[`05_ml_mathematics`](../../day01_python_data_math/05_ml_mathematics/README.md) (gradient descent, calculus).

## Core concepts

- **Linear regression**: `ŷ = wx + b` (one feature) or `ŷ = Xw + b` (many features) — fit the
  line/hyperplane that minimizes squared error.
- **Multiple regression**: more than one feature; `X` becomes a matrix, `w` becomes a vector.
- **Least squares (closed-form)**: `w = (XᵗX)⁻¹Xᵗy` — directly solves for the optimal weights
  when `XᵗX` is invertible.
- **Gradient descent**: iteratively approaches the same optimum (useful when the closed-form
  solution doesn't scale or doesn't exist, e.g. non-invertible `XᵗX`).
- **MSE / MAE / RMSE / R²**: MSE punishes large errors more (squared); MAE is robust to outliers
  (absolute); RMSE puts the error back in the original units; R² is the fraction of variance
  explained (1.0 = perfect, 0.0 = no better than predicting the mean).

```mermaid
flowchart LR
  A[Data: X, y] --> B[Closed-form: w = (XᵗX)⁻¹Xᵗy]
  A --> C[Gradient descent: iteratively minimize MSE]
  B --> D[Predictions ŷ = Xw + b]
  C --> D
  D --> E[Evaluate: MSE / MAE / RMSE / R²]
  E -.connects to.-> F[12_ml_evaluation: the same metrics, formalized]
```

## Important APIs / functions

NumPy: `np.linalg.inv`, `@`, manual gradient loop (from `05_ml_mathematics`).
scikit-learn: `sklearn.linear_model.LinearRegression`, `sklearn.metrics.mean_squared_error`,
`mean_absolute_error`, `r2_score`.

## Notebooks

| Notebook | Purpose | Expected output |
|---|---|---|
| `01_linear_regression_numpy.ipynb` | Linear + multiple regression from scratch: closed-form least squares AND gradient descent, on synthetic data | Learned weights matching the true generating weights within tolerance; closed-form and gradient-descent solutions agreeing with each other |
| `02_linear_regression_sklearn.ipynb` | The same data fit with `sklearn.linear_model.LinearRegression`; MSE/MAE/RMSE/R² computed and interpreted; a comparison table against the from-scratch results | scikit-learn's learned coefficients matching the from-scratch closed-form solution within floating-point tolerance |

## Key takeaways

- The closed-form least-squares solution and gradient descent converge to (approximately) the same answer — gradient descent is the general-purpose tool that also works when the closed form doesn't (e.g. in neural networks).
- R² can be negative on test data — it means the model is worse than just predicting the mean.
- Always report the metric in context: RMSE in the target's original units is much easier for stakeholders to interpret than raw MSE.

## Common mistakes

- Reporting only R² without also showing residuals/MAE — R² can mask the fact that a model is skewed by a few outliers.
- Forgetting to add a bias/intercept term and being confused why a NumPy from-scratch fit underperforms scikit-learn's default (which includes an intercept).
- Confusing MSE (useful for optimization, sensitive to outliers) with MAE (useful for reporting typical error, robust to outliers) and using the wrong one for the situation.

## Interview questions

**Conceptual**
1. Derive (at a high level) why minimizing squared error leads to the normal equation `w = (XᵗX)⁻¹Xᵗy`.
2. When would gradient descent be preferred over the closed-form solution?
3. What does an R² of 0 mean? Can R² be negative, and what does that indicate?
4. Why is RMSE more interpretable than MSE in a business report?
5. Why is MAE more robust to outliers than MSE?

**Coding**
1. Implement least-squares regression from scratch using the normal equation and verify it matches `sklearn.linear_model.LinearRegression`.
2. Implement gradient descent for multiple linear regression (2+ features) and verify convergence to the closed-form solution.
3. Compute MSE, MAE, RMSE, and R² from raw predictions and true values without using `sklearn.metrics`.

**Debugging**
1. Your from-scratch gradient descent for linear regression diverges (loss grows) — what are the two most likely causes?
2. Your closed-form solution throws a `LinAlgError: Singular matrix` — what does that imply about your features?

**Real-world engineering**
1. You're predicting house prices and a few mansions are 100x the median price — how would MSE vs MAE change your model's behavior, and which would you choose?
2. Your feature matrix has two nearly-identical (collinear) columns — why might this break the closed-form solution, and what's a practical fix?

## Exercises

1. Generate `y = 3x1 - 2x2 + 5 + noise`; recover `w ≈ [3, -2]` and `b ≈ 5` via both closed-form and gradient descent.
2. Compute MSE/MAE/RMSE/R² for a model that always predicts the mean of `y`, and verify R² ≈ 0.
3. Add a single extreme outlier to your dataset and compare how much MSE vs MAE change.

## References

- [scikit-learn — Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [Stanford CS229 — Linear Regression notes](https://cs229.stanford.edu/notes2022fall/cs229-notes1.pdf)
- [scikit-learn — Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
