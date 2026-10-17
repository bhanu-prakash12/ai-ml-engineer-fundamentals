# 08 — Classification

> Pick the right model to put a label on each example — and know why it wins or fails.

| At a glance | |
|---|---|
| **Day** | Day 2 — Machine Learning |
| **Topic** | Classification algorithms (linear, instance, tree, ensemble, margin, probabilistic) |
| **Notebooks** | 2 (`01_classification_fundamentals`, `02_xgboost_lightgbm`) |
| **Builds on** | [`07_regression`](../07_regression/README.md) |
| **Leads to** | [`09_unsupervised_learning`](../09_unsupervised_learning/README.md), [`12_ml_evaluation`](../12_ml_evaluation/README.md) |

## 📖 Short forms used in this topic

| Term | Full form | What it is |
|---|---|---|
| **k-NN** | k-Nearest Neighbors | Predicts from the majority label of the k closest training points. |
| **SVM** | Support Vector Machine | Classifier that finds the maximum-margin separating boundary. |
| **SVC** | Support Vector Classifier | scikit-learn's SVM classifier class (`sklearn.svm.SVC`). |
| **RBF** | Radial Basis Function | Similarity kernel that decays with distance; a common SVM kernel for non-linear boundaries. |
| **XGBoost** | eXtreme Gradient Boosting | Fast, regularised gradient-boosted decision-tree library. |
| **LightGBM** | Light Gradient Boosting Machine | Histogram-based gradient-boosted tree library, optimised for speed and memory. |
| **GBM** | Gradient Boosting Machine | Trees built sequentially, each correcting the previous errors. |
| **OpenMP** | Open Multi-Processing | Multi-threading runtime XGBoost/LightGBM need (`libomp` on macOS). |
| **XOR** | Exclusive OR | Pattern (true only if inputs differ) that no single straight line can separate. |
| **F1** | F1 score | Harmonic mean of precision and recall. |
| **ROC** | Receiver Operating Characteristic | Curve of true-positive rate vs false-positive rate across thresholds. |
| **AUC** | Area Under the Curve | Single-number summary of a curve; ROC-AUC = chance a random positive is ranked above a random negative. |

## 🧠 In plain words

Classification predicts a *category* (spam / not spam, which digit, which intent) instead of a number. Different algorithms draw the boundary between categories in different ways: straight lines, vote of neighbours, a flowchart of yes/no questions, a committee of flowcharts, a widest-gap street, or probability bookkeeping. No single one wins everywhere.

## 🎒 Analogy to remember

Sorting mail into bins. Logistic regression uses one straight dividing rule; k-NN asks the nearest neighbours' houses what they are; a decision tree follows a checklist of questions; a random forest takes a vote from many different checklists; boosting has each new clerk fix the previous clerks' mistakes. Where it breaks down: real mail has a fixed label, but ML must also guess well on mail it has never seen (generalization).

## Learning objectives

By the end of this topic you can:
- Distinguish binary, multiclass, and multilabel classification.
- Fit and reason about logistic regression, k-NN, decision trees, random forests, gradient boosting, XGBoost, LightGBM, SVM, and Naive Bayes.
- For each algorithm, state what it does, how it works, when to use it, its strengths/weaknesses, and common failure cases.

## Why it matters

Classification is the most common supervised-learning task in production ML (spam detection,
fraud, churn, intent classification). Knowing *when* to reach for logistic regression vs a
gradient-boosted tree vs an SVM — and why — is a staple interview topic and a real engineering
decision with cost/latency/interpretability tradeoffs.

## Prerequisites

[`07_regression`](../07_regression/README.md) (the same fit/predict/evaluate pipeline, now with
a categorical target).

## Core concepts

- **Binary / multiclass / multilabel**: binary = 2 classes; multiclass = >2 mutually exclusive classes; multilabel = each sample can belong to multiple classes simultaneously.
- **Logistic regression**: linear regression's output passed through a sigmoid, trained with log-loss — *what*: a linear decision boundary with probabilistic output. *How*: gradient descent on cross-entropy loss. *When*: interpretable baseline, linearly (or near-linearly) separable data. *Strengths*: fast, interpretable, probabilistic output. *Weaknesses*: can't capture non-linear boundaries without manual feature engineering. *Failure case*: XOR-like data that isn't linearly separable.
- **k-NN**: classify by majority vote among the `k` nearest training points. *When*: small datasets, non-linear boundaries, no training phase needed. *Weaknesses*: slow at inference (must scan training data), sensitive to feature scaling and the curse of dimensionality.
- **Decision tree**: recursively split on the feature/threshold that best separates classes. *Strengths*: interpretable, handles non-linear boundaries and mixed feature types. *Weaknesses*: prone to overfitting if unconstrained (unlimited depth).
- **Random forest**: an ensemble of decision trees, each trained on a bootstrap sample with random feature subsets, averaged/voted. *Strengths*: much less overfitting than a single tree, good default performance. *Weaknesses*: less interpretable, slower, larger memory footprint.
- **Gradient boosting**: builds trees sequentially, each correcting the previous ensemble's errors (fits the residual/gradient of the loss). *Strengths*: often the best tabular-data performance. *Weaknesses*: more hyperparameters to tune, easier to overfit than random forest if not careful, slower to train (sequential).
- **XGBoost / LightGBM**: optimized, production-grade gradient boosting implementations — XGBoost is depth-wise with strong regularization controls; LightGBM grows leaf-wise and uses histogram-based splits for speed on large datasets. *When*: large tabular datasets, Kaggle-style competitions, when squeezing out maximum tabular accuracy matters.
- **SVM**: finds the maximum-margin separating hyperplane (optionally using a kernel trick for non-linear boundaries). *Strengths*: effective in high dimensions, strong theoretical guarantees. *Weaknesses*: doesn't scale well to very large datasets, requires careful kernel/hyperparameter choice.
- **Naive Bayes**: applies Bayes' theorem assuming features are conditionally independent given the class. *Strengths*: extremely fast, works surprisingly well for text classification despite the "naive" independence assumption. *Weaknesses*: the independence assumption is usually false, which can bias probability estimates (even when the final classification is still correct).

## 🔍 Concepts explained

> [!NOTE]
> Group chips: 🟦 linear / probabilistic, 🟩 tree-based and ensembles, 🟧 instance / margin based, 🟪 task framing.

### 🟪 Binary / multiclass / multilabel
- **Definition**: binary = 2 exclusive classes; multiclass = more than 2 exclusive classes; multilabel = each sample can carry several labels.
- **Analogy**: one yes/no switch; a multiple-choice question; a tag cloud.
- **Use when / watch out**: multilabel is not multiclass — labels are not mutually exclusive.

### 🟦 Logistic regression
- **Definition**: a linear score passed through a sigmoid to give a probability, trained with log-loss.
- **Analogy**: a weighted checklist converted to a percentage.
- **Code idea**: `LogisticRegression(max_iter=1000).fit(X, y)`.
- **Use when / watch out**: interpretable baseline; fails on XOR-like, non-linearly-separable data.

### 🟦 Naive Bayes
- **Definition**: Bayes' theorem assuming features are independent given the class.
- **Analogy**: judging an email by tallying each word's spam-ness separately.
- **Code idea**: `GaussianNB().fit(X, y)`.
- **Use when / watch out**: very fast text baseline; probabilities are often poorly calibrated.

### 🟧 k-NN
- **Definition**: vote among the `k` closest training points.
- **Analogy**: asking your nearest neighbours which party to join.
- **Code idea**: `KNeighborsClassifier(n_neighbors=5)`.
- **Use when / watch out**: scale features; small `k` gives jagged boundaries, large `k` smooth ones; slow inference.

### 🟧 SVM
- **Definition**: find the hyperplane with the widest margin; the kernel trick (e.g. RBF) allows non-linear boundaries.
- **Analogy**: building the widest possible street between two neighbourhoods.
- **Code idea**: `SVC(kernel="rbf")`.
- **Use when / watch out**: scale features; poor scaling to very large datasets.

### 🟩 Decision tree
- **Definition**: recursively split on the feature/threshold that best separates classes.
- **Analogy**: a game of twenty questions.
- **Code idea**: `DecisionTreeClassifier(max_depth=5)`.
- **Use when / watch out**: interpretable; unconstrained depth reaches 100% training accuracy (overfit).

### 🟩 Random forest (bagging)
- **Definition**: many trees, each on a bootstrap sample with random feature subsets, voted together.
- **Analogy**: a jury of independent experts.
- **Code idea**: `RandomForestClassifier(n_estimators=100)`.
- **Use when / watch out**: strong default; less interpretable, bigger memory.

### 🟩 Gradient boosting / XGBoost / LightGBM
- **Definition**: trees built one after another, each fitting the errors (loss gradient) of the ensemble so far. XGBoost grows depth-wise with explicit regularization (`reg_alpha`, `reg_lambda`, `gamma`); LightGBM grows leaf-wise and bins features into histograms.
- **Analogy**: a relay of editors, each fixing what the last missed.
- **Code idea**: `xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)`, `lgb.LGBMClassifier(...)`.
- **Use when / watch out**: top tabular accuracy; more hyperparameters, can overfit small/noisy data; leaf-wise growth can overfit small datasets.

### Differences at a glance

| Algorithm | Group | Boundary | Needs scaling? | Strength | Pitfall |
|---|---|---|---|---|---|
| Logistic regression | 🟦 | Linear | Helpful | Fast, interpretable, probabilistic | Non-linear data |
| Naive Bayes | 🟦 | Probabilistic | No | Very fast, good for text | Independence assumption |
| k-NN | 🟧 | Local, jagged | Yes | No training phase | Slow inference, high dimensions |
| SVM (RBF) | 🟧 | Margin-based, non-linear | Yes | High dimensions | Large datasets, kernel choice |
| Decision tree | 🟩 | Axis-aligned splits | No | Interpretable | Overfits |
| Random forest | 🟩 | Averaged trees | No | Robust default | Size, less interpretable |
| Gradient boosting / XGBoost / LightGBM | 🟩 | Sequential trees | No | Best tabular accuracy | Tuning, overfitting |

> [!IMPORTANT]
> Trees do not need feature scaling; k-NN and SVM do, because they depend on distances or margins.

> [!WARNING]
> Accuracy alone misleads on imbalanced classes; use the metrics in `12_ml_evaluation`.

> [!TIP]
> Start with a fast baseline (logistic regression or Naive Bayes), then try a tree ensemble, and compare on the same split.

## Concept map

```mermaid
flowchart LR
  A["Classification task"] --> B["Framing: binary / multiclass / multilabel"]
  subgraph LIN["Linear and probabilistic"]
    C["Logistic Regression"]
    I["Naive Bayes"]
  end
  subgraph INST["Instance and margin based"]
    D["k-NN"]
    H["SVM"]
  end
  subgraph TREE["Tree based"]
    E["Decision Tree"]
    F["Bagging: Random Forest"]
    G["Boosting: GBM / XGBoost / LightGBM"]
  end
  B --> C
  B --> I
  B --> D
  B --> H
  B --> E
  E --> F
  E --> G
  J["12_ml_evaluation: accuracy / precision / recall / F1 / ROC-AUC"]
  C --> J
  D --> J
  F --> J
  G --> J
  classDef task fill:#EDE9FE,stroke:#7C3AED,color:#1a1a1a;
  classDef lin fill:#DBEAFE,stroke:#2563EB,color:#1a1a1a;
  classDef inst fill:#FEF3C7,stroke:#D97706,color:#1a1a1a;
  classDef tree fill:#DCFCE7,stroke:#16A34A,color:#1a1a1a;
  classDef eval fill:#FEE2E2,stroke:#DC2626,color:#1a1a1a;
  class A,B task;
  class C,I lin;
  class D,H inst;
  class E,F,G tree;
  class J eval;
```

Choosing a starting model:

```mermaid
flowchart TD
  Q1{"Tabular data?"}
  Q1 -->|"yes, want best accuracy"| R1["Gradient boosting: XGBoost or LightGBM"]
  Q1 -->|"yes, want robust default"| R2["Random forest"]
  Q1 -->|"need interpretable baseline"| R3["Logistic regression or shallow tree"]
  Q1 -->|"sparse text features"| R4["Naive Bayes or logistic regression"]
  classDef q fill:#DBEAFE,stroke:#2563EB,color:#1a1a1a;
  classDef a fill:#DCFCE7,stroke:#16A34A,color:#1a1a1a;
  class Q1 q;
  class R1,R2,R3,R4 a;
```

## Important APIs / functions

`sklearn.linear_model.LogisticRegression`, `sklearn.neighbors.KNeighborsClassifier`,
`sklearn.tree.DecisionTreeClassifier`, `sklearn.ensemble.RandomForestClassifier`,
`sklearn.ensemble.GradientBoostingClassifier`, `xgboost.XGBClassifier`,
`lightgbm.LGBMClassifier`, `sklearn.svm.SVC`, `sklearn.naive_bayes.GaussianNB`,
`sklearn.metrics.accuracy_score/classification_report`.

## Notebooks

| Notebook | Purpose | Expected output |
|---|---|---|
| `01_classification_fundamentals.ipynb` | Binary/multiclass/multilabel framing; logistic regression, k-NN, decision tree, random forest, gradient boosting, SVM, Naive Bayes, all fit on the same dataset(s) and compared | A comparison table of accuracy across all 7 algorithms on one dataset; a visualized decision boundary per algorithm on a 2D toy dataset |
| `02_xgboost_lightgbm.ipynb` | XGBoost and LightGBM specifically, compared to `GradientBoostingClassifier` from notebook 1 | Accuracy/training-time comparison across sklearn's gradient boosting, XGBoost, and LightGBM; gracefully skips with a clear message if either library isn't installed in the current environment |

## Key takeaways

- There's no universally "best" classifier — the right choice depends on data size, dimensionality, interpretability needs, and latency budget.
- Tree ensembles (random forest, gradient boosting, XGBoost, LightGBM) are usually the strongest default for tabular data; linear models and Naive Bayes are strong, fast baselines.
- Decision boundaries make the differences between algorithms visually obvious: logistic regression and SVM (linear kernel) draw straight lines; k-NN and trees draw jagged, local boundaries.

## Common mistakes

- Not scaling features before k-NN or SVM (both are distance/margin-based and sensitive to feature scale).
- Letting a single decision tree grow unconstrained and being surprised it overfits.
- Assuming XGBoost/LightGBM will always beat simpler models — on small or very noisy datasets they can overfit without careful tuning.
- Using accuracy as the only metric on an imbalanced classification dataset (see `12_ml_evaluation`).

## Interview questions

**Conceptual**
1. What's the practical difference between bagging (random forest) and boosting (gradient boosting)?
2. Why is feature scaling important for k-NN and SVM but not for decision trees?
3. What does "naive" refer to in Naive Bayes, and why does the assumption still often work in practice?
4. When would you prefer LightGBM over XGBoost, or vice versa?
5. Why can a single unconstrained decision tree achieve 100% training accuracy, and why is that a red flag?

**Coding**
1. Fit logistic regression, a decision tree, and a random forest on the same dataset and compare test accuracy.
2. Visualize the decision boundary of an SVM with a linear kernel vs an RBF kernel on the same 2D dataset.
3. Train an XGBoost classifier (or gracefully skip with a clear message if unavailable) and compare its accuracy and training time to `GradientBoostingClassifier`.

**Debugging**
1. Your k-NN classifier performs far worse than expected — what's the first thing to check about your features?
2. Your random forest gets near-100% training accuracy and poor test accuracy — is this still overfitting, given it's an ensemble? Why or why not?

**Real-world engineering**
1. You need sub-millisecond inference latency for a classifier in a trading system — which of these algorithms would you rule out immediately, and why?
2. You're classifying text with 50,000 sparse features and want a fast, strong baseline before trying anything fancier — which algorithm from this list would you reach for first?

## Exercises

1. Fit all 7 fundamentals-notebook algorithms on the same synthetic dataset and rank them by test accuracy and by training time.
2. Visualize decision boundaries for logistic regression, k-NN (k=1 vs k=15), and a decision tree on the same 2D moons-shaped dataset, and describe how each differs.
3. Tune `max_depth` for a single decision tree from 1 to 20 and plot train vs test accuracy — identify where overfitting begins.

## References

- [scikit-learn — Supervised learning](https://scikit-learn.org/stable/supervised_learning.html)
- [XGBoost docs](https://xgboost.readthedocs.io/)
- [LightGBM docs](https://lightgbm.readthedocs.io/)
- [scikit-learn — SVM](https://scikit-learn.org/stable/modules/svm.html)
