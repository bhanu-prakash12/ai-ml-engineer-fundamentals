# 09 — Unsupervised Learning

> Find structure in data that has no labels: group similar points, or squeeze many features into few.

| At a glance | |
|---|---|
| **Day** | Day 2 — Machine Learning |
| **Topic** | Clustering and dimensionality reduction |
| **Notebooks** | 2 (`01_clustering`, `02_dimensionality_reduction`) |
| **Builds on** | [`08_classification`](../08_classification/README.md), Day 1 `05_ml_mathematics` (eigenvectors, SVD) |
| **Leads to** | [`10_ml_optimization`](../10_ml_optimization/README.md) |

## 📖 Short forms used in this topic

| Term | Full form | What it is |
|---|---|---|
| **DBSCAN** | Density-Based Spatial Clustering of Applications with Noise | Clusters dense regions and labels sparse points as noise; no need to set k. |
| **eps** | Epsilon (neighbourhood radius) | DBSCAN hyperparameter: how close points must be to be neighbours. |
| **min_samples** | Minimum samples | DBSCAN hyperparameter: neighbours needed to form a dense core. |
| **ARI** | Adjusted Rand Index | Agreement between clusters and true labels, corrected for chance (1 = perfect, ~0 = random). |
| **PCA** | Principal Component Analysis | Linear dimensionality reduction: projects data onto orthogonal directions of maximum variance. |
| **SVD** | Singular Value Decomposition | Factorises a matrix as U·Σ·Vᵀ; basis for low-rank approximation and PCA. |
| **t-SNE** | t-distributed Stochastic Neighbor Embedding | Non-linear 2D/3D embedding that preserves local neighbourhoods; for visualisation only. |
| **UMAP** | Uniform Manifold Approximation and Projection | Non-linear dimensionality reduction; faster than t-SNE and keeps more global structure. |
| **API** | Application Programming Interface | The set of functions/classes a library exposes. |
| **WCSS** | Within-Cluster Sum of Squares | Sum of squared distances of points to their cluster centre; this is K-means' `inertia_`. |

## 🧠 In plain words

Unsupervised learning works on data with no answer column. Clustering groups similar rows together (customer segments, outliers). Dimensionality reduction rewrites many columns as a few new ones, either to compress data (PCA/SVD) or to draw it on a 2D plot (t-SNE/UMAP).

## 🎒 Analogy to remember

Sorting a pile of unlabeled photos: clustering is making piles of look-alikes; PCA is choosing the one or two camera angles that show the most difference between photos; t-SNE is arranging photos on a table so near-duplicates sit together, even though distances across the table mean little. Where it breaks down: a person knows what "similar" means; an algorithm only knows the distance or density rule you gave it.

## Learning objectives

By the end of this topic you can:
- Fit and compare K-means, hierarchical clustering, and DBSCAN.
- Explain PCA and SVD as dimensionality reduction, and apply them before and after clustering.
- Use t-SNE/UMAP for visualization and explain why they're for visualization, not general-purpose dimensionality reduction for downstream models.

## Why it matters

Most real-world data has no labels. Clustering surfaces structure (customer segments, anomalies);
dimensionality reduction makes high-dimensional data tractable to visualize, store, and feed to
simpler downstream models — both are everyday tools, not just academic exercises.

## Prerequisites

[`08_classification`](../08_classification/README.md); Day 1's
[`05_ml_mathematics`](../../day01_python_data_math/05_ml_mathematics/README.md) (eigenvectors/SVD, covered again here in an ML-algorithm context).

## Core concepts

- **K-means**: partitions data into `k` clusters by iteratively assigning points to the nearest centroid and recomputing centroids. *Strengths*: fast, simple. *Weaknesses*: must choose `k` in advance, assumes roughly spherical, similarly-sized clusters.
- **Hierarchical clustering**: builds a tree (dendrogram) of nested clusters by repeatedly merging (agglomerative) the closest pair. *Strengths*: no need to pick `k` upfront, the dendrogram shows structure at every granularity. *Weaknesses*: O(n²) or worse, doesn't scale to very large datasets.
- **DBSCAN**: density-based — groups points that are densely packed together, marking sparse points as noise/outliers. *Strengths*: finds arbitrarily shaped clusters, automatically detects outliers, no need to specify `k`. *Weaknesses*: sensitive to its two hyperparameters (`eps`, `min_samples`), struggles when clusters have very different densities.
- **PCA**: projects data onto the directions (principal components) of maximum variance — a linear dimensionality reduction (covered mathematically in Day 1's `05_ml_mathematics`).
- **SVD**: the general matrix factorization PCA is built on; applies even to non-square, sparse matrices (e.g. recommender systems).
- **t-SNE / UMAP**: non-linear techniques that preserve *local* neighborhood structure for 2D/3D visualization. They distort global distances and are **not** suitable as a general preprocessing step before a downstream supervised model (unlike PCA) — their primary use is making high-dimensional data visually inspectable.

## 🔍 Concepts explained

> [!NOTE]
> Group chips: 🟦 clustering, 🟩 linear dimensionality reduction, 🟧 non-linear (visualization) reduction.

### 🟦 K-means
- **Definition**: assign each point to the nearest of `k` centroids, recompute centroids, repeat.
- **Analogy**: placing `k` fire stations so each house goes to its nearest one.
- **Code idea**: `KMeans(n_clusters=3, n_init=10).fit(X)`; `inertia_` for the elbow plot, `cluster_centers_` for centroids.
- **Use when / watch out**: round, similar-sized clusters; must choose `k`; scale features.

### 🟦 Hierarchical (agglomerative) clustering
- **Definition**: repeatedly merge the two closest clusters, producing a dendrogram you can cut at any height.
- **Analogy**: a family tree built from the bottom up.
- **Code idea**: `AgglomerativeClustering`, `linkage` + `dendrogram` from SciPy.
- **Use when / watch out**: small data and when `k` is unknown; costly on large data.

### 🟦 DBSCAN
- **Definition**: grows clusters from dense neighbourhoods (`eps`, `min_samples`); sparse points get label `-1` (noise).
- **Analogy**: finding crowds in a park; loners stay unassigned.
- **Code idea**: `DBSCAN(eps=0.2, min_samples=5).fit(X)`.
- **Use when / watch out**: arbitrary shapes and outlier detection; sensitive to `eps` and to varied densities.

### 🟩 PCA
- **Definition**: project onto orthogonal directions of maximum variance; a scree plot shows variance explained per component.
- **Analogy**: choosing the camera angle that shows the most spread.
- **Code idea**: `PCA(n_components=2)`; `np.cumsum(explained_variance_ratio_)` to pick components for 95% variance.
- **Use when / watch out**: safe as preprocessing for downstream models; only captures linear structure.

### 🟩 SVD
- **Definition**: factor a matrix as `U·Σ·Vᵀ`; scikit-learn's PCA uses SVD of the centred data.
- **Analogy**: breaking a complex movement into its strongest simple motions.
- **Code idea**: `np.linalg.svd(X_centered, full_matrices=False)`; top rows of `Vt` match PCA components up to sign.
- **Use when / watch out**: works on non-square matrices; centre the data first for PCA equivalence.

### 🟧 t-SNE / UMAP
- **Definition**: non-linear embeddings that keep local neighbourhoods when projecting to 2D/3D.
- **Analogy**: seating guests so friends sit together; table distances across the room are not meaningful.
- **Code idea**: `TSNE(n_components=2, perplexity=30)`; `umap.UMAP(n_components=2)` (optional install).
- **Use when / watch out**: for looking at data only, not as model features; layouts vary by seed.

### Differences at a glance

| Method | Group | Needs `k`? | Shape assumed | Strength | Pitfall |
|---|---|---|---|---|---|
| K-means | 🟦 | Yes | Round, similar size | Fast, simple | Fails on moons or uneven sizes |
| Hierarchical | 🟦 | No (cut dendrogram) | Depends on linkage | Shows structure at all scales | Does not scale |
| DBSCAN | 🟦 | No | Arbitrary | Finds outliers | `eps` and `min_samples` tuning |
| PCA / SVD | 🟩 | n components | Linear | Stable, reusable for models | Misses non-linear structure |
| t-SNE | 🟧 | n components (2-3) | Non-linear | Great local clusters in plots | Not for features; seed dependent |
| UMAP | 🟧 | n components | Non-linear | Faster, keeps more global structure | Optional dependency |

> [!WARNING]
> Never feed t-SNE/UMAP coordinates to a downstream classifier: distances are not globally meaningful and layouts are not stable across runs.

> [!TIP]
> Scale features before K-means, DBSCAN and PCA; inertia always falls as `k` grows, so look for the elbow.

## Concept map

```mermaid
flowchart LR
  A["Unlabeled data"] --> B["Clustering"]
  A --> F["Dimensionality reduction"]
  subgraph CL["Clustering"]
    C["K-means: round clusters, fixed k"]
    D["Hierarchical: dendrogram, any k afterwards"]
    E["DBSCAN: density based, finds outliers"]
  end
  subgraph DR["Dimensionality reduction"]
    G["PCA and SVD: linear, global variance"]
    H["t-SNE and UMAP: non-linear, visualization only"]
  end
  B --> C
  B --> D
  B --> E
  F --> G
  F --> H
  G -.->|"often applied before"| B
  classDef data fill:#EDE9FE,stroke:#7C3AED,color:#1a1a1a;
  classDef clu fill:#DBEAFE,stroke:#2563EB,color:#1a1a1a;
  classDef lin fill:#DCFCE7,stroke:#16A34A,color:#1a1a1a;
  classDef nonlin fill:#FEF3C7,stroke:#D97706,color:#1a1a1a;
  class A,B,F data;
  class C,D,E clu;
  class G lin;
  class H nonlin;
```

Which technique?

```mermaid
flowchart TD
  Q1{"Goal?"}
  Q1 -->|"group points"| Q2{"Know k and round clusters?"}
  Q2 -->|"yes"| R1["K-means"]
  Q2 -->|"no, odd shapes or outliers"| R2["DBSCAN"]
  Q2 -->|"want a hierarchy"| R3["Hierarchical"]
  Q1 -->|"compress for models"| R4["PCA or SVD"]
  Q1 -->|"look at data in 2D"| R5["t-SNE or UMAP"]
  classDef q fill:#DBEAFE,stroke:#2563EB,color:#1a1a1a;
  classDef a fill:#DCFCE7,stroke:#16A34A,color:#1a1a1a;
  classDef w fill:#FEF3C7,stroke:#D97706,color:#1a1a1a;
  class Q1,Q2 q;
  class R1,R2,R3,R4 a;
  class R5 w;
```

## Important APIs / functions

`sklearn.cluster.KMeans`, `sklearn.cluster.AgglomerativeClustering`,
`scipy.cluster.hierarchy.dendrogram`, `sklearn.cluster.DBSCAN`, `sklearn.decomposition.PCA`,
`numpy.linalg.svd`, `sklearn.manifold.TSNE`, `umap.UMAP` (optional — falls back gracefully if
not installed).

## Notebooks

| Notebook | Purpose | Expected output |
|---|---|---|
| `01_clustering.ipynb` | K-means, hierarchical clustering (with dendrogram), and DBSCAN on the same 2D synthetic dataset, plus a case where K-means fails and DBSCAN succeeds (non-spherical clusters) | Side-by-side cluster visualizations; a dendrogram; a clear visual demonstration of K-means failing on non-convex clusters that DBSCAN handles correctly |
| `02_dimensionality_reduction.ipynb` | PCA (reusing/extending Day 1's eigendecomposition), SVD, and t-SNE (UMAP if available, else skipped gracefully) on a higher-dimensional dataset (e.g. digits) | A scree plot (variance explained per component); a 2D PCA projection vs a 2D t-SNE projection of the same data, compared |

## Key takeaways

- K-means assumes clusters are roughly spherical and similar in size — it fails visibly on elongated or nested shapes, which DBSCAN handles via density instead of distance-to-centroid.
- PCA's components are ranked by variance explained — a scree plot tells you how many components actually matter.
- t-SNE/UMAP plots are for *looking at* your data, not for feeding into a downstream model — their distances aren't globally meaningful.

## Common mistakes

- Picking `k` for K-means without checking an elbow plot or silhouette score.
- Feeding unscaled features into K-means or DBSCAN (both are distance-based).
- Using t-SNE output as input features for a classifier (loses global distance structure and isn't stable across re-runs).
- Forgetting DBSCAN's `eps` needs to match the data's scale — unscaled features make a single `eps` meaningless across dimensions.

## Interview questions

**Conceptual**
1. Why does K-means fail on non-convex or very differently-sized clusters?
2. What does DBSCAN do with points that don't belong to any dense region?
3. What's the practical difference between PCA and t-SNE, and why can't you substitute one for the other?
4. Why does hierarchical clustering not require you to pick `k` upfront?
5. What does a PCA scree plot tell you, and how would you use it to choose the number of components to keep?

**Coding**
1. Fit K-means with `k` from 2 to 10 and plot the elbow curve (inertia vs `k`).
2. Fit DBSCAN on data with two differently-shaped clusters (e.g. two interleaving moons) where K-means visibly fails.
3. Reduce a high-dimensional dataset to 2D with PCA and separately with t-SNE, and plot both side by side.

**Debugging**
1. Your K-means clusters look nothing like the obvious visual groups in a 2D scatter plot — what's the most likely cause?
2. DBSCAN marks almost every point as noise — which hyperparameter would you adjust first, and in which direction?

**Real-world engineering**
1. You want to compress a 500-dimensional embedding to 50 dimensions before storing it at scale — would you use PCA or t-SNE, and why?
2. You're segmenting customers by purchase behavior but don't know how many segments exist — which clustering algorithm would you start with, and why?

## Exercises

1. Generate 3 clusters of different sizes/densities and compare K-means vs DBSCAN cluster assignments visually.
2. Build a dendrogram for a small dataset and identify a sensible "cut height" to produce 3 clusters.
3. Compute the number of PCA components needed to explain 95% of variance on a real small dataset (e.g. `sklearn.datasets.load_digits`).

## References

- [scikit-learn — Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [scikit-learn — Decomposing signals (PCA)](https://scikit-learn.org/stable/modules/decomposition.html)
- [Distill.pub — How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)
- [UMAP documentation](https://umap-learn.readthedocs.io/)
