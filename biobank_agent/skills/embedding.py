"""t-SNE/UMAP patient embedding visualization."""

import numpy as np
from biobank_agent.registry import skill
from biobank_agent.utils.plotting import nature_figure, save_figure, PALETTE


@skill(
    name="embedding",
    description="Generate a 2D embedding (t-SNE or UMAP) of patients based on biomarker "
                "features, coloured by disease status. Requires a cohort to be built first.",
    parameters={
        "model_key": {
            "type": "string",
            "description": "Model/cohort key (e.g. 'E11_xgb'). Uses most recent if omitted.",
            "default": "",
        },
        "method": {
            "type": "string",
            "description": "Dimensionality reduction: 'umap' (default) or 'tsne'",
            "default": "umap",
            "enum": ["umap", "tsne"],
        },
        "sample_size": {
            "type": "integer",
            "description": "Optional max subjects to embed. Use 0 or omit to embed all rows currently in the model cohort.",
            "default": 0,
        },
    },
    required=[],
)
def embedding(model_key: str = "", method: str = "umap",
              sample_size: int = 0, *, ctx=None) -> dict:
    X = ctx.state.feature_matrix
    y = ctx.state.labels

    if X is None or y is None:
        return {"error": "No feature matrix. Run train_model first to create a cohort."}

    # Use the full model cohort by default. A positive sample_size remains
    # available for explicit smoke tests or resource-constrained visualisation.
    if sample_size and sample_size > 0:
        n = min(sample_size, len(X))
        idx = np.random.RandomState(42).choice(len(X), n, replace=False)
        X_sub = X.iloc[idx].fillna(X.median())
        y_sub = y.iloc[idx]
        sampling_applied = n < len(X)
    else:
        n = len(X)
        X_sub = X.fillna(X.median())
        y_sub = y
        sampling_applied = False

    # Scale
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X_sub)

    # Reduce
    if method == "umap":
        try:
            from umap import UMAP
            n_neighbors = min(30, n - 1) if n > 1 else 2
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.3)
        except ImportError:
            return {"error": "umap-learn not installed. pip install umap-learn"}
    else:
        from sklearn.manifold import TSNE
        perplexity = min(30, max(2, n - 1))
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)

    coords = reducer.fit_transform(X_scaled)

    # Plot — Nature style scatter
    fig, ax = nature_figure(width="single", height_ratio=1.0)

    unique_labels = sorted(y_sub.unique())
    if len(unique_labels) <= 2:
        controls = y_sub == 0
        cases = y_sub == 1
        ax.scatter(coords[controls, 0], coords[controls, 1],
                   c=PALETTE[0], s=3, alpha=0.3, label=f"Controls (n={int(controls.sum())})",
                   edgecolors="none", rasterized=True)
        ax.scatter(coords[cases, 0], coords[cases, 1],
                   c=PALETTE[1], s=3, alpha=0.5, label=f"Cases (n={int(cases.sum())})",
                   edgecolors="none", rasterized=True)
    else:
        for i, lbl in enumerate(unique_labels):
            mask = y_sub == lbl
            color = PALETTE[i % len(PALETTE)]
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=color, s=8, alpha=0.7, label=f"Class {lbl} (n={int(mask.sum())})",
                       edgecolors="none", rasterized=True)

    method_label = method.upper()
    ax.set_xlabel(f"{method_label}-1")
    ax.set_ylabel(f"{method_label}-2")
    ax.legend(frameon=False, markerscale=3, loc="upper right")
    ax.set_title(f"Patient embedding ({method_label}, n={n})")
    # Remove ticks for embedding plots
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    paths = save_figure(fig, f"embedding_{method}", ctx.report_dir)
    ctx.state.figures.extend(paths)

    label_counts = {}
    for lbl in unique_labels:
        label_counts[f"class_{lbl}"] = int((y_sub == lbl).sum())
    n_controls = int((y_sub == 0).sum()) if 0 in set(unique_labels) else 0
    n_cases = int((y_sub == 1).sum()) if 1 in set(unique_labels) else 0

    return {
        "method": method,
        "n_subjects": n,
        "requested_sample_size": int(sample_size or 0),
        "sampling_applied": sampling_applied,
        "label_distribution": label_counts,
        "n_cases": n_cases,
        "n_controls": n_controls,
        "n_features": X_sub.shape[1],
        "figure": str(paths[0]),
    }
