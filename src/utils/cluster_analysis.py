#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cluster-vs-Supervised Evaluation (Overall Only)

This module evaluates how well *unsupervised* clusters (e.g., `model_class_id`)
align with *supervised* “ground-truth” types (e.g., `supervised_type`) that are
already stored in a GeoJSON FeatureCollection of cell objects.

Typical usage (CLI):
    python -m src.eval.cluster_eval_overall \
        --input path/to/cells.geojson \
        --outdir outputs/eval \
        --plot

Inputs
------
- One or more GeoJSON files, each a FeatureCollection where each Feature has a
  `properties` dict containing (at least):
    * model_class_id : int | str   (the cluster ID produced by your pipeline)
    * supervised_type: int | str   (the supervised/ground-truth label)
      - If `supervised_type` is missing, the script will fall back to
        `anet_class_label` when available.
    * Optional helpful fields (not required): id, model_label, slide.

Outputs (written under --outdir)
-------------------------------
- overall_contingency.csv          : Crosstab of (#cells) clusters × supervised.
- overall_per_cluster_stats.csv    : Size, purity, entropy per cluster.
- overall_metrics.json             : ARI/AMI/NMI/H/C/V/FMI + optimal mapping.
- overall_contingency.png          : Heatmap (if --plot is set).

What the metrics mean (short version)
-------------------------------------
- ARI/AMI/NMI/Homogeneity/Completeness/V-measure/Fowlkes–Mallows:
  standard clustering-vs-label agreement metrics (all chance-adjusted or
  normalized appropriately).
- Optimal mapping accuracy:
  Accuracy after mapping clusters to classes with a one-to-one assignment
  (Hungarian algorithm) that maximizes the number of “correct” cells.
  This is useful when you want a pragmatic cluster→class mapping.

Assumptions & Constraints
-------------------------
- No model inference is performed; we only *read* the labels from GeoJSON.
- Rows missing either `model_class_id` or `supervised_type`/`anet_class_label`
  are skipped.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn import metrics

# Optional plotting (enabled by --plot)
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#                                 I/O Helpers                                  #
# --------------------------------------------------------------------------- #

def _read_geojson(path: str | Path) -> list[dict[str, Any]]:
    """
    Read a single GeoJSON FeatureCollection file and return its features.

    Parameters
    ----------
    path
        Path to a GeoJSON file with top-level "type": "FeatureCollection".

    Returns
    -------
    list[dict[str, Any]]
        A list of GeoJSON Feature dictionaries.

    Raises
    ------
    ValueError
        If the file is not a valid FeatureCollection or 'features' is malformed.

    Notes
    -----
    - We tolerate 'features' or 'Features' as the container key.
    - The function does *not* validate geometry; we only need properties.
    """
    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    if data.get("type") != "FeatureCollection":
        raise ValueError(f"{path} is not a GeoJSON FeatureCollection.")

    feats = data.get("features") or data.get("Features") or []
    if not isinstance(feats, list):
        raise ValueError(f"{path} has an unexpected 'features' structure.")
    return feats


def _extract_records(features: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of GeoJSON Features into a tidy DataFrame for evaluation.

    Each Feature is expected to have a `properties` mapping with:
      - model_class_id      : required (cluster id)
      - supervised_type     : preferred (ground-truth class)
      - id, model_label, slide : optional

    Parameters
    ----------
    features
        List of GeoJSON Features (dicts).

    Returns
    -------
    pd.DataFrame
        Columns:
            cell_id        : str | None
            cluster_id     : int | str
            cluster_label  : str | None
            supervised     : int | str
            slide          : str | None

    Raises
    ------
    ValueError
        If no valid rows (with both cluster_id and supervised) are found.

    Design choices
    --------------
    - Rows missing *either* cluster_id or supervised are dropped silently to
      avoid breaking the whole run due to a few malformed features.
    - No type coercion is forced on labels; we encode them later.
    """
    rows: list[dict[str, Any]] = []

    for feat in features:
        props: dict[str, Any] = feat.get("properties", {})

        cluster_id = props.get("model_class_id")
        supervised = props.get("supervised_type")

        if cluster_id is None or supervised is None:
            # Skip malformed or incomplete entries
            logger.warning(f"Skipping feature with missing fields for cell ID {props.get('id')}")
            logger.debug(f"Feature properties skipped: {feat}")
            continue

        rows.append(
            {
                "cell_id": props.get("id"),
                "cluster_id": cluster_id,
                "cluster_label": props.get("model_label"),
                "supervised": supervised,
                "slide": props.get("slide"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No valid records found. Check your input schema/keys.")
    return df


def load_inputs(paths: list[str | Path]) -> pd.DataFrame:
    """
    Load and combine one or more GeoJSON FeatureCollections into a single DataFrame.

    Parameters
    ----------
    paths
        Paths to GeoJSON FeatureCollections.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with the evaluation-ready columns and an
        additional `source_file` column indicating the origin file of each row.

    Example
    -------
    >>> df = load_inputs(["a.geojson", "b.geojson"])
    >>> df.columns
    Index(['cell_id','cluster_id','cluster_label','supervised','slide','source_file'], dtype='object')
    """
    parts: list[pd.DataFrame] = []
    for p in paths:
        feats = _read_geojson(p)
        df = _extract_records(feats)
        df["source_file"] = str(p)
        parts.append(df)

    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- #
#                           Tables & Metric Utilities                          #
# --------------------------------------------------------------------------- #

def contingency_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the contingency table (#cells) with rows=cluster_id and cols=supervised.

    Parameters
    ----------
    df
        DataFrame with columns 'cluster_id' and 'supervised'.

    Returns
    -------
    pd.DataFrame
        Crosstab (clusters x supervised) with integer counts.

    Notes
    -----
    - The values represent the number of cells that belong to a given
      (cluster_id, supervised) pair.
    """
    table = pd.crosstab(df["cluster_id"], df["supervised"], dropna=False)
    table.index.name = "cluster_id"
    table.columns.name = "supervised"
    return table


def _entropy(counts: np.ndarray) -> float:
    """
    Compute Shannon entropy (in nats) for a vector of non-negative counts.

    Parameters
    ----------
    counts
        1D array of counts (non-negative).

    Returns
    -------
    float
        Entropy in natural log base; 0.0 if the vector is empty or sums to 0.

    Intuition
    ---------
    - Entropy is low when a cluster is dominated by a single class (pure),
      and high when it is uniformly mixed across classes.
    """
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def per_cluster_stats(ct: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-cluster summary statistics from a contingency table.

    For each cluster (row), compute:
      - size                : total number of cells in the cluster
      - top_supervised      : supervised class with maximum count in the cluster
      - top_count           : count of that majority class
      - purity              : top_count / size ∈ [0, 1]
      - entropy             : Shannon entropy (nats)
      - normalized_entropy  : entropy / log(#classes) ∈ [0, 1]

    Parameters
    ----------
    ct
        Contingency table (clusters x supervised) with counts.

    Returns
    -------
    pd.DataFrame
        One row per cluster_id, sorted by 'size' descending.

    Caveats
    -------
    - If a cluster row is all zeros (shouldn't happen if built from crosstab),
      the stats default to zeros/None appropriately.
    """
    rows: list[dict[str, Any]] = []
    n_classes = ct.shape[1]
    norm = math.log(n_classes) if n_classes > 1 else 1.0

    for cluster_id, row in ct.iterrows():
        counts = row.values.astype(int)
        size = int(counts.sum())

        if size == 0:
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "size": 0,
                    "top_supervised": None,
                    "top_count": 0,
                    "purity": 0.0,
                    "entropy": 0.0,
                    "normalized_entropy": 0.0,
                }
            )
            continue

        top_idx = int(counts.argmax())
        top_count = int(counts[top_idx])
        ent = _entropy(counts)

        rows.append(
            {
                "cluster_id": cluster_id,
                "size": size,
                "top_supervised": ct.columns[top_idx],
                "top_count": top_count,
                "purity": top_count / size,
                "entropy": ent,
                "normalized_entropy": float(ent / norm if norm > 0 else 0.0),
            }
        )

    return pd.DataFrame(rows).sort_values(by="size", ascending=False)


def _encode_labels(series: pd.Series) -> tuple[np.ndarray, dict[Any, int]]:
    """
    Encode arbitrary labels (strings/ints) into contiguous integer IDs [0..K-1].

    Parameters
    ----------
    series
        A pandas Series of labels (e.g., cluster_id or supervised).

    Returns
    -------
    tuple[np.ndarray, dict[Any, int]]
        encoded_labels, mapping where mapping: original_label -> encoded_int

    Why do this?
    ------------
    Many sklearn metrics expect or operate faster with integer-coded labels.
    This function preserves the original label set via the returned mapping.
    """
    cats = pd.Index(series.astype("category").cat.categories)
    mapping = {v: i for i, v in enumerate(cats)}
    encoded = series.map(mapping).to_numpy()
    return encoded, mapping


def clustering_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute standard clustering-vs-label metrics.

    Parameters
    ----------
    y_true
        Ground-truth labels encoded as ints [0..K-1].
    y_pred
        Cluster labels encoded as ints [0..M-1].

    Returns
    -------
    dict[str, float]
        {
            "adjusted_rand_index",
            "adjusted_mutual_info",
            "normalized_mutual_info",
            "homogeneity",
            "completeness",
            "v_measure",
            "fowlkes_mallows",
        }

    Notes
    -----
    - ARI is chance-adjusted; 1.0 means perfect agreement, ~0.0 ≈ random.
    - AMI/NMI measure shared information (normalized).
    - Homogeneity/Completeness/V-measure describe different aspects of
      label consistency and coverage.
    """
    return {
        "adjusted_rand_index": float(metrics.adjusted_rand_score(y_true, y_pred)),
        "adjusted_mutual_info": float(
            metrics.adjusted_mutual_info_score(y_true, y_pred, average_method="arithmetic")
        ),
        "normalized_mutual_info": float(
            metrics.normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")
        ),
        "homogeneity": float(metrics.homogeneity_score(y_true, y_pred)),
        "completeness": float(metrics.completeness_score(y_true, y_pred)),
        "v_measure": float(metrics.v_measure_score(y_true, y_pred)),
        "fowlkes_mallows": float(metrics.fowlkes_mallows_score(y_true, y_pred)),
    }


def optimal_mapping_accuracy(ct: pd.DataFrame) -> tuple[float, dict[int, int]]:
    """
    Compute the best 1-1 cluster→class mapping (Hungarian algorithm)
    and return the resulting accuracy.

    Intuition
    ---------
    If you later want to “name” clusters using the closest supervised classes,
    you want a one-to-one mapping that maximizes correct assignments. That's
    exactly the linear assignment (Hungarian) solution over the contingency.

    Parameters
    ----------
    ct
        Contingency table (clusters x supervised) of counts.

    Returns
    -------
    tuple[float, dict[int, int]]
        acc, mapping
        - acc: accuracy in [0, 1] after applying the optimal mapping
        - mapping: row_idx -> col_idx (indices into ct.index / ct.columns)

    Implementation detail
    ---------------------
    The Hungarian method *minimizes* cost, so we negate counts to maximize
    the sum of assigned counts. We also pad the matrix to square shape if
    needed (extra rows/cols are zeros and ignored in the final mapping).
    """
    counts = ct.values.astype(int)
    n_rows, n_cols = counts.shape
    dim = max(n_rows, n_cols)

    # Pad to square for the solver (safe no-op if already square)
    padded = np.zeros((dim, dim), dtype=int)
    padded[:n_rows, :n_cols] = counts

    # Hungarian on negative counts = maximize original counts
    row_ind, col_ind = linear_sum_assignment(-padded)

    correct = 0
    for r, c in zip(row_ind, col_ind):
        if r < n_rows and c < n_cols:
            correct += counts[r, c]

    total = int(counts.sum())
    acc = float(correct) / float(total) if total > 0 else 0.0

    # Only keep mappings within the original (non-padded) dimensions
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind) if (r < n_rows and c < n_cols)}
    return acc, mapping


# --------------------------------------------------------------------------- #
#                                   Plotting                                  #
# --------------------------------------------------------------------------- #

def _plot_heatmap(ct: pd.DataFrame, outpath: str | Path, title: str) -> None:
    """
    Save a simple heatmap of the contingency table.

    Parameters
    ----------
    ct
        Contingency table (clusters × supervised).
    outpath
        Output image path (.png).
    title
        Title to display on the figure.

    Notes
    -----
    - Uses only matplotlib (no seaborn) to minimize deps and keep consistent
      with restricted environments.
    """
    # Figure size scales mildly with table size for legibility
    fig, ax = plt.subplots(figsize=(max(6, ct.shape[1] * 0.6), max(4, ct.shape[0] * 0.6)))

    im = ax.imshow(ct.values, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("supervised")
    ax.set_ylabel("cluster_id")

    # Tick labels for both dims
    ax.set_xticks(range(ct.shape[1]))
    ax.set_yticks(range(ct.shape[0]))
    ax.set_xticklabels([str(c) for c in ct.columns], rotation=90)
    ax.set_yticklabels([str(i) for i in ct.index])

    # Colorbar with units
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("# cells")

    fig.tight_layout()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#                               Orchestration                                 #
# --------------------------------------------------------------------------- #

def evaluate(df: pd.DataFrame, outdir: str | Path, plot: bool = False) -> dict[str, Any]:
    """
    Run the full overall evaluation, persist artifacts, and return a JSON report.

    Parameters
    ----------
    df
        DataFrame with columns: cluster_id, supervised (slide optional).
    outdir
        Directory where outputs will be written (created if missing).
    plot
        If True, also write a PNG heatmap of the contingency table.

    Returns
    -------
    dict[str, Any]
        JSON-serializable report containing:
            counts, n_clusters, n_supervised_classes,
            metrics (dict),
            optimal_cluster_to_supervised_mapping (human-readable dict)

    Files written
    -------------
    - contingency_table.csv
    - per_cluster_stats.csv
    - metrics.json
    - contingency_map.png (if plot=True)

    Side effects
    ------------
    Creates `outdir` if it does not exist.

    Example
    -------
    >>> df = load_inputs(["cells.geojson"])
    >>> report = evaluate(df, outdir="outputs/eval", prefix="overall", plot=True)
    >>> report["metrics"]["adjusted_rand_index"]
    0.73
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Contingency table (clusters × supervised)
    ct = contingency_table(df)
    ct.to_csv(outdir / f"contingency_table.csv")

    # 2) Per-cluster statistics (purity, entropy, etc.)
    pcs = per_cluster_stats(ct)
    pcs.to_csv(outdir / f"per_cluster_stats.csv", index=False)

    # 3) Encode labels for metric computation
    y_true_enc, true_map = _encode_labels(df["supervised"])
    y_pred_enc, pred_map = _encode_labels(df["cluster_id"])

    # 4) Global clustering metrics
    metr = clustering_metrics(y_true_enc, y_pred_enc)

    # 5) Optimal 1–1 mapping accuracy via Hungarian assignment
    acc, mapping_idx = optimal_mapping_accuracy(ct)
    metr["optimal_mapping_accuracy"] = acc

    # 6) Human-readable mapping (original labels, not encoded ints)
    idx_to_cluster = {v: k for k, v in pred_map.items()}
    idx_to_super = {v: k for k, v in true_map.items()}
    human_mapping = {str(idx_to_cluster[i]): str(idx_to_super[j]) for i, j in mapping_idx.items()}

    report: dict[str, Any] = {
        "counts": int(len(df)),
        "n_clusters": int(len(ct.index)),
        "n_supervised_classes": int(len(ct.columns)),
        "metrics": metr,
        "optimal_cluster_to_supervised_mapping": human_mapping,
    }

    with open(outdir / f"metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # 7) Optional visualization
    if plot:
        _plot_heatmap(ct, outdir / f"contingency_map.png", title=f"Contingency Map")

    return report


# --------------------------------------------------------------------------- #
#                                    CLI                                      #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the overall-only evaluation script.

    Returns
    -------
    argparse.Namespace
        - input : list[str] of GeoJSON paths
        - outdir: str output directory for artifacts
        - plot  : bool flag to enable PNG heatmap output

    Example
    -------
    $ python -m src.eval.cluster_eval_overall \\
        --input data/cells.geojson \\
        --outdir outputs/eval \\
        --plot
    """
    p = argparse.ArgumentParser(
        description="Evaluate clusters vs. supervised labels from GeoJSON."
    )
    p.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Path(s) to GeoJSON FeatureCollection file(s).",
    )
    p.add_argument(
        "--outdir",
        required=True,
        help="Output directory to write reports (created if missing).",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Save a contingency heatmap (PNG).",
    )
    return p.parse_args()


def main() -> None:
    """
    CLI entry point:
    - Loads GeoJSON(s),
    - Runs overall evaluation,
    - Prints a concise metric summary and the optimal cluster→class mapping.
    """
    args = parse_args()
    df = load_inputs(args.input)

    overall = evaluate(df, outdir=args.outdir, plot=args.plot)

    # Console summary (quick human check without opening files)
    print("\n=== Overall Metrics ===")
    for k, v in overall["metrics"].items():
        # Robust formatting for floats; don't crash on non-floats
        try:
            print(f"{k:>28s}: {float(v):.4f}")
        except Exception:
            print(f"{k:>28s}: {v}")

    print("\nOptimal cluster→class mapping:")
    for c, s in overall["optimal_cluster_to_supervised_mapping"].items():
        print(f"  cluster {c} -> class {s}")


if __name__ == "__main__":
    main()
