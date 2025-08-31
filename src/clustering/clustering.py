#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clustering.py

Cluster per-cell embeddings into phenotypes.

Input: a Parquet or CSV produced after embedding extraction, with rows like:
  tile_i, tile_j, x, y, w, h, cell_id, e_0, e_1, ..., e_{D-1}, [type], [type_prob], ...

Usage (fixed K=6):
    python scripts/clustering.py \
      --input outputs/run1/cells.parquet \
      --output outputs/run1_clustered/cells_clustered.parquet \
      --algo kmeans --k 6 --pca 50 --umap 2 --min-conf 0.0

Usage (density-based):
    python scripts/clustering.py \
      --input outputs/run1/cells.parquet \
      --output outputs/run1_clustered/cells_clustered.parquet \
      --algo hdbscan --min-samples 20 --min-cluster-size 50 \
      --pca 50 --umap 2
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import hdbscan
import umap

from src.logging_config import configure_logging
import logging

configure_logging()
logger = logging.getLogger(__name__)


# ------------------------- Utilities -------------------------

def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def find_embedding_columns(df):
    """Detect embedding columns named e_0, e_1, ... (sorted numerically)."""
    cols = [c for c in df.columns if c.startswith("e_")]
    if not cols:
        raise RuntimeError("No embedding columns found (expected e_0, e_1, ...).")
    return sorted(cols, key=lambda c: int(c.split("_", 1)[1]))

def standardize(X, with_mean=True, with_std=True):
    scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
    return scaler.fit_transform(X), scaler

def run_pca(X, n_components):
    if n_components <= 0 or n_components >= X.shape[1]:
        return X, None
    p = PCA(n_components=n_components, svd_solver="auto", random_state=42)
    return p.fit_transform(X), p

def run_umap(X, n_components):
    if n_components <= 0:
        return None, None
    reducer = umap.UMAP(n_components=n_components, random_state=42, metric="euclidean")
    return reducer.fit_transform(X), reducer

def remap_to_consecutive(labels, start=1, noise_label=None):
    """
    Remap arbitrary integer labels to {start, start+1, ...} by cluster size (desc).
    Noise label (e.g., -1) is kept as-is if provided.
    """
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    # order non-noise by frequency
    order = [int(u) for u, c in sorted(zip(unique, counts), key=lambda x: -x[1])
             if (noise_label is None or u != noise_label)]
    mapping = {}
    nxt = start
    for u in order:
        mapping[u] = nxt
        nxt += 1
    if noise_label is not None and (noise_label in unique):
        mapping[noise_label] = noise_label
    remapped = np.array([mapping.get(int(v), int(v)) for v in labels], dtype=int)
    return remapped, mapping

def plot_umap_scatter(X2, labels, out_png, title):
    ensure_dir(out_png)
    plt.figure(figsize=(7, 6))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=4, alpha=0.85)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def save_cluster_centroids(output_dir, centroids):
    if centroids is None:
        return
    ensure_dir(os.path.join(output_dir, "centroids.json"))
    with open(os.path.join(output_dir, "centroids.json"), "w") as f:
        json.dump({"centroids": centroids.tolist()}, f, indent=2)


# ------------------------- Clustering runners -------------------------

def cluster_kmeans(X, k, seed=42):
    """KMeans with fixed K. Returns labels (1..K) and centroids."""
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    y = km.fit_predict(X)
    y_remap, _ = remap_to_consecutive(y, start=1)
    return y_remap, km.cluster_centers_

def cluster_hdbscan(X, min_samples=10, min_cluster_size=50):
    """HDBSCAN density-based clustering. Labels with noise = -1. Centroids undefined."""
    clusterer = hdbscan.HDBSCAN(min_samples=int(min_samples),
                                min_cluster_size=int(min_cluster_size),
                                metric="euclidean",
                                core_dist_n_jobs=1)
    y = clusterer.fit_predict(X)
    y_remap, _ = remap_to_consecutive(y, start=1, noise_label=-1)
    return y_remap, None



# ------------------------- Public in-memory APIs -------------------------

def run_clustering(
    df,
    algo="kmeans",
    k=6,
    pca=50,
    standardize=True,
    hdbscan_min_samples=10,
    hdbscan_min_cluster_size=50,
    umap=0,
    output=None
):
    """
    Cluster a DataFrame of cell rows in memory, adding cluster columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain embedding columns named e_0..e_{D-1}.
    algo : {"kmeans", "hdbscan"}
        Clustering algorithm. Use "kmeans" if you need exactly K clusters.
    k : int
        Number of clusters for KMeans.
    pca : int
        PCA dimensionality (0 to skip).
    standardize : bool
        If True, z-score standardization before PCA/clustering.
    hdbscan_min_samples : int
    hdbscan_min_cluster_size : int
    umap : int
        UMAP dimensionality (0 to skip).
    output : str
        Path to save output files.

    Returns
    -------
    df_out : pandas.DataFrame
        Copy of input with added:
          - cluster_id (int)
          - cluster_label (str)
    """
    df = df.copy()

    # Embedding matrix
    emb_cols = find_embedding_columns(df)
    X = df[emb_cols].to_numpy(dtype=np.float32)

    # Standardize + PCA
    Xs, _ = standardize(X, with_mean=True, with_std=True)
    Xp, _ = run_pca(Xs, n_components=int(pca))
    
    # Cluster
    if algo == "kmeans":
        labels, centroids = cluster_kmeans(Xp, k=int(k))
    elif algo == "hdbscan":
        labels, centroids = cluster_hdbscan(Xp, min_samples=int(hdbscan_min_samples), min_cluster_size=int(hdbscan_min_cluster_size))
    else:
        raise ValueError(f"Unknown algo: {algo}")

    df["cluster_id"] = labels
    df["cluster_label"] = df["cluster_id"].apply(lambda v: f"Cluster {v}" if v > 0 else "Noise")


    # Saving 
    if output:

        save_dir = os.path.splitext(output)[0] + "_assets"

        # UMAP (diagnostic)
        if umap > 0:
            Xu, _ = run_umap(Xp, n_components=int(umap))

            os.makedirs(save_dir, exist_ok=True)
            if Xu is not None:
                plot_umap_scatter(Xu, np.zeros(len(Xu)), os.path.join(save_dir, "umap_raw.png"), "UMAP (uncolored)")

            # UMAP colored
            if Xu is not None:
                plot_umap_scatter(Xu, labels, os.path.join(save_dir, "umap_clusters.png"), "UMAP (colored by cluster)")



        # Save centroids (KMeans)
        save_cluster_centroids(save_dir, centroids)

        # Save dataframe 
        ensure_dir(output)
        out_is_csv = output.lower().endswith(".csv")
        if out_is_csv:
            out_path = output if output.lower().endswith(".csv") else output + ".csv"
            df.to_csv(out_path, index=False)
        else:
            out_path = output if output.lower().endswith(".parquet") else output + ".parquet"
            df.to_parquet(out_path, index=False)

        # Save metadata
        meta = {
            "algo": algo,
            "k": int(k),
            "min_samples": int(hdbscan_min_samples),
            "min_cluster_size": int(hdbscan_min_cluster_size),
            "pca": int(pca),
            "umap": int(umap),
            "n_rows": int(len(df)),
            "n_embed_dims": int(len(emb_cols)),
        }
        with open(os.path.join(save_dir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Wrote clustered table to: {out_path}")
        logger.info(f"Diagnostics in: {save_dir}")


    return df


def cluster_rows(
    rows,
    **kwargs,
):
    """
    Convenience wrapper: accept a list of dict rows (like build_rows_for_saving output),
    turn into a DataFrame, and call cluster_df_in_memory(...).

    Parameters
    ----------
    rows : list[dict]
        One row per cell. Must include embedding columns e_0..e_{D-1}.
    kwargs : forwarded to cluster_df_in_memory

    Returns
    -------
    df_out : pandas.DataFrame
    result : dict
    """
    if not rows:
        raise RuntimeError("Empty 'rows' list; nothing to cluster.")
    df = pd.DataFrame(rows)
    return run_clustering(df, **kwargs)


# ------------------------- Main pipeline -------------------------

def main():
    ap = argparse.ArgumentParser(description="Cluster per-cell embeddings.")
    ap.add_argument("--input", required=True, help="Input Parquet/CSV with e_* columns.")
    ap.add_argument("--output", required=True, help="Output Parquet/CSV with cluster labels added.")
    ap.add_argument("--algo", choices=["kmeans", "hdbscan"], default="kmeans")
    ap.add_argument("--k", type=int, default=6, help="K for KMeans (default: 6).")
    ap.add_argument("--min-samples", type=int, default=10, help="HDBSCAN min_samples.")
    ap.add_argument("--min-cluster-size", type=int, default=50, help="HDBSCAN min_cluster_size.")
    ap.add_argument("--pca", type=int, default=50, help="PCA components (0 to skip).")
    ap.add_argument("--umap", type=int, default=0, help="UMAP components for viz (0 to skip; typically 2).")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="Drop cells with type_prob below this threshold (0 disables).")
    ap.add_argument("--standardize", action="store_true",
                    help="Standardize features before clustering.")
    args = ap.parse_args()

    # Load
    ext = os.path.splitext(args.input)[1].lower()
    if ext in [".parquet", ".pq"]:
        df = pd.read_parquet(args.input)
    elif ext == ".csv":
        df = pd.read_csv(args.input)
    else:
        # prefer parquet; fallback csv
        try:
            df = pd.read_parquet(args.input)
        except Exception:
            df = pd.read_csv(args.input)

    # Filter by confidence
    if args.min_conf > 0 and "type_prob" in df.columns:
        df = df[df["type_prob"].astype(float) >= float(args.min_conf)].copy()

    run_clustering( df, 
                    algo=args.algo, 
                    k=args.k, 
                    pca=args.pca, 
                    standardize=args.standardize, 
                    hdbscan_min_samples=args.min_samples, 
                    hdbscan_min_cluster_size=args.min_cluster_size,
                    umap=args.umap,
                    output=args.output
                    )


if __name__ == "__main__":
    main()
