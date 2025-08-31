#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geojson_export.py

Export clustered cells to a GeoJSON FeatureCollection with this schema:

{
  "type": "Feature",
  "geometry": { "type": "Polygon", "coordinates": [ ... ] },  # WSI coords
  "properties": {
    "id": "unique_cell_id",
    "model_label": "Cluster X",
    "model_class_id": X,
    "model_magnification": 40,
    "class_type": "object",
    "model_name": "HibouLCellVIT",
    "model_version": "1.0",
    "anet_class_label": "Cluster X",
    "slide": "slide_name.svs"
  }
}

Usage:
  python scripts/geojson_export.py \
    --input outputs/run1_clustered/cells_clustered.parquet \
    --output outputs/run1_clustered/cells.geojson \
    --slide slide_name.svs \
    --model-name HibouLCellVIT \
    --model-version 1.0 \
    --model-magnification 40 \
    [--contours outputs/run1/contours.jsonl]
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import logging


# -------- IO helpers --------

def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def load_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    if ext == ".json":
        return pd.read_json(path, orient="records")
    # fallback
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path)

def load_contours_sidecar(path):
    """
    Sidecar must contain rows with: tile_i, tile_j, cell_id, contour (list of [x,y], tile coords).
    Returns dict[(tile_i, tile_j, cell_id)] -> np.ndarray(N,2) float32
    """
    if path is None:
        return {}
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jsonl", ".ndjson"):
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_json(path, orient="records")
    lut = {}
    for _, r in df.iterrows():
        key = (int(r["tile_i"]), int(r["tile_j"]), int(r["cell_id"]))
        cnt = np.asarray(r["contour"], dtype=np.float32)
        lut[key] = cnt
    return lut


# -------- Geometry helpers --------

def ringify(contour_xy):
    """
    Close polygon ring if needed.
    Input: np.ndarray (N,2) in WSI coords.
    Output: list[[x,y],...], closed.
    """
    arr = np.asarray(contour_xy, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
        logging.warning("Contour must have shape (N,2) with N>=3, got shape %s", arr.shape)
        return []
    if not np.allclose(arr[0], arr[-1]):
        arr = np.vstack([arr, arr[0]])
    return [[float(x), float(y)] for x, y in arr]


# -------- Feature builder (STRICT schema) --------

def make_feature_from_row(row, contour_tile_xy, slide_name, model_name, model_version, model_magnification):
    """
    Build a Feature exactly matching the required schema.
    - contour_tile_xy is in TILE coordinates; we shift to WSI coords by (x,y).
    """
    if contour_tile_xy is None or len(contour_tile_xy) == 0:
        return None  # No geometry, skip

    # Required fields from row
    x0, y0 = float(row["x"]), float(row["y"])             # tile origin in WSI coords
    tile_i, tile_j = int(row["tile_i"]), int(row["tile_j"])
    cid = int(row["cell_id"])
    class_id = int(row["cluster_id"])
    class_label = str(row.get("cluster_label", f"Cluster {class_id}"))

    # Shift contour to WSI coords
    cnt = np.asarray(contour_tile_xy, dtype=np.float64).copy()
    cnt[:, 0] += x0
    cnt[:, 1] += y0

    ring = ringify(cnt)
    if not ring:
        return None

    geom = { "type": "Polygon", "coordinates": [ring] }

    # Build properties
    props = {
        "id": f"{slide_name}:{tile_i}-{tile_j}-{cid}",   # unique_cell_id
        "model_label": class_label,                      # "Cluster X"
        "model_class_id": class_id,                      # X
        "model_magnification": int(model_magnification),
        "class_type": "object",
        "model_name": str(model_name),
        "model_version": str(model_version),
        "anet_class_label": class_label,                 # same as model_label
        "slide": str(slide_name),
        "supervised_cluster_id": row.get("type", None)  # Added the supervised cluster id for later analysis
    }

    return {
        "type": "Feature",
        "geometry": geom,
        "properties": props,
    }


# -------- Main --------

def main():
    ap = argparse.ArgumentParser(description="Export clustered cells to GeoJSON (strict schema).")
    ap.add_argument("--input", required=True, help="Clustered table (Parquet/CSV/JSONL/JSON).")
    ap.add_argument("--output", required=True, help="Output GeoJSON path.")
    ap.add_argument("--slide", required=True, help="Slide filename (e.g., slide_name.svs).")
    ap.add_argument("--model-name", default="HibouLCellVIT", help="model_name property.")
    ap.add_argument("--model-version", default="1.0", help="model_version property.")
    ap.add_argument("--model-magnification", type=int, default=40, help="model_magnification property.")
    ap.add_argument("--contours", default=None,
                    help="Optional sidecar (JSONL/JSON) with tile_i,tile_j,cell_id,contour (tile coords).")
    args = ap.parse_args()

    df = load_table(args.input)
    lut = {} if ("contour" in df.columns) else load_contours_sidecar(args.contours)

    features = []
    missing = 0

    for _, row in df.iterrows():
        key = (int(row["tile_i"]), int(row["tile_j"]), int(row["cell_id"]))

        # Prefer inline contour if present; else sidecar
        contour_tile = None
        if "contour" in df.columns and pd.notna(row["contour"]):
            val = row["contour"]
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except Exception:
                    val = None
            if val is not None:
                contour_tile = np.asarray(val, dtype=np.float32)
        if contour_tile is None:
            contour_tile = lut.get(key, None)

        feat = make_feature_from_row(
            row=row,
            contour_tile_xy=contour_tile,
            slide_name=args.slide,
            model_name=args.model_name,
            model_version=args.model_version,
            model_magnification=args.model_magnification,
        )
        if feat is None:
            missing += 1
            continue
        features.append(feat)

    fc = { "type": "FeatureCollection", "features": features }

    ensure_dir(args.output)
    with open(args.output, "w") as f:
        json.dump(fc, f, indent=2)

    logging.info(f"[done] wrote GeoJSON with {len(features)} features to: {args.output}")
    if missing:
        logging.warning(f"[warn] skipped {missing} rows with missing/invalid contours")

if __name__ == "__main__":
    main()
