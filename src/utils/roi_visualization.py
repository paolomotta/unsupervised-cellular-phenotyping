#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROI-based visualizations over the WSI background.

Given:
  - An .svs Whole Slide Image (WSI)
  - An ROI.geojson polygon (in WSI level-0 pixel coordinates)
  - A cells.geojson FeatureCollection with cell polygons in the same coord space,
    and properties including:
      * model_class_id   (unsupervised cluster ID)   [REQUIRED]
      * supervised_type  (supervised class label)    [REQUIRED; no fallback]

This script produces two PNG figures for the ROI:
  1) Cells colored by unsupervised clusters (model_class_id)
  2) Cells colored by supervised classes (supervised_type)

Design goals:
  - Robust coordinate handling: transform level-0 polygons → downsampled region pixels.
  - Readable output over the WSI region (semi-transparent overlays).
  - Minimal CLI; logging via src.logging_config.
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import openslide
import pandas as pd
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, shape
from shapely.geometry.base import BaseGeometry

from src.logging_config import configure_logging
import logging

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def load_roi(roi_path: str | Path) -> BaseGeometry:
    """
    Load ROI.geojson (polygon/multipolygon) and return a single merged geometry.

    Parameters
    ----------
    roi_path : str | Path
        Path to the ROI GeoJSON.

    Returns
    -------
    BaseGeometry
        The unary union of all ROI geometries.

    Raises
    ------
    ValueError
        If the file is empty or not polygonal.
    """
    roi_gdf = gpd.read_file(roi_path)
    if roi_gdf.empty:
        raise ValueError("ROI file has no geometries.")

    roi_union = roi_gdf.unary_union
    if not isinstance(roi_union, (Polygon, MultiPolygon)):
        raise ValueError("ROI geometry is not a Polygon/MultiPolygon.")
    return roi_union


def load_cells(cells_path: str | Path) -> gpd.GeoDataFrame:
    """
    Load cells GeoJSON and validate required columns.

    Required columns:
      - model_class_id
      - supervised_type

    Parameters
    ----------
    cells_path : str | Path
        Path to the cells FeatureCollection GeoJSON.

    Returns
    -------
    GeoDataFrame

    Raises
    ------
    KeyError
        If required properties are missing.
    """
    gdf = gpd.read_file(cells_path)

    # Validate cluster column
    if "model_class_id" not in gdf.columns:
        raise KeyError("Missing 'model_class_id' in cells GeoJSON properties.")

    # Validate supervised column (NO fallback to anet_class_label)
    if "supervised_type" not in gdf.columns:
        raise KeyError("Missing 'supervised_type' in cells GeoJSON properties (no fallback allowed).")

    return gdf


# -----------------------------------------------------------------------------
# WSI region reading and geometry transforms
# -----------------------------------------------------------------------------

def read_region_for_roi(
    svs_path: str | Path,
    roi_geom: BaseGeometry,
    target_max_px: int = 4096,
) -> tuple[Any, float, float, int, int, int, float]:
    """
    Read a downsampled WSI image region that tightly covers the ROI.

    Parameters
    ----------
    svs_path : str | Path
        Path to the .svs file.
    roi_geom : shapely BaseGeometry
        ROI polygon(s) in level-0 coordinates.
    target_max_px : int
        Max dimension (in pixels) for the returned region to keep memory in check.

    Returns
    -------
    (img, sx, sy, x0, y0, level, downsample)
        img         : PIL.Image (RGB) of the region
        sx, sy      : scale factors from level-0 pixels → img pixels (both = 1/downsample)
        x0, y0      : top-left of the ROI bounding box in level-0 pixels
        level       : chosen OpenSlide level
        downsample  : level downsample factor (float)
    """
    slide = openslide.OpenSlide(str(svs_path))

    minx, miny, maxx, maxy = roi_geom.bounds  # in level-0 pixels
    x0, y0 = int(minx), int(miny)
    w0, h0 = int(maxx - minx), int(maxy - miny)

    # Choose a level such that the region fits within target_max_px
    chosen_level = 0
    for lv in range(slide.level_count):
        ds = float(slide.level_downsamples[lv])
        w_lv, h_lv = int(np.ceil(w0 / ds)), int(np.ceil(h0 / ds))
        if max(w_lv, h_lv) <= target_max_px:
            chosen_level = lv
            break
        chosen_level = lv  # fallback to deepest level if none smaller

    downsample = float(slide.level_downsamples[chosen_level])
    w_lv, h_lv = int(np.ceil(w0 / downsample)), int(np.ceil(h0 / downsample))
    img = slide.read_region((x0, y0), chosen_level, (w_lv, h_lv)).convert("RGB")
    slide.close()

    sx = sy = 1.0 / downsample
    return img, sx, sy, x0, y0, chosen_level, downsample


def transform_level0_to_region(
    geoms: gpd.GeoSeries,
    x0: int,
    y0: int,
    sx: float,
    sy: float,
) -> gpd.GeoSeries:
    """
    Convert level-0 coordinates to the pixel space of a downsampled region image.

    Steps:
      1) translate by (-x0, -y0) to move ROI top-left to (0,0)
      2) scale by (sx, sy) = (1/downsample, 1/downsample)

    Parameters
    ----------
    geoms : GeoSeries
        Geometries in level-0 pixel coordinates.
    x0, y0 : int
        Top-left of the region in level-0 coordinates.
    sx, sy : float
        Scale factors from level-0 → region pixels.

    Returns
    -------
    GeoSeries
        Transformed geometries in region pixel space.
    """
    return geoms.apply(
        lambda g: affinity.scale(
            affinity.translate(g, xoff=-x0, yoff=-y0),
            xfact=sx, yfact=sy, origin=(0, 0)
        )
    )


# -----------------------------------------------------------------------------
# ROI filtering / clipping
# -----------------------------------------------------------------------------

def subset_and_clip_to_roi(
    cells: gpd.GeoDataFrame,
    roi_geom: BaseGeometry,
    simplify_tolerance: float | None = None,
) -> gpd.GeoDataFrame:
    """
    Keep only cells intersecting the ROI and clip their polygons to the ROI.

    Parameters
    ----------
    cells : GeoDataFrame
        Full cells table (level-0 coords).
    roi_geom : BaseGeometry
        ROI polygon/multipolygon (level-0 coords).
    simplify_tolerance : float | None
        Optional simplify tolerance (in pixels) to speed up plotting.

    Returns
    -------
    GeoDataFrame
        Cells trimmed to the ROI.
    """
    # Fast spatial filter
    mask = cells.geometry.intersects(roi_geom)
    cells_roi = cells.loc[mask].copy()

    # Optional simplification (visual only)
    if simplify_tolerance and simplify_tolerance > 0:
        cells_roi["geometry"] = cells_roi.geometry.simplify(
            simplify_tolerance, preserve_topology=True
        )

    # Clip to the ROI (clean edges)
    cells_roi = gpd.clip(cells_roi, roi_geom)

    return cells_roi


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _categoricalize(series: pd.Series) -> pd.Categorical:
    """Ensure stable categorical dtype for consistent coloring/legend."""
    return pd.Categorical(series.astype("string"), ordered=False)


def plot_cells_over_region_image(
    region_img,
    cells_px: gpd.GeoDataFrame,
    color_by: str,
    title: str,
    outfile: str | Path,
    cmap: str,
    alpha: float = 0.5,
) -> None:
    """
    Overlay polygons (already transformed to region pixel space) on the region image.

    Parameters
    ----------
    region_img : PIL.Image
        Background region image from the WSI.
    cells_px : GeoDataFrame
        Polygons in region pixel coordinates; must include `color_by` column.
    color_by : str
        Column to color by ('model_class_id' or 'supervised_type').
    title : str
        Plot title.
    outfile : str | Path
        Output PNG path.
    cmap : str
        Matplotlib colormap name for categorical mapping.
    alpha : float
        Polygon transparency (0..1).
    """
    if cells_px.empty:
        raise ValueError("No cells to plot after ROI filtering/clipping.")

    gdf = cells_px.copy()
    gdf[color_by] = _categoricalize(gdf[color_by])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(region_img)
    gdf.plot(
        column=color_by,
        categorical=True,
        legend=True,
        cmap=cmap,
        alpha=alpha,
        edgecolor="none",
        linewidth=0,
        ax=ax,
    )
    ax.set_title(title, fontsize=14)
    ax.axis("off")

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Reusable orchestration function
# -----------------------------------------------------------------------------

def generate_roi_visualizations(
    svs_path: str | Path,
    roi_path: str | Path,
    cells_path: str | Path,
    outdir: str | Path,
    simplify_tolerance: float = 0.0,
) -> tuple[Path, Path]:
    """
    Produce the two required ROI visualizations over the WSI background.

    Parameters
    ----------
    svs_path : str | Path
        Path to the .svs WSI.
    roi_path : str | Path
        Path to ROI.geojson (same coord space as cells).
    cells_path : str | Path
        Path to cells GeoJSON (FeatureCollection).
    outdir : str | Path
        Output directory for the PNG images.
    simplify_tolerance : float
        Optional polygon simplification tolerance (in pixels) for speed.

    Returns
    -------
    (clusters_png, supervised_png) : tuple[Path, Path]
        Paths to the generated figures.
    """
    svs_path = Path(svs_path)
    roi_path = Path(roi_path)
    cells_path = Path(cells_path)
    outdir = Path(outdir)

    logger.info("Loading ROI from %s", roi_path)
    roi_geom = load_roi(roi_path)

    logger.info("Loading cells from %s", cells_path)
    cells = load_cells(cells_path)

    logger.info("Filtering and clipping cells to ROI")
    cells_roi = subset_and_clip_to_roi(cells, roi_geom, simplify_tolerance=simplify_tolerance)

    logger.info("Reading WSI region covering ROI from %s", svs_path)
    region_img, sx, sy, x0, y0, level, downsample = read_region_for_roi(svs_path, roi_geom)

    logger.info("Transforming polygons to region pixel space (level=%d, ~%.0fx)", level, downsample)
    cells_px = cells_roi.copy()
    cells_px["geometry"] = transform_level0_to_region(cells_roi.geometry, x0, y0, sx, sy)

    # Output paths
    clusters_png = outdir / "roi_clusters_on_wsi.png"
    supervised_png = outdir / "roi_supervised_on_wsi.png"

    logger.info("Rendering clusters overlay → %s", clusters_png)
    plot_cells_over_region_image(
        region_img,
        cells_px,
        color_by="model_class_id",
        title=f"ROI over WSI - colored by unsupervised clusters",
        outfile=clusters_png,
        cmap="Dark2",
        alpha=0.5,
    )

    logger.info("Rendering supervised overlay → %s", supervised_png)
    plot_cells_over_region_image(
        region_img,
        cells_px,
        color_by="supervised_type",
        title=f"ROI over WSI - colored by supervised classes",
        outfile=supervised_png,
        cmap="Set1",
        alpha=0.5,
    )

    logger.info(f"Saved visualizations to {outdir}.")
    return clusters_png, supervised_png


# -----------------------------------------------------------------------------
# Minimal CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate ROI visualizations over the WSI background."
    )
    p.add_argument("--svs", required=True, help="Path to the input WSI (.svs).")
    p.add_argument("--roi", required=True, help="Path to ROI.geojson (WSI level-0 coords).")
    p.add_argument("--cells", required=True, help="Path to cells GeoJSON (FeatureCollection).")
    p.add_argument("--outdir", required=True, help="Directory to save the output figures.")
    p.add_argument(
        "--simplify",
        type=float,
        default=0.0,
        help="Optional polygon simplification tolerance (pixels). Try 0.5-2.0 for speed.",
    )
    return p.parse_args()


def main() -> None:
    # Configure logging early
    configure_logging()
    logger.info("Starting ROI visualization job")

    args = parse_args()
    generate_roi_visualizations(
        svs_path=args.svs,
        roi_path=args.roi,
        cells_path=args.cells,
        outdir=args.outdir,
        simplify_tolerance=args.simplify,
    )


if __name__ == "__main__":
    main()
