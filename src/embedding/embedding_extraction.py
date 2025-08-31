"""
embedding_extraction.py

Polygon-aware extraction of a single embedding per cell from tile-level token embeddings.

Inputs per tile
--------------
- tok: TokenOutput
    .tokens    -> torch.Tensor, shape (1, T, D), CPU float32
    .grid_hw   -> (Gh, Gw) token grid, e.g., (16,16) for 256/16
    .patch_size-> int, ViT patch size in pixels (e.g., 16)
- cells: dict[int, dict]  (from CellViTHibouWrapper.forward_tile)
    { cell_id: {
        'bbox': (x0,y0,x1,y1),         # in wrapper input_size space
        'centroid': (x,y),
        'contour': np.ndarray(N,2),     # polygon vertices in wrapper input_size space (x,y)
        'type_prob': ...,
        'type': int
      }, ... }
- tile_hw: (H, W)  (the actual tile size as read from the WSI reader, e.g., 224x224)
- input_size: int  (the size used inside wrapper; polygons live in this space, e.g., 256)

Outputs per tile
----------------
- cell_embeds: dict[cell_id -> np.ndarray(D,)]
- cell_meta: dict[cell_id -> dict] (centroid, type, type_prob, bbox, contour scaled to tile space)

Strategy
--------
1) Build token rectangles in tile space from (grid_hw, patch_size).
2) Scale cell contours from wrapper's input_size to tile space.
3) Compute per-token fractional overlap with each cell polygon.
4) Pool token embeddings into one vector per cell.
   - Weighted mean (area fractions) or
   - Ridge unmix (decontamination against neighboring cells/background)

Note: Functions are pure and stateless. No I/O here; keep I/O in scripts.
"""

import numpy as np
import cv2
import torch


# ---------------------- Geometry helpers ----------------------

def token_grid_rects_from_tokenoutput(tile_h, tile_w, grid_hw, patch_size):
    """
    Build (T,4) array of token rectangles (x,y,w,h) in tile coordinates.

    Parameters
    ----------
    tile_h, tile_w : int
        Size of the visible tile in pixels (e.g., 224x224 after WSI read/pad).
    grid_hw : tuple
        (Gh, Gw) token grid reported by the model (e.g., (16,16)).
    patch_size : int
        ViT patch size (commonly 14).

    Returns
    -------
    rects : np.ndarray, shape (T,4), dtype=int32
        Each row is (x, y, w, h).
    """
    Gh, Gw = grid_hw
    rects = []
    for iy in range(Gh):
        for ix in range(Gw):
            x = ix * patch_size
            y = iy * patch_size
            rects.append((x, y, patch_size, patch_size))
    rects = np.asarray(rects, dtype=np.int32)

    # If the tile is not an exact multiple (rare if you resize to SxS), clamp at borders
    rects[:, 0] = np.clip(rects[:, 0], 0, max(0, tile_w - 1))
    rects[:, 1] = np.clip(rects[:, 1], 0, max(0, tile_h - 1))
    rects[:, 2] = np.minimum(rects[:, 2], tile_w - rects[:, 0])
    rects[:, 3] = np.minimum(rects[:, 3], tile_h - rects[:, 1])
    return rects


def scale_contour_to_tile(contour_xy, src_size, dst_h, dst_w):
    """
    Scale a contour from wrapper input_size (src_size x src_size) to tile space (dst_h x dst_w).

    Parameters
    ----------
    contour_xy : np.ndarray, shape (N,2)
        Polygon vertices in (x,y) with 0..src_size-1 coordinates.
    src_size : int
        Size used inside wrapper during inference (e.g., 256).
    dst_h, dst_w : int
        Target tile size.

    Returns
    -------
    cnt : np.ndarray, shape (N,2), float32
        Scaled polygon vertices in tile coordinates.
    """
    if contour_xy is None or len(contour_xy) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    contour_xy = np.asarray(contour_xy, dtype=np.float32)
    sx = float(dst_w) / float(src_size)
    sy = float(dst_h) / float(src_size)
    cnt = contour_xy.copy()
    cnt[:, 0] *= sx
    cnt[:, 1] *= sy
    return cnt.astype(np.float32)


# ---------------------- Overlap computation ----------------------

def overlap_fractions_polygon(tokens_xywh, contour_xy, tile_h, tile_w):
    """
    Compute per-token fractional overlap with a single polygon by rasterization.

    Approach:
    - Rasterize the polygon into a binary mask (tile_h x tile_w).
    - For each token rectangle, compute mean(mask[y:y+h, x:x+w]).
      This equals area_fraction inside that token.

    Parameters
    ----------
    tokens_xywh : np.ndarray (T,4)
        Token rectangles in tile coordinates.
    contour_xy : np.ndarray (N,2), float32
        Cell polygon in tile coordinates.
    tile_h, tile_w : int
        Tile size.

    Returns
    -------
    fractions : np.ndarray (T,), float32
        Fraction of each token covered by the polygon. In [0,1].
    """
    fractions = np.zeros((tokens_xywh.shape[0],), dtype=np.float32)
    if contour_xy is None or len(contour_xy) == 0:
        return fractions

    # Rasterize polygon (fill = 1)
    mask = np.zeros((tile_h, tile_w), dtype=np.uint8)
    cnt = contour_xy.reshape(-1, 1, 2).astype(np.int32)
    cv2.fillPoly(mask, [cnt], color=1)

    # Average over each token rectangle
    for t, (x, y, w, h) in enumerate(tokens_xywh):
        if w <= 0 or h <= 0:
            continue
        patch = mask[y:y+h, x:x+w]
        if patch.size == 0:
            continue
        fractions[t] = float(patch.mean())  # fraction in [0,1]
    return fractions


# ---------------------- Pooling / Decontamination ----------------------

def weighted_pooling(token_embeddings, fractions_by_cell):
    """
    Area-weighted average of tokens for each cell.

    Parameters
    ----------
    token_embeddings : np.ndarray (T, D)
        Token features (float32).
    fractions_by_cell : dict[cell_id -> np.ndarray(T,)]
        Fractional coverages for each cell.

    Returns
    -------
    pooled : dict[cell_id -> np.ndarray(D,)]
    """
    # TODO: Cell id is skipped if no good fractional coverage --> How to deal with that? 
    pooled = {}
    for cid, w in fractions_by_cell.items():
        s = float(w.sum())
        if s <= 1e-8:
            continue
        pooled[cid] = (w[:, None] * token_embeddings).sum(axis=0) / s
    return pooled


def ridge_unmix(token_embeddings, fractions_by_cell, lam=1e-2, include_background=True):
    """
    Ridge regression with optional background column.

    Solves:  E ≈ W Z
      E: (T, D)   token embeddings
      W: (T, C[+1])  fractional coverages (cells [+ background])
      Z: (C[+1], D)  per-cell [+ background] embeddings

    Background column:
      bg = max(0, 1 - sum_c W_cell[:, c])
      This absorbs signal from tissue/padding/empty space so it doesn't
      contaminate nearby nuclei embeddings.

    Parameters
    ----------
    token_embeddings : np.ndarray (T, D)  float32
    fractions_by_cell : dict[cell_id -> np.ndarray(T,)]
    lam : float   L2 regularization strength
    include_background : bool  add bg column as described above

    Returns
    -------
    unmixed : dict[cell_id -> np.ndarray(D,)]
    """
    # Edge case: no cells
    if not fractions_by_cell:
        return {}

    # (1) Stack per-cell fractions into W_cells ∈ R^{T×C}
    cell_ids = sorted(int(k) for k in fractions_by_cell.keys())
    W_cells = np.stack([fractions_by_cell[cid].astype(np.float32) for cid in cell_ids], axis=1)  # (T, C)

    # (2) Optional background column
    if include_background:
        bg = np.maximum(0.0, 1.0 - W_cells.sum(axis=1, keepdims=True))  # (T,1)
        W = np.concatenate([W_cells, bg], axis=1)  # (T, C+1)
    else:
        W = W_cells

    # (3) Solve ridge: Z = (W^T W + λI)^{-1} W^T E
    E = token_embeddings.astype(np.float32)  # (T, D)
    WT = W.T
    A = WT @ W
    # Regularize (tiny floor on lam to avoid exact singularities)
    A += max(lam, 1e-8) * np.eye(A.shape[0], dtype=np.float32)
    B = WT @ E
    Z = np.linalg.solve(A, B)  # (C[+1], D)

    # (4) Return only the per-cell rows of Z (discard background row if present)
    C = W_cells.shape[1]
    Z_cells = Z[:C, :].astype(np.float32)
    return {cid: Z_cells[i] for i, cid in enumerate(cell_ids)}


# ---------------------- Main per-tile routine ----------------------

def per_tile_cell_embeddings(tok, cells, tile_hw, input_size,
                             method="ridge_unmix", ridge_lambda=1e-2, include_background=True):
    """
    Compute one embedding per cell for a single tile.

    Parameters
    ----------
    tok : TokenOutput
        tokens: (1, T, D) CPU float32
        grid_hw: (Gh, Gw)
        patch_size: int
    cells : dict[int, dict]
        Hibou instance dict (contour in input_size space).
    tile_hw : tuple (H, W)
        Actual tile size in pixels (after WSI read + padding if any).
    input_size : int
        Size used inside wrapper during inference (e.g., 256).
    method : {"weighted_pool", "ridge_unmix"}
        Pooling / decontamination method.
    ridge_lambda : float
        L2 reg for ridge unmix.
    include_background : bool
        Add background column in ridge.

    Returns
    -------
    cell_embeds : dict[cell_id -> np.ndarray(D,)]
    cell_meta   : dict[cell_id -> dict]   # with scaled contour, centroid, type info, etc.
    """
    H, W = int(tile_hw[0]), int(tile_hw[1])
    T, D = tok.tokens.shape[1], tok.tokens.shape[2]

    # 1) Token rectangles in tile space
    rects = token_grid_rects_from_tokenoutput(H, W, tok.grid_hw, tok.patch_size)  # (T,4)

    # 2) Convert tokens to np for math
    token_np = tok.tokens.squeeze(0).numpy().astype(np.float32)  # (T, D)

    # 3) Build per-cell overlap fractions
    fractions_by_cell = {}
    cell_meta = {}

    # Rescale factors
    sx = W / float(input_size)
    sy = H / float(input_size)

    for cid, cdict in (cells or {}).items():
        # Scale contour to tile space
        cnt_src = cdict.get("contour", None)
        cnt_tile = scale_contour_to_tile(cnt_src, src_size=input_size, dst_h=H, dst_w=W)

        # Compute fractions for this polygon
        frac = overlap_fractions_polygon(rects, cnt_tile, H, W)  # (T,)
        if frac.sum() <= 1e-8:
            continue  # no overlap with any token

        fractions_by_cell[int(cid)] = frac

        # Prepare metadata (scaled)
        centroid_tile = _parse_centroid_2(cdict.get("centroid", None), sx, sy)
        bbox_tile = _parse_bbox_2x2(cdict.get("bbox", None), sx, sy)
        ctype = cdict.get("type", None)
        cprob = _parse_type_prob_scalar(cdict.get("type_prob", None))

        cell_meta[int(cid)] = {
            "contour": cnt_tile,                  # np.ndarray(N,2) in tile coords
            "centroid": centroid_tile,            # (x,y) in tile coords
            "bbox": bbox_tile,                    # (x0,y0,x1,y1) in tile coords
            "type": int(ctype) if ctype is not None else None,
            "type_prob": cprob,
        }

    # 4) Pool embeddings
    if method == "weighted_pool":
        cell_embeds = weighted_pooling(token_np, fractions_by_cell)
    elif method == "ridge_unmix":
        cell_embeds = ridge_unmix(token_np, fractions_by_cell, lam=ridge_lambda, include_background=include_background)
    else:
        raise ValueError(f"Unknown method: {method}")

    return cell_embeds, cell_meta


# ---------------------- Optional: tile row builder for saving ----------------------

def build_rows_for_saving(cell_embeds, cell_meta, tile_xywh, tile_index):
    """
    Build flat dict rows (one per cell) ready for DataFrame/Parquet saving.

    Parameters
    ----------
    cell_embeds : dict[cell_id -> np.ndarray(D,)]
    cell_meta : dict[cell_id -> dict]
    tile_xywh : tuple (x, y, w, h)
        Tile top-left and size in slide coordinates (level you tiled at).
    tile_index : tuple (i, j)
        Grid index for the tile.

    Returns
    -------
    rows : list[dict]
        Each row contains tile metadata, cell_id, and embedding vector (e_0..e_{D-1}).
        You can later join with serialized contour/centroid if desired.
    """
    rows = []
    for cid, emb in cell_embeds.items():
        row = {
            "tile_i": tile_index[0],
            "tile_j": tile_index[1],
            "x": tile_xywh[0], "y": tile_xywh[1], "w": tile_xywh[2], "h": tile_xywh[3],
            "cell_id": int(cid),
        }
        emb = np.asarray(emb, dtype=np.float32).ravel()
        for k, v in enumerate(emb.tolist()):
            row[f"e_{k}"] = float(v)
        # (Optional) include type/centroid/contour summary:
        meta = cell_meta.get(cid, {})
        if meta.get("type") is not None:
            row["type"] = int(meta["type"])
        if meta.get("type_prob") is not None:
            row["type_prob"] = float(meta["type_prob"])
        if meta.get("centroid") is not None:
            cx, cy = meta["centroid"]
            row["centroid_x"] = float(cx); row["centroid_y"] = float(cy)
        if meta.get("contour") is not None:
            # Serialize contour as a list of [x, y] pairs
            contour_arr = np.asarray(meta["contour"], dtype=np.float32)
            row["contour"] = contour_arr.tolist()
        rows.append(row)
    return rows


# ---------------------- Helpers ----------------------


def _parse_bbox_2x2(bbox, sx, sy):
    """
    Accept bbox as np.ndarray shape (2,2) -> [[x0,y0],[x1,y1]] in source space.
    Scale to tile space using (sx, sy). Return (x0, y0, x1, y1) as floats.
    """
    if bbox is None:
        return None
    bbox = np.asarray(bbox, dtype=np.float32)
    if bbox.shape != (2, 2):
        # Fallback: try to interpret flat or tuple-like
        bbox = bbox.reshape(-1)
        if bbox.size >= 4:
            x0, y0, x1, y1 = map(float, bbox[:4])
            return (x0 * sx, y0 * sy, x1 * sx, y1 * sy)
        return None
    x0, y0 = float(bbox[0, 0]), float(bbox[0, 1])
    x1, y1 = float(bbox[1, 0]), float(bbox[1, 1])
    return (x0 * sx, y0 * sy, x1 * sx, y1 * sy)

def _parse_centroid_2(centroid, sx, sy):
    """
    Accept centroid as np.ndarray shape (2,) -> [x, y] in source space.
    Scale to tile space using (sx, sy). Return (cx, cy) as floats.
    """
    if centroid is None:
        return None
    c = np.asarray(centroid, dtype=np.float32).reshape(-1)
    if c.size < 2:
        return None
    cx, cy = float(c[0]), float(c[1])
    return (cx * sx, cy * sy)

def _parse_type_prob_scalar(tp):
    """
    Ensure type_prob is a scalar float (confidence). If an array sneaks in, take max or first.
    """
    if tp is None:
        return None
    if np.isscalar(tp):
        return float(tp)
    arr = np.asarray(tp, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    # choose a sensible scalar: max confidence if vector provided
    return float(arr.max())