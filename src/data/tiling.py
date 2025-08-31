from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class Tile:
    """
    A single tile specification.

    Attributes
    ----------
    xywh : tuple(int, int, int, int)
        Rectangle in (x, y, width, height) format. Coordinates are in image space.
    index : tuple(int, int)
        Grid index of the tile (row_idx, col_idx).
    pad : tuple(int, int, int, int)
        Padding actually applied when reading this tile, in (top, left, bottom, right).
        Useful to know which pixels are artificial (e.g. reflect or zero).
    """
    xywh: tuple[int, int, int, int]
    index: tuple[int, int]
    pad: tuple[int, int, int, int] = (0, 0, 0, 0)


def _axis_starts(length: int, tile: int, stride: int) -> list[int]:
    """
    Compute tile start positions along one axis (x or y).

    Ensures:
    - Tiles are placed every `stride` pixels.
    - The final tile is 'snapped' to the border if stride does not divide length.
    - All tiles are at least partially inside the image.

    Parameters
    ----------
    length : int
        Image dimension (height or width).
    tile : int
        Desired tile size in pixels.
    stride : int
        Step between consecutive tiles.

    Returns
    -------
    starts : list of int
        Starting positions along the axis.
    """
    # Regular stepping, but ensure we don't overshoot
    starts = list(range(0, max(1, length - tile + 1), stride))

    # Last tile should always reach the border
    last = length - tile
    if last not in starts:
        starts.append(max(0, last))

    return sorted(set(starts))



def generate_tiles(h: int, w: int, tile_size: int = 224, stride: int = 224) -> list[Tile]:
    """
    Generate tiles that cover the entire image.

    By default, stride = tile_size means non-overlapping tiling.
    If stride < tile_size, tiles overlap.
    If stride > tile_size, there will be gaps in coverage.

    Parameters
    ----------
    h, w : int
        Height and width of the image.
    tile_size : int, default=224
        Desired tile size.
    stride : int, default=224
        Step between consecutive tiles.

    Returns
    -------
    tiles : list of Tile
        List of Tile objects covering the image.
    """
    ys = _axis_starts(h, tile_size, stride)
    xs = _axis_starts(w, tile_size, stride)

    tiles = []
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            tiles.append(Tile(xywh=(x, y, tile_size, tile_size),
                              index=(i, j),
                              pad=(0, 0, 0, 0)))
    return tiles


def crop_tile(img: np.ndarray, xywh: tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop a raw tile from an image without padding.

    Parameters
    ----------
    img : np.ndarray, shape (H, W, C)
        Image array (uint8).
    xywh : tuple(int, int, int, int)
        Tile rectangle (x, y, w, h).

    Returns
    -------
    tile_img : np.ndarray
        Cropped region. May be smaller than (w, h) if at border.
    """
    x, y, w, h = xywh
    return img[y:y+h, x:x+w]


def pad_to_size(img: np.ndarray, H: int, W: int, mode: str = "constant") -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Pad an image to reach the desired size (H, W).

    Useful when reading edge tiles that are smaller than the desired
    tile size, ensuring all inputs fed to the model are the same size.

    Parameters
    ----------
    img : np.ndarray
        Input tile (may be smaller than H, W).
    H, W : int
        Target size.
    mode : str, {"reflect", "constant"}
        Padding mode. "reflect" mirrors the border, "constant" fills with zeros.

    Returns
    -------
    padded : np.ndarray
        Padded tile of size (H, W, C).
    pad : tuple(int, int, int, int)
        Amount of padding applied (top, left, bottom, right).
    """
    h, w = img.shape[:2]
    top = left = 0
    bottom = max(0, H - h)
    right = max(0, W - w)

    if bottom == 0 and right == 0:
        return img, (0, 0, 0, 0)

    if mode == "constant":
        padded = cv2.copyMakeBorder(img, top, bottom, left, right,
                                    borderType=cv2.BORDER_CONSTANT,
                                    value=(0, 0, 0))
    elif mode == "reflect":
        padded = cv2.copyMakeBorder(img, top, bottom, left, right,
                                    borderType=cv2.BORDER_REFLECT_101)
    else:
        raise ValueError(f"Unknown pad mode: {mode}")

    return padded, (top, left, bottom, right)