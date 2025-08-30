from pathlib import Path
import numpy as np
import openslide 



class WSIReader:
    """
    Wrapper around OpenSlide for reading WSIs.

    Parameters
    ----------
    path : str
        Path to a whole slide image (WSI).
    level : int, default=0
        Pyramid level to operate on (0 = highest resolution).

    Attributes
    ----------
    slide : openslide.OpenSlide
        Underlying OpenSlide object.
    level : int
        Current pyramid level.
    path : str
        Original path to the WSI file.
    """

    def __init__(self, path: str | Path, level: int = 0):
        self.path = Path(path)
        self.level = level
        try:
            self.slide = openslide.OpenSlide(str(self.path))
        except Exception as e:
            raise RuntimeError(f"Failed to open slide with OpenSlide: {e}")
    

    # ---------------- Public API ----------------


    def size(self) -> tuple[int, int]: # (H, W)
        """
        Returns
        -------
        (H, W) : tuple of int
            Dimensions at the current pyramid level.
            Returned as (height, width) to match numpy conventions.
        """
        w, h = self.slide.level_dimensions[self.level]
        return h, w


    def levels(self):
        """
        Returns
        -------
        list of (W, H) tuples
            Dimensions of all pyramid levels, from level 0 .. N.
            Note: OpenSlide stores these as (width, height).
        """
        return list(self.slide.level_dimensions)
    

    def read_region(self, x, y, w, h):
        """
        Read an RGB region at the current level.

        Parameters
        ----------
        x, y : int
            Top-left coordinates in the *current level* coordinate system.
        w, h : int
            Width and height in pixels (at the current level).

        Returns
        -------
        region : np.ndarray of shape (h, w, 3), dtype=uint8
            RGB image region.

        Notes
        -----
        - OpenSlide requires coordinates in level-0 reference frame.
        - We convert (x, y) from current level → level-0.
        - If the requested region extends outside the slide, OpenSlide pads with black.
        """
        # Convert to level-0 coordinates
        downsample = self.slide.level_downsamples[self.level]
        x0 = int(round(x * downsample))
        y0 = int(round(y * downsample))

        pil_img = self.slide.read_region(
            (x0, y0), self.level, (int(w), int(h))
        ).convert("RGB")

        return np.asarray(pil_img)

    def mpp(self):
        """
        Microns-per-pixel (MPP) metadata from slide properties.

        Returns
        -------
        dict with keys {"mpp_x", "mpp_y"}
            Values are floats or None if missing.
        """
        props = self.slide.properties

        def _get_float(key):
            v = props.get(key, None)
            try:
                return float(v) if v is not None else None
            except Exception:
                return None

        return {
            "mpp_x": _get_float("openslide.mpp-x"),
            "mpp_y": _get_float("openslide.mpp-y"),
        }

    def close(self):
        """
        Close the OpenSlide file handle.
        """
        self.slide.close()

    # -------------- Context Manager --------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
    