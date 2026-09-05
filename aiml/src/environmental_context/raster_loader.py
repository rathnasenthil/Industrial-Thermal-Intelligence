"""
Local raster loaders for Stage I.6.

Never downloads data. Returns None when the configured path is missing or
unreadable, so callers can emit unavailable evidence instead of fabricating
values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RasterDataset:
    """In-memory handle for a local single-band (or first-band) raster."""

    path: Path
    dataset: Any  # rasterio dataset object (kept open by caller/context)
    transform: Any
    crs: Any
    nodata: float | None
    width: int
    height: int
    source_name: str

    def sample_values(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Sample band 1 at lon/lat (EPSG:4326) or projected coords matching raster CRS.

        ``xs``/``ys`` must already be in the raster's CRS.
        Returns float array with NaN for nodata / out-of-bounds.
        """
        import rasterio
        from rasterio.windows import Window

        values = np.full(len(xs), np.nan, dtype=float)
        rows, cols = rasterio.transform.rowcol(self.transform, xs, ys)
        for i, (r, c) in enumerate(zip(rows, cols)):
            if r < 0 or c < 0 or r >= self.height or c >= self.width:
                continue
            # Read 1x1 window
            window = Window(c, r, 1, 1)
            data = self.dataset.read(1, window=window)
            val = float(data[0, 0])
            if self.nodata is not None and (val == self.nodata or np.isnan(val)):
                continue
            values[i] = val
        return values

    def close(self) -> None:
        try:
            self.dataset.close()
        except Exception:
            pass


def resolve_existing_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.exists() and p.is_file() else None


def open_raster(path: Path | None, *, source_name: str = "local_raster") -> RasterDataset | None:
    """Open a local GeoTIFF/COG if present; otherwise return None."""
    existing = resolve_existing_path(path)
    if existing is None:
        return None
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError("rasterio is required to load land-cover/satellite rasters.") from exc

    try:
        ds = rasterio.open(existing)
    except Exception:
        return None
    return RasterDataset(
        path=existing,
        dataset=ds,
        transform=ds.transform,
        crs=ds.crs,
        nodata=ds.nodata,
        width=ds.width,
        height=ds.height,
        source_name=source_name,
    )
