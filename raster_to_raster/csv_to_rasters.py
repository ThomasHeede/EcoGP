"""
Convert a predictions CSV back to GeoTIFF rasters, one file per column.

The input CSV must have a 'pixel_index' index column produced by rasters_to_csv.py
(i.e. the flat pixel position in the original raster grid). A reference raster
from the original folder is required to recover the spatial metadata
(CRS, affine transform, grid dimensions).

Usage:
    python csv_to_rasters.py <predictions.csv> <reference_tiff> [--output-dir <dir>]

Example:
    python csv_to_rasters.py Y_pred.csv input_rasters/bio01.tif --output-dir output_rasters/

Requirements:
    pip install rasterio numpy pandas
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


def _first_tif(folder: Path) -> Path:
    for pattern in ("*.tif", "*.tiff"):
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No .tif/.tiff files found in {folder}")

NODATA = np.nan  # written as float32 NaN for unsampled pixels


def csv_to_rasters(
    csv_path: str | Path,
    reference_tif: str | Path,
    output_dir: str | Path = ".",
) -> None:
    csv_path     = Path(os.path.join(csv_path, "Y_pred.csv"))
    output_dir   = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_tif = _first_tif(Path(reference_tif))
    print(f"Reading spatial metadata from {reference_tif} ...")

    # --- Load predictions ---
    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path, index_col=0)
    n_rows, n_cols = df.shape
    print(f"  {n_rows:,} rows × {n_cols} output variable(s): {list(df.columns)}")

    # --- Load spatial metadata from reference raster ---
    print(f"Reading spatial metadata from {reference_tif} ...")
    with rasterio.open(reference_tif) as src:
        height    = src.height
        width     = src.width
        transform = src.transform
        crs       = src.crs
        dtype     = "float32"

    n_pixels = height * width
    print(f"  Grid: {height} rows × {width} cols = {n_pixels:,} pixels")

    # Validate indices
    bad = df.index[(df.index < 0) | (df.index >= n_pixels)]
    if len(bad):
        raise ValueError(
            f"{len(bad)} pixel_index values are out of range [0, {n_pixels - 1}]: "
            f"{bad[:5].tolist()}{'…' if len(bad) > 5 else ''}"
        )

    # --- Write one GeoTIFF per output column ---
    for col in df.columns:
        out_path = output_dir / f"{col}.tif"
        print(f"Writing {out_path} ...")

        # Full flat array filled with NaN, then scatter predictions into it
        flat = np.full(n_pixels, np.nan, dtype=np.float32)
        flat[df.index.to_numpy()] = df[col].to_numpy(dtype=np.float32)

        raster = flat.reshape(height, width)

        with rasterio.open(
            out_path,
            mode="w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=dtype,
            crs=crs,
            transform=transform,
            nodata=np.nan,
            compress="deflate",   # lossless compression — keeps file sizes small
            predictor=2,          # horizontal differencing (good for float data)
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst:
            dst.write(raster, 1)

    print(f"\nDone. {n_cols} raster(s) written to {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a predictions CSV (with pixel_index from rasters_to_csv.py) "
            "back into GeoTIFF rasters, one file per output column."
        )
    )
    parser.add_argument(
        "--prediction-dir",
        help="Direction to directory holding predictions CSV with a 'pixel_index' index column",
    )
    parser.add_argument(
        "--reference-tif",
        help="Any input .tif from the original raster folder (supplies CRS / grid)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        metavar="DIR",
        help="Directory for output GeoTIFFs (default: current directory)",
    )
    args = parser.parse_args()
    csv_to_rasters(args.prediction_dir, args.reference_tif, args.output_dir)


if __name__ == "__main__":
    main()
