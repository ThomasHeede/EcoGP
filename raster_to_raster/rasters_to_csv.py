"""
Convert a folder of GeoTIFF rasters to X.csv (features) and XY.csv (coordinates).

Each TIFF becomes one feature column (multi-band TIFFs produce one column per band).
Rows with any NaN/nodata value are dropped.
Coordinates are derived once at the end from the first raster's geotransform.

Usage:
    python rasters_to_csv.py <folder> [--output-dir <dir>]

Requirements:
    pip install rasterio numpy pandas pyproj
"""

# ── Fix PROJ db version mismatch ──────────────────────────────────────────────
# Must happen before ANY geo-library import. pyproj ≥ 3.0 ships its own
# proj.db inside the package; we redirect PROJ_DATA there so it takes
# precedence over a stale conda-env share/proj/proj.db.
import importlib.util
import os

def _fix_proj_data() -> str | None:
    spec = importlib.util.find_spec("pyproj")
    if spec is None or spec.origin is None:
        return None
    pkg_dir = os.path.dirname(spec.origin)
    for candidate in (
        os.path.join(pkg_dir, "proj_data"),   # pyproj >= 3.0
        os.path.join(pkg_dir, "_data"),        # some wheel builds
        os.path.join(pkg_dir, "data"),
    ):
        if os.path.isfile(os.path.join(candidate, "proj.db")):
            os.environ["PROJ_DATA"] = candidate
            os.environ["PROJ_LIB"] = candidate  # legacy name
            return candidate
    return None

_proj_data_dir = _fix_proj_data()
# ──────────────────────────────────────────────────────────────────────────────

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS as ProjCRS
from pyproj import Transformer


def rasters_to_csv(folder: str | Path, output_dir: str | Path = ".") -> None:
    folder = Path(folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if _proj_data_dir:
        print(f"[proj] Using bundled PROJ data: {_proj_data_dir}")

    tiff_files = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
    if not tiff_files:
        raise ValueError(f"No .tif/.tiff files found in {folder}")

    print(f"Found {len(tiff_files)} TIFF file(s). Reading...")

    arrays: list[np.ndarray] = []
    columns: list[str] = []
    ref_transform = None
    ref_crs = None
    ref_shape: tuple[int, int] | None = None

    for tiff_path in tiff_files:
        stem = tiff_path.stem
        with rasterio.open(tiff_path) as src:
            if ref_transform is None:
                ref_transform = src.transform
                ref_crs = src.crs
                ref_shape = (src.height, src.width)
            elif (src.height, src.width) != ref_shape:
                raise ValueError(
                    f"{tiff_path.name} has shape {(src.height, src.width)} "
                    f"but expected {ref_shape}. All rasters must be co-registered."
                )

            data = src.read().astype(np.float64)  # (bands, height, width)

            for band_idx in range(data.shape[0]):
                band = data[band_idx]
                if src.nodata is not None:
                    band[band == src.nodata] = np.nan
                arrays.append(band.ravel())
                col_name = stem if data.shape[0] == 1 else f"{stem}_band{band_idx + 1}"
                columns.append(col_name)

        print(f"  {tiff_path.name}: {data.shape[0]} band(s)")

    n_pixels = ref_shape[0] * ref_shape[1]
    print(f"\nStacking {len(columns)} feature(s) over {n_pixels:,} pixels...")

    # Sort features alphabetically before stacking
    order = np.argsort(columns)
    columns = [columns[i] for i in order]
    arrays  = [arrays[i]  for i in order]

    # Shape: (n_pixels, n_features)
    X = np.column_stack(arrays)
    del arrays

    valid_mask = ~np.any(np.isnan(X), axis=1)
    n_valid = int(valid_mask.sum())
    print(f"Valid pixels after dropping NaNs: {n_valid:,} / {n_pixels:,} ({n_valid / n_pixels:.1%})")

    # Flat pixel indices of valid rows — used as the CSV index so that
    # csv_to_rasters.py can map predictions back to the correct pixels.
    valid_indices = np.where(valid_mask)[0]

    X_valid = X[valid_mask]
    del X

    # --- Derive coordinates from reference raster ---
    print("Computing coordinates...")
    height, width = ref_shape
    row_idx, col_idx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    # pixel-centre coordinates in the raster's native CRS
    xs, ys = rasterio.transform.xy(ref_transform, row_idx.ravel(), col_idx.ravel())
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    if ref_crs is not None and not ref_crs.is_geographic:
        # Reproject to WGS-84 using pyproj.
        # Build both CRS from WKT / PROJ4 to avoid any EPSG DB lookup.
        src_proj = ProjCRS.from_wkt(ref_crs.to_wkt())
        wgs84    = ProjCRS.from_proj4("+proj=longlat +datum=WGS84 +no_defs")
        transformer = Transformer.from_crs(src_proj, wgs84, always_xy=True)
        lons, lats = transformer.transform(xs, ys)
        lons = np.asarray(lons, dtype=np.float64)
        lats = np.asarray(lats, dtype=np.float64)
    else:
        # Already geographic: x = longitude, y = latitude
        lons, lats = xs, ys

    lons_valid = lons[valid_mask]
    lats_valid = lats[valid_mask]
    del lons, lats, valid_mask

    # --- Write output ---
    x_path  = output_dir / "X.csv"
    xy_path = output_dir / "XY.csv"

    print(f"Writing {x_path} ...")
    pd.DataFrame(X_valid, columns=columns, index=valid_indices).to_csv(
        x_path, index=True, index_label="pixel_index"
    )

    print(f"Writing {xy_path} ...")
    pd.DataFrame(
        {"longitude": lons_valid, "latitude": lats_valid},
        index=valid_indices,
    ).to_csv(xy_path, index=True, index_label="pixel_index")

    print(f"\nDone. {n_valid:,} rows written to:\n  {x_path}\n  {xy_path}")
    print(f"  Index column 'pixel_index' encodes flat pixel position in a "
          f"{ref_shape[0]} × {ref_shape[1]} grid — used by csv_to_rasters.py.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a folder of co-registered GeoTIFFs to X.csv and XY.csv."
    )
    parser.add_argument("--input-dir", help="Folder containing .tif / .tiff files")
    parser.add_argument(
        "--output-dir",
        default=".",
        metavar="DIR",
        help="Directory for output CSVs (default: current directory)",
    )
    args = parser.parse_args()
    rasters_to_csv(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
    # python au_raster/rasters_to_csv.py au_raster/dummy_data/input_raster --output-dir au_raster/dummy_data/model_data
