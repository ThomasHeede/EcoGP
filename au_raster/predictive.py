import rasterio
from rasterio.warp import transform
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pyro
import os
import argparse
import tqdm


class RasterDataset(Dataset):
    def __init__(self, df, traits=None, device="cpu"):
        self.X = torch.tensor(df.values, dtype=torch.float32)
        self.Y = None
        self.traits = traits
        self.coords = torch.tensor(df.index.to_frame(index=False).iloc[:, -2:].values, dtype=torch.float32)

        self.raster_idx = torch.tensor(df.index.to_frame(index=False).iloc[:, :2].values)

        self.env_names = df.columns.tolist()

        self.device = device

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return idx

    def get_batch_data(self, idx):
        """
        Returning data for batch indices
        :param idx: list of integers for indices
        :return: Target data if available
        """
        X = self.X[idx].to(self.device) if self.X is not None else None
        Y = self.Y[idx].to(self.device) if self.Y is not None else None
        coords = self.coords[idx].to(self.device) if self.coords is not None else None
        traits = self.traits.to(self.device) if self.traits is not None else None
        raster_idx = self.raster_idx[idx] if self.raster_idx is not None else None

        return X, Y, coords, traits, raster_idx


if __name__ == "__main__":
    # Specify paths to input TIFF files and output folder
    parser = argparse.ArgumentParser()

    parser.add_argument("--OUT_FOLDER", type=str, default="predicted_rasters")
    parser.add_argument("--IN_FOLDER", type=str, default="environmental_rasters")
    parser.add_argument("--N_SAMPLES", type=int, default=100)
    parser.add_argument("--BATCH_SIZE", type=int, default=512)

    args = parser.parse_args()

    OUT_FOLDER = args.OUT_FOLDER
    IN_FOLDER = args.IN_FOLDER
    N_SAMPLES = args.N_SAMPLES
    BATCH_SIZE = args.BATCH_SIZE

    TIFF_PATHS = [
        os.path.join(IN_FOLDER, f)
        for f in os.listdir(IN_FOLDER)
        if os.path.isfile(os.path.join(IN_FOLDER, f))
    ]
    print(f"Found {len(TIFF_PATHS)} TIFF files\n"
          f"Loading:")

    ##### READ TIFF FILES #####
    dfs = []
    for tiff_path in tqdm.tqdm(TIFF_PATHS):
        with rasterio.open(tiff_path) as src:
            data = src.read(1, masked=True)

            rows, cols = np.where(~data.mask)

            xs, ys = rasterio.transform.xy(src.transform, rows, cols)

            # Convert to lat/lon
            lon, lat = transform(src.crs, "EPSG:4326", xs, ys)

        dfs.append(
            pd.DataFrame({
                "row": rows,
                "col": cols,
                "longitude": lon,
                "latitude": lat,
                tiff_path.split("/")[-1].strip(".tif"): data.data[rows, cols]
            }).set_index(["row", "col", "longitude", "latitude"])
        )

    df = pd.concat(dfs, axis=1, join="inner")
    
    ##### PREPROCESSING FOR MODEL PREDICTION #####
    dataset = torch.load("learned_model/dataset.pt", weights_only=False)

    # Verify that all env_names correspond to the columns in the dataframe
    environments_match = set(dataset.env_names) == set(df.columns)
    assert environments_match, (
        f"Expected raster files {df.columns.tolist()} to be contained in training environmental features {dataset.env_names}\n"
        f"Missing raster for: {set(df.columns) - set(dataset.env_names)}\n"
        f"Raster with no match: {set(dataset.env_names) - set(df.columns)}")

    df = df[dataset.env_names]

    if dataset.normalize_X:
        print("Normalizing environmental features")
        mask = dataset.X_continuous
        cols = df.columns[mask]

        df[cols] = df[cols].astype(float).sub(dataset.X_continuous_mean).div(dataset.X_continuous_std)

    # normalize coordinates
    normalize_coords = True
    if normalize_coords:
        print("Normalizing coordinates (numerical stability)\n")
        index = df.index.names
        df = df.reset_index()  # [["longitude", "latitude"]]
        df[["longitude", "latitude"]] = (df[["longitude", "latitude"]] - dataset.coords.mean(
            dim=0)) / dataset.coords.std(dim=0)
        df = df.set_index(["row", "col", "longitude", "latitude"])

    raster_dataset = RasterDataset(df)
    del df

    raster_dataset.Y = torch.zeros((len(raster_dataset), len(dataset.taxon_names)), dtype=torch.float32).to(dataset.device)  # Dummy Y for prediction
    
    ##### LOAD MODEL AND PREDICT #####
    model = torch.load("learned_model/model.pt", weights_only=False)

    predictive = pyro.infer.Predictive(model.model, guide=model.guide, num_samples=N_SAMPLES, parallel=True)

    predict_dataloader = DataLoader(dataset=raster_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False)

    raster_idxs = []
    y_prob_means = []
    y_prob_uncertainties = []

    print("Making Predictions:")
    for idx in tqdm.tqdm(predict_dataloader):
        X, Y, coords, traits, raster_idx = raster_dataset.get_batch_data(idx)
        raster_idxs.append(raster_idx)

        predictive_samples = predictive(X, Y, coords, traits, training=False)

        y_prob_means.append(predictive_samples["y"].squeeze().mean(dim=0))
        y_prob_uncertainties.append(predictive_samples["y"].squeeze().var(dim=0))

    raster_idxs = torch.cat(raster_idxs, dim=0)
    y_prob_means = torch.cat(y_prob_means, dim=0)
    y_prob_uncertainties = torch.cat(y_prob_uncertainties, dim=0)
    print("Saving predictions to TIFF files")
    
    ##### SAVE PREDICTIONS #####
    # Use first raster as template for metadata
    with rasterio.open(TIFF_PATHS[0]) as src:
        meta = src.meta.copy()
        meta["dtype"] = "float32"
        meta["nodata"] = float("nan")
        meta["count"] = 2  # 3 channels for mean, uncertainty
        
    # Writing predictions to new TIFF files
    os.makedirs(OUT_FOLDER, exist_ok=True)

    for i, taxon in enumerate(dataset.taxon_names):
        grid = torch.full((meta["count"], meta["height"], meta["width"]), meta["nodata"], dtype=torch.float32)
        grid[0, raster_idxs[:, 0], raster_idxs[:, 1]] = y_prob_means[:, i]
        grid[1, raster_idxs[:, 0], raster_idxs[:, 1]] = y_prob_uncertainties[:, i]

        with rasterio.open(os.path.join(OUT_FOLDER, f"{taxon}.tif"), "w", **meta) as dst:
            dst.write(grid)

            # Set band descriptions (names)
            for i, name in enumerate(["mean", "uncertainty"]):
                dst.set_band_description(i + 1, name)

    print("Done! Predictions saved to TIFF files in folder:", OUT_FOLDER)
    