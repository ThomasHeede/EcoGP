import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import pyro
import torch
import rioxarray
import argparse
import numpy as np
import xarray as xr

# python raster_to_raster/direct_pred.py --predictor_dir raster_to_raster/test/data/input_raster --pretrained_dir raster_to_raster/test/pretrained_model/ --output raster_to_raster/test/results/prediction_mean_stack.tif --chunk_x 512 --chunk_y 512 --pixel_batch_size 20000 --threads 6

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict EcoGP over raster predictors."
    )

    parser.add_argument(
        "--predictor_dir",
        required=True,
        help="Directory containing predictor rasters."
    )

    parser.add_argument(
        "--pretrained_dir",
        required=True,
        help="Directory containing the trained model."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output GeoTIFF path for prediction mean stack."
    )

    parser.add_argument(
        "--output_sd",
        required=True,
        help="Output GeoTIFF path for prediction sd stack."
    )

    parser.add_argument(
        "--chunk_x",
        type=int,
        default=512,
        help="Dask chunk size in x direction."
    )

    parser.add_argument(
        "--chunk_y",
        type=int,
        default=512,
        help="Dask chunk size in y direction."
    )

    parser.add_argument(
        "--pixel_batch_size",
        type=int,
        default=20000,
        help="Number of valid pixels passed to EcoGP at once."
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Number of Dask threads."
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of posterior predictive samples."
    )
    return parser.parse_args()

args = parse_args()

RASTER_DIR = Path(args.predictor_dir)
PRETRAINED_DIR = Path(args.pretrained_dir)
OUTPUT_TIF = Path(args.output)
OUTPUT_SD_TIF = Path(args.output_sd)
CHUNK_X = args.chunk_x
CHUNK_Y = args.chunk_y
PIXEL_BATCH_SIZE = args.pixel_batch_size
THREADS = args.threads
N_SAMPLES = args.n_samples

PROJECT_DIR = "/home/fdelogu/EcoGP_pred_test/EcoGP"
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, MODELS_DIR)

model = torch.load(
    os.path.join(PRETRAINED_DIR, "model.pt"),
    map_location="cpu",
    weights_only=False,
)

def find_raster_by_name(directory, variable_name, extensions=(".tif", ".tiff", ".vrt")):
    """
    Find a raster whose file stem exactly matches the variable name.

    Example:
        variable_name = "elevation"
        matches elevation.tif, elevation.tiff, or elevation.vrt
    """

    directory = Path(directory)

    matches = [
        path
        for ext in extensions
        for path in directory.glob(f"{variable_name}{ext}")
    ]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No raster found for variable '{variable_name}' in {directory}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple rasters found for variable '{variable_name}': {matches}"
        )

    return matches[0]


def load_predictor_stack_from_directory(
    predictor_dir,
    feature_names,
    chunks=None,
    masked=True,
):
    """
    Load predictor rasters lazily from a directory.

    The raster filenames must match the feature names used by the model.

    Parameters
    ----------
    predictor_dir : str or Path
        Directory containing predictor rasters.
    feature_names : list[str]
        Feature names expected by the fitted model.
    chunks : dict or None
        Dask chunking, for example {"x": 1024, "y": 1024}.
    masked : bool
        Whether to open rasters as masked arrays.

    Returns
    -------
    xarray.DataArray
        Lazy raster stack with dimensions:
        feature, y, x
    """

    predictor_dir = Path(predictor_dir)

    layers = []

    for name in feature_names:
        raster_path = find_raster_by_name(predictor_dir, name)

        da = (
            rioxarray.open_rasterio(
                raster_path,
                chunks=chunks,
                masked=masked,
            )
            .squeeze("band", drop=True)
            .astype("float32")
        )

        da.name = name
        layers.append(da)

    from rasterio.enums import Resampling

    reference = layers[0]
    aligned_layers = [reference]

    for name, da in zip(feature_names[1:], layers[1:]):

        same_grid = (
                da.rio.crs == reference.rio.crs
                and da.rio.shape == reference.rio.shape
                and da.rio.transform() == reference.rio.transform()
        )

        if not same_grid:
            print(f"Aligning {name} to reference raster grid")

            da = da.rio.reproject_match(
                reference,
                resampling=Resampling.bilinear,
            )

        aligned_layers.append(da)

    layers = aligned_layers

    predictors = xr.concat(
        layers,
        dim=xr.IndexVariable("feature", feature_names),
    ).transpose("feature", "y", "x")

    if chunks is not None:
        predictors = predictors.chunk(
            {
                "feature": len(feature_names),
                **chunks,
            }
        )

    return predictors

#PRETRAINED_DIR = "raster_to_raster/test/pretrained_model/"
dataset = torch.load(os.path.join(PRETRAINED_DIR, "dataset.pt"), weights_only=False)
#print(dataset.env_names)

feature_names = dataset.env_names
# or: feature_names = ["elevation", "slope", "temperature", "precipitation"]

#RASTER_DIR = "raster_to_raster/test/data/input_raster/"
predictors = load_predictor_stack_from_directory(
    predictor_dir=RASTER_DIR,
    feature_names=feature_names,
    chunks={"x": CHUNK_X, "y": CHUNK_Y},
)

#print(predictors.feature.values)


def get_model_dtype_device(model):
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype, p.device

    for b in model.buffers():
        if b.is_floating_point():
            return b.dtype, b.device

    return torch.float32, torch.device("cpu")


def as_batch_by_species(array, batch_size, n_species, name):
    """
    Convert model output to shape:
        n_pixels x n_species
    """

    array = array.detach().cpu().numpy()
    array = np.squeeze(array)

    if array.ndim == 1:
        if n_species == 1 and array.shape[0] == batch_size:
            return array[:, None]

        if batch_size == 1 and array.shape[0] == n_species:
            return array[None, :]

    if array.ndim == 2:
        if array.shape == (batch_size, n_species):
            return array

        if array.shape == (n_species, batch_size):
            return array.T

    raise ValueError(
        f"Could not interpret {name} shape {array.shape}. "
        f"Expected ({batch_size}, {n_species}) or ({n_species}, {batch_size})."
    )


def predict_gp_chunk(
    block,
    model,
    feature_names,
    species_names,
    training_mean=None,
    training_std=None,
    coords_mean=None,
    coords_std=None,
    pixel_batch_size=20000,
    n_samples=100,
    param_store_state=None,
):
    import numpy as np
    import torch
    import xarray as xr
    import pyro

    def get_model_dtype_device(model):
        for p in model.parameters():
            if p.is_floating_point():
                return p.dtype, p.device

        for b in model.buffers():
            if b.is_floating_point():
                return b.dtype, b.device

        return torch.float32, torch.device("cpu")

    def y_samples_to_pixel_species(y_samples, batch_size, n_species):
        """
        Convert predictive_samples["y"] to shape:

            n_samples x batch_size x n_species
        """

        y = y_samples.detach().cpu()

        # Keep sample dimension, remove singleton dimensions after it
        for dim in reversed(range(1, y.ndim)):
            if y.shape[dim] == 1:
                y = y.squeeze(dim)

        if y.ndim == 2:
            # n_samples x batch_size, only one species
            if n_species == 1 and y.shape[1] == batch_size:
                y = y[:, :, None]
            else:
                raise ValueError(
                    f"Could not interpret y shape {tuple(y.shape)} "
                    f"for batch_size={batch_size}, n_species={n_species}"
                )

        elif y.ndim == 3:
            if y.shape[1:] == (batch_size, n_species):
                pass

            elif y.shape[1:] == (n_species, batch_size):
                y = y.transpose(1, 2)

            else:
                raise ValueError(
                    f"Could not interpret y shape {tuple(y.shape)} "
                    f"for batch_size={batch_size}, n_species={n_species}"
                )

        else:
            raise ValueError(
                f"Could not interpret y shape {tuple(y.shape)}"
            )

        return y

    if param_store_state is not None:
        pyro.clear_param_store()
        pyro.get_param_store().set_state(param_store_state)

    block = block.sel(feature=feature_names)

    values = np.asarray(block.data, dtype=np.float32)

    # values shape: feature, y, x
    n_features, n_rows, n_cols = values.shape
    n_species = len(species_names)

    valid = np.isfinite(values).all(axis=0)

    prediction = np.full(
        (n_species, n_rows, n_cols),
        np.nan,
        dtype=np.float32,
    )

    prediction_sd = np.full(
        (n_species, n_rows, n_cols),
        np.nan,
        dtype=np.float32,
    )

    if not valid.any():
        return xr.Dataset(
            {
                "prediction": (("species", "y", "x"), prediction),
                "prediction_sd": (("species", "y", "x"), prediction_sd),
            },
            coords={
                "species": species_names,
                "y": block.y,
                "x": block.x,
            },
        )

    # Shape: n_valid_pixels × n_features
    X = values[:, valid].T.astype(np.float32)

    if training_mean is not None and training_std is not None:
        training_mean = np.asarray(training_mean, dtype=np.float32)
        training_std = np.asarray(training_std, dtype=np.float32)

        X = (X - training_mean) / training_std

    # Coordinates: n_valid_pixels x 2
    xx, yy = np.meshgrid(
        block.x.values,
        block.y.values,
    )

    coords = np.column_stack(
        [xx[valid], yy[valid]]
    ).astype(np.float32)

    if coords_mean is not None and coords_std is not None:
        coords = (
            coords - np.asarray(coords_mean, dtype=np.float32)
        ) / np.asarray(coords_std, dtype=np.float32)

    model_dtype, model_device = get_model_dtype_device(model)

    predictive = pyro.infer.Predictive(
        model.model,
        guide=model.guide,
        num_samples=n_samples,
        parallel=True,
        return_sites=("y",),
    )

    means = np.empty(
        (X.shape[0], n_species),
        dtype=np.float32,
    )

    sds = np.empty(
        (X.shape[0], n_species),
        dtype=np.float32,
    )

    for start in range(0, X.shape[0], pixel_batch_size):
        end = min(start + pixel_batch_size, X.shape[0])
        batch_n = end - start

        X_batch = torch.as_tensor(
            X[start:end],
            dtype=model_dtype,
            device=model_device,
        )

        coords_batch = torch.as_tensor(
            coords[start:end],
            dtype=model_dtype,
            device=model_device,
        )

        # Only needed because model.model expects a Y argument.
        # During training=False it should not condition on these values.
        Y_batch = torch.zeros(
            (batch_n, n_species),
            dtype=model_dtype,
            device=model_device,
        )

        with torch.inference_mode():
            predictive_samples = predictive(
                X_batch,
                Y_batch,
                coords_batch,
                training=False,
            )

        y_samples = predictive_samples["y"].detach()

        # Expected shape:
        # n_samples × batch_n × n_species
        y_samples = y_samples.squeeze()

        if y_samples.ndim == 2 and n_species == 1:
            y_samples = y_samples.unsqueeze(-1)

        if y_samples.shape[1:] == (n_species, batch_n):
            y_samples = y_samples.transpose(1, 2)

        expected_shape = (batch_n, n_species)

        if tuple(y_samples.shape[1:]) != expected_shape:
            raise ValueError(
                f"Unexpected predictive y shape: {tuple(y_samples.shape)}. "
                f"Expected (n_samples, {batch_n}, {n_species})."
            )

        y_mean = y_samples.mean(dim=0)
        y_sd = y_samples.std(dim=0, correction=0)

        means[start:end, :] = y_mean.cpu().numpy()
        sds[start:end, :] = y_sd.cpu().numpy()

    prediction[:, valid] = means.T
    prediction_sd[:, valid] = sds.T

    return xr.Dataset(
        {
            "prediction": (("species", "y", "x"), prediction),
            "prediction_sd": (("species", "y", "x"), prediction_sd),
        },
        coords={
            "species": species_names,
            "y": block.y,
            "x": block.x,
        },
    )

species_names = dataset.taxon_names

base = predictors.isel(feature=0, drop=True).astype("float32")

template = xr.Dataset(
    {
        "prediction": xr.full_like(
            base,
            np.nan,
            dtype=np.float32,
        ).expand_dims(species=species_names),

        "prediction_sd": xr.full_like(
            base,
            np.nan,
            dtype=np.float32,
        ).expand_dims(species=species_names),
    }
)

#PRETRAINED_DIR = "raster_to_raster/test/pretrained_model/"
param_store_path = os.path.join(PRETRAINED_DIR, "param_store.pt") #"raster_to_raster/test/pretrained_model/param_store.pt"

pyro.clear_param_store()

param_store_state = torch.load(
    param_store_path,
    map_location="cpu",
    weights_only=False,
)

pyro.get_param_store().set_state(param_store_state)

param_store_state = pyro.get_param_store().get_state()

#print(list(pyro.get_param_store().keys()))
#print(param_store_state)
template = template.transpose("species", "y", "x")

#MODEL_DIR = "raster_to_raster/test/pretrained_model/"
model = torch.load(os.path.join(PRETRAINED_DIR, "model.pt"), weights_only=False)
#print(list(pyro.get_param_store().keys()))

coords_mean = dataset.coords_mean.detach().cpu().numpy()
coords_std = dataset.coords_std.detach().cpu().numpy()

#print("model attrs:", model.__dict__.keys())
#print("dataset attrs:", dataset.__dict__.keys())

predicted = xr.map_blocks(
    predict_gp_chunk,
    predictors,
    kwargs={
        "model": model,
        "feature_names": feature_names,
        "species_names": species_names,
        "training_mean": dataset.X_continuous_mean,
        "training_std": dataset.X_continuous_std,
        "coords_mean": coords_mean,
        "coords_std": coords_std,
        "pixel_batch_size": PIXEL_BATCH_SIZE,
        "param_store_state": param_store_state,
        "n_samples": args.n_samples,
    },
    template=template,
)

#print(predicted)

import rasterio
from dask.diagnostics import ProgressBar

mean_stack = predicted["prediction"].transpose("species", "y", "x")

mean_stack = mean_stack.rio.write_crs(predictors.rio.crs)
mean_stack = mean_stack.rio.write_transform(predictors.rio.transform())
mean_stack = mean_stack.rio.write_nodata(np.nan)

mean_stack = mean_stack.rename({"species": "band"})
mean_stack = mean_stack.assign_coords(
    band=np.arange(1, len(species_names) + 1)
)

std_stack = predicted["prediction_sd"].transpose("species", "y", "x")

std_stack = std_stack.rio.write_crs(predictors.rio.crs)
std_stack = std_stack.rio.write_transform(predictors.rio.transform())
std_stack = std_stack.rio.write_nodata(np.nan)

std_stack = std_stack.rename({"species": "band"})
std_stack = std_stack.assign_coords(
    band=np.arange(1, len(species_names) + 1)
)

import torch
from dask.distributed import Client
from dask.diagnostics import ProgressBar

# imports
# function definitions:
# - load_predictor_stack_from_directory
# - predict_gp_chunk
# - get model class
# etc.


def main():
    torch.set_num_threads(1)

    client = Client(
        n_workers=THREADS,
        threads_per_worker=1,
        memory_limit="20GB",
        dashboard_address=":0",   # use any free dashboard port
    )

    #print(client)

    with ProgressBar():
        mean_stack.rio.to_raster(
            OUTPUT_TIF, #"prediction_mean_stack.tif",
            dtype="float32",
            tiled=True,
            compress="deflate",
            lock=True,
        )

    with ProgressBar():
        std_stack.rio.to_raster(
            OUTPUT_SD_TIF,
            dtype="float32",
            tiled=True,
            compress="deflate",
            lock=True,
        )

    client.close()

    #with rasterio.open("prediction_mean_stack.tif", "r+") as dst:
    with rasterio.open(OUTPUT_TIF, "r+") as dst:
       for i, name in enumerate(species_names, start=1):
           dst.set_band_description(i, str(name))

    with rasterio.open(OUTPUT_SD_TIF, "r+") as dst:
       for i, name in enumerate(species_names, start=1):
           dst.set_band_description(i, str(name))

if __name__ == "__main__":
    main()

#with ProgressBar():
#    mean_stack.rio.to_raster(
#        "raster_to_raster/test/results/prediction_mean_stack.tif",
#        dtype="float32",
#        tiled=True,
#        compress="deflate",
#        lock=True,
#    )

#with rasterio.open("prediction_mean_stack.tif", "r+") as dst:
#    for i, name in enumerate(species_names, start=1):
#        dst.set_band_description(i, str(name))

# raster_to_raster/test/results/prediction_mean_stack.tif