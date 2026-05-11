import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pyro
import torch
from torch.utils.data import DataLoader
import tqdm
import pandas as pd

import argparse

# from models.EcoGP import EcoGP
from models.DataLoad import DataLoad
from models.DataSampler import DataSampler

parser = argparse.ArgumentParser(description="")

parser.add_argument("--input-dir", help="Folder containing CSV files to make predictions on. (default: raster_to_raster/data/model_data)", default="raster_to_raster/data/model_data", type=str)
parser.add_argument("--output-dir", help="Directory for output CSVs (default: raster_to_raster/data/output_raster)", default="raster_to_raster/data/output_raster", type=str)
parser.add_argument("--pretrained-dir", help="Directory for pretrained model (default: raster_to_raster/pretrained_model)", default="raster_to_raster/pretrained_model", type=str)
parser.add_argument("--n-samples", help="Number of samples (default: 100)", default=100, type=int)
parser.add_argument("--batch-size", help="Batch size to speed up predictions (default: 64)", default=64, type=int)

args = parser.parse_args()

IN_DIR = args.input_dir
OUT_DIR = args.output_dir
PRETRAINED_DIR = args.pretrained_dir
N_SAMPLES = args.n_samples
BATCH_SIZE = args.batch_size


X_path = os.path.join(IN_DIR, "X.csv")
coords_path = os.path.join(IN_DIR, "XY.csv")
coords_path = coords_path if os.path.exists(coords_path) else ""

training_dataset = torch.load(os.path.join(PRETRAINED_DIR, "dataset.pt"), weights_only=False)

data = DataLoad(
    Y_path="",
    X_path=X_path,
    coords_path=coords_path,
    traits_path="",
    device=training_dataset.device,
    normalize_X=False,
    total_counts_path="",
    presence_absence_Y=True,
    verbose=training_dataset.verbose
)

raster_dataset = DataSampler(data)

raster_dataset.n_species = training_dataset.n_species
raster_dataset.taxon_names = training_dataset.taxon_names

if training_dataset.using_traits:
    raster_dataset.traits = training_dataset.traits
    raster_dataset.using_traits = True
    raster_dataset.n_traits = raster_dataset.traits.shape[1]

if training_dataset.normalize_X:
    raster_dataset.X[:, training_dataset.X_continuous] = (
            (raster_dataset.X[:, training_dataset.X_continuous] - training_dataset.X_continuous_mean)
            / training_dataset.X_continuous_std)

if training_dataset.normalize_X and coords_path:
    raster_dataset.coords = (raster_dataset.coords - training_dataset.coords_mean) / training_dataset.coords_std

raster_dataset.Y = torch.zeros((len(raster_dataset), len(training_dataset.taxon_names)), dtype=torch.float32).to(
    training_dataset.device)  # Dummy Y for prediction

##### LOAD MODEL AND PREDICT #####
model = torch.load(os.path.join(PRETRAINED_DIR, "model.pt"), weights_only=False)
pyro.get_param_store().set_state(torch.load(os.path.join(PRETRAINED_DIR, "param_store.pt"), weights_only=False))

predictive = pyro.infer.Predictive(model.model, guide=model.guide, num_samples=N_SAMPLES, parallel=True)

predict_dataloader = DataLoader(dataset=raster_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False)
y_prob_means = []
for idx in tqdm.tqdm(predict_dataloader):
    X, Y, coords, traits = raster_dataset.get_batch_data(idx)

    predictive_samples = predictive(X, Y, coords, traits, training=False)

    y_prob_means.append(predictive_samples["y"].squeeze().mean(dim=0))

y_prob_means = torch.cat(y_prob_means)

df = pd.DataFrame(y_prob_means, index=raster_dataset.site_names, columns=training_dataset.taxon_names)
df.to_csv(os.path.join(OUT_DIR, "Y_pred.csv"))

# cd EcoGP/
# python raster_to_raster/preds_from_pretrained.py --input-dir-csv raster_to_raster/data_test/model_data