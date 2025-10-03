import os
import torch

# from configs.data_folder_path import data_folder_path
from configs.base_path import base_path

config = {
    "data": {
        "X_path": os.path.join(base_path, "data/clean/X.csv"),
        "Y_path": os.path.join(base_path, "data/clean/Y.csv"),
        "coords_path": os.path.join(base_path, "data/clean/XY.csv"),
        "traits_path": os.path.join(base_path, "data/clean/traits.csv"),
        "normalize_X": True,
        "prevalence_threshold": 0.0,
        "total_counts_path": os.path.join(base_path, "data/clean/total_counts.csv"),
        "hierarchy_path": os.path.join(base_path, "data/clean/genome_taxonomy.csv"),
    },
    "general": {
        "n_iter": 1000,
        "n_particles": 1,
        "lr": 0.01,
        "batch_size": 512,
        "train_pct": 0.8,
        "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        "verbose": True,
        "save_model_path": os.path.join(base_path, "results/saved_models/"),
    },
    "environmental": {
        "n_latents": 10,
        "n_inducing_points": 200,
    },
    "spatial": {
        "n_latents": 5,
        "n_inducing_points": 500,
    },
    "hmsc": {
        "k_folds": 5,
        "cross_validation": False,
        "likelihood": "bernoulli",
    },
    "additive": {  # To specify if certain components should be included or omitted.
        "environment": True,
        "spatial": False,
        "traits": False,
    }
}
