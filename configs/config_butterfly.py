import os
import torch

from configs.data_folder_path import data_folder_path
from models.likelihoods import DirichletMultinomialLikelihood, BernoulliLikelihood


config = {
    "data": {
        "X_path": os.path.join(data_folder_path, "butterfly/X_val.csv"),
        "Y_path": os.path.join(data_folder_path, "butterfly/Y_val.csv"),
        "coords_path": os.path.join(data_folder_path, "butterfly/XY_val.csv"),
        "traits_path": os.path.join(data_folder_path, "butterfly/traits.csv"),
        "normalize_X": True,
        "presence_absence": True
    },
    "general": {
        "likelihood": BernoulliLikelihood,
        "n_iter": 100,
        "n_particles": 1,
        "lr": 0.01,
        "batch_size": 512,
        "split_pct": [0.7, 0.2, 0.1],  # Train/Test/Val
        "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        "verbose": True,
        "save_model_path": os.path.join(data_folder_path, "../results/butterfly/"),
        "seed": 0,
    },
    "environmental": {
        "n_latents": 10,
        "n_inducing_points": 200,
    },
    "spatial": {
        "n_latents": 10,
        "n_inducing_points": 50,
    },
    "hmsc": {
        "k_folds": 5,
        "cross_validation": False,
        "likelihood": "bernoulli",
    },
    "additive": {  # To specify if certain components should be included or omitted.
        "environment": True,
        "spatial": True,
        "traits": False,
    }
}
