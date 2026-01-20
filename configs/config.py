import os
import torch

# from configs.data_folder_path import data_folder_path
from configs.base_path import base_path
from EcoGP.likelihoods import DirichletMultinomialLikelihood, BernoulliLikelihood

config = {
    "data": {
        "X_path": os.path.join(base_path, "data/clean/X.csv"),
        "Y_path": os.path.join(base_path, "data/clean/Y.csv"),
        "coords_path": os.path.join(base_path, "data/clean/XY.csv"),
        "traits_path": os.path.join(base_path, "data/clean/traits.csv"),
        "normalize_X": True,
        "prevalence_threshold": 0.0,
        "total_counts_path": "",#os.path.join(base_path, "data/clean/total_counts.csv"),
        "presence_absence": True,
    },
    "general": {
        "likelihood": BernoulliLikelihood,  # TODO: Overwritten for testing
        "n_iter": 1_000,  # TODO: Overwritten for testing
        "n_particles": 1,
        "lr": 0.01,  # TODO: Overwritten for testing
        "batch_size": 2048,
        "split_pct": [0.7, 0.2, 0.1],  # Train/Test/Val
        "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        "verbose": True,
        "save_model_path": os.path.join(base_path, "results/saved_models/"),
        "seed": 2,
    },
    "environmental": {
        "n_latents": 20,  # TODO: Overwritten for testing
        "n_inducing_points": 50,  # TODO: Overwritten for testing
    },
    "spatial": {
        "n_latents": 5,  # TODO: Overwritten for testing
        "n_inducing_points": 50,  # TODO: Overwritten for testing
    },
    "hmsc": {
        "k_folds": 5,
        "cross_validation": False,
        "likelihood": "bernoulli",
    },
}
