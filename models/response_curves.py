import torch
import pyro
import pyro.distributions as dist
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import os
#from configs.config_mfd2_public_noNAs import config

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EcoGP import *
# from models.DataSampler import DataSampler

import plotly.graph_objects as go


def get_response(n_samples, n_values, variable, model, dataset, iter_range=range(100)):
    predictive = pyro.infer.Predictive(model.model, guide=model.guide, num_samples=n_samples)

    diff_env_inputs = torch.linspace(dataset.X[:, variable].min(), dataset.X[:, variable].max(), n_values)

    means = []
    aboves = []
    belows = []

    #for i in tqdm(range(dataset.X.shape[0])):
    for i in iter_range:
    # for i in [1]:
        x = dataset.X[i, :].repeat(n_values, 1)

        x[:, variable] = diff_env_inputs

        range_batch = {'n_samples_batch': n_values, 'n_species': dataset.n_species, 'n_env': dataset.n_env, 'X': x, "training": False}

        samples_z = predictive(range_batch)["z"].squeeze()

        logits_mean = samples_z.mean(dim=0)
        logits_std = samples_z.std(dim=0)

        y_prob_mean = dist.Bernoulli(logits=logits_mean).mean.detach()
        y_prob_above = dist.Bernoulli(logits=logits_mean + 2 * logits_std).mean.detach()
        y_prob_below = dist.Bernoulli(logits=logits_mean - 2 * logits_std).mean.detach()

        means.append(y_prob_mean)
        aboves.append(y_prob_above)
        belows.append(y_prob_below)

    y_prob_mean = torch.stack(means).mean(dim=0)
    y_prob_above = torch.stack(aboves).mean(dim=0)
    y_prob_below = torch.stack(belows).mean(dim=0)

    return y_prob_mean, y_prob_above, y_prob_below


def _to_cpu_np(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else t

def _worker_run_one(args):
    """
    One isolated CPU worker for a single variable_idx.
    Loads model/dataset/param_store fresh to avoid Pyro global state sharing.
    Writes CSVs suffixed with the variable index.
    """
    (var_idx, n_samples, n_values, iter_range, save_model_path) = args

    # Keep threads low per process to avoid oversubscription on CPUs
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    pyro.clear_param_store()

    # CPU-only loads
    model = torch.load(os.path.join(save_model_path, "model.pt"),
                       map_location="cpu", weights_only=False)
    dataset = torch.load(os.path.join(save_model_path, "dataset.pt"),
                         map_location="cpu", weights_only=False)
    state = torch.load(os.path.join(save_model_path, "param_store.pt"),
                       map_location="cpu", weights_only=False)
    pyro.get_param_store().set_state(state)

    model.spatial = False

    # If get_response reads a global iter_range, expose it
    globals()["iter_range"] = iter_range

    # Compute response curves
    mean, above, below = get_response(n_samples, n_values, var_idx, model, dataset)

    # Unstandardize x values for this variable
    diff_env_inputs = torch.linspace(dataset.X[:, var_idx].min(),
                                     dataset.X[:, var_idx].max(),
                                     n_values)
    x_values = torch.round(
        diff_env_inputs * dataset.X_continuous_std[var_idx]
        + dataset.X_continuous_mean[var_idx],
        decimals=2
    )

    # # Save per-variable files
    # pd.DataFrame(_to_cpu_np(mean),  columns=dataset.Y_cols_species) \
    #     .to_csv(os.path.join(save_model_path, f"mean_var{var_idx}.csv"), index=False)
    # pd.DataFrame(_to_cpu_np(above), columns=dataset.Y_cols_species) \
    #     .to_csv(os.path.join(save_model_path, f"above_var{var_idx}.csv"), index=False)
    # pd.DataFrame(_to_cpu_np(below), columns=dataset.Y_cols_species) \
    #     .to_csv(os.path.join(save_model_path, f"below_var{var_idx}.csv"), index=False)
    # pd.DataFrame(_to_cpu_np(x_values)) \
    #     .to_csv(os.path.join(save_model_path, f"x_values_var{var_idx}.csv"),
    #             index=False, header=False)

    # return var_idx

    species = list(dataset.taxon_names)

    mean_np  = _to_cpu_np(mean)          # shape: (n_values, n_species)
    above_np = _to_cpu_np(above)
    below_np = _to_cpu_np(below)
    x_np     = _to_cpu_np(x_values).reshape(-1)  # shape: (n_values,)

    # Flatten to long form: rows = n_values * n_species
    df_one = pd.DataFrame({
        "variable_idx": np.repeat(dataset.env_names[var_idx], len(x_np) * len(species)),
        "x_value":      np.repeat(x_np, len(species)),
        "species":      np.tile(species, len(x_np)),
        "mean":         mean_np.reshape(-1),
        "above":        above_np.reshape(-1),
        "below":        below_np.reshape(-1),
    })

    return df_one


if __name__ == "__main__":
    # ---- inputs / knobs ----

    import importlib
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config_butterfly",  # TODO: Change config here or when running in terminal
        help="Name of the config file (without .py extension, must be in configs/)",
    )

    args = parser.parse_args()

    print(f"Config File: {args.config}")

    config_module = importlib.import_module(f"configs.{args.config}")
    config = config_module.config  # Import the config module

    save_model_path = config["general"]["save_model_path"]

    pyro.clear_param_store()
    dataset = torch.load(os.path.join(save_model_path, "dataset.pt"),
                         map_location="cpu", weights_only=False)

    print("N_env: ", dataset.n_env)
    print("Dim X", dataset.X.shape)
    n_samples = 50
    n_values  = 250
    iter_range = [0]                  # or range(K)
    variable_idxs = list(range(dataset.n_env))   # e.g., 0..9
    print(variable_idxs)

    # ---- compute importance ONCE (CPU) ----
    
    model = torch.load(os.path.join(save_model_path, "model.pt"),
                       map_location="cpu", weights_only=False)
    state = torch.load(os.path.join(save_model_path, "param_store.pt"),
                       map_location="cpu", weights_only=False)
    pyro.get_param_store().set_state(state)
    model.spatial = False

    if dataset.traits is not None:
        gamma = pyro.param("gamma_loc").T

        lengthscale = model.f.covar_module.base_kernel.lengthscale.squeeze()
        outputscale = model.f.covar_module.outputscale
        #print("lengthscale: ", lengthscale.shape)
        #print("outputscale: ", outputscale.shape)
        #print("gamma: ", gamma.shape)
        #print(dataset.traits_names)
        #print(dataset.env_names)
        importance = ((gamma ** 2 * outputscale ** 2) @ (1 / lengthscale)).detach()

        pd.DataFrame(_to_cpu_np(importance), columns=dataset.env_names, index=dataset.traits_names) \
            .to_csv(os.path.join(save_model_path, "env_traits_importance.csv"), index=True)

    else:
        w = pyro.param("w_loc").T

        lengthscale = model.f.covar_module.base_kernel.lengthscale.squeeze()
        outputscale = model.f.covar_module.outputscale

        importance = ((w ** 2 * outputscale ** 2) @ (1 / lengthscale)).detach()

        pd.DataFrame(_to_cpu_np(importance), columns=dataset.env_names, index=dataset.taxon_names) \
            .to_csv(os.path.join(save_model_path, "env_species_importance.csv"), index=True)

        # ---- parallel over variable_idx with multiprocessing.Pool ----
        # Use "spawn" to avoid inheriting Pyro/Torch state; safe on CPU too.
        mp.set_start_method("spawn", force=True)

        # Pick a sensible number of processes (tune for your node)
        n_procs = min(len(variable_idxs), max(1, mp.cpu_count() // 2))

        args_iterable = [
            (vidx, n_samples, n_values, iter_range, save_model_path)
            for vidx in variable_idxs
        ]

        with mp.Pool(processes=n_procs) as pool:
            for vidx_done in pool.imap_unordered(_worker_run_one, args_iterable):
                print(f"[OK] variable_idx") #={vidx_done}")
            
            dfs = list(pool.imap_unordered(_worker_run_one, args_iterable))  # worker returns df_one
            pd.concat(dfs, ignore_index=True).to_csv(os.path.join(save_model_path, "responses_all_vars.csv"), index=False)


        print("All variable_idx jobs finished.")




# if __name__ == "__main__":
#     save_model_path = config["general"]["save_model_path"]

#     # Loading model and setting learned params
#     pyro.clear_param_store()
#     model = torch.load(os.path.join(save_model_path, "model.pt"), weights_only=False)
#     dataset = torch.load(os.path.join(save_model_path, "dataset.pt"), weights_only=False)
#     pyro.get_param_store().set_state(torch.load(os.path.join(save_model_path, "param_store.pt"), weights_only=False))

#     model.spatial = False

#     n_samples = 50  # Number of samples to calculate the individual probabilities from (Higher, to reduce spikes)
#     n_values = 250  # Number of points between minimum and maximum for the chosen variable (Higher, the smoother it will be)
#     variable_idx = -1  # Index for which variable to look at
#     iter_range = [0]  # Sites to include in the calculation of the response curves. Can be multiple as "[0, 1, ...]" or as a "range(10)"

#     # Mean predicted probabilities with +-2 std
#     mean, above, below = get_response(n_samples, n_values, variable_idx)

#     # Converting the features back from standard normalization
#     diff_env_inputs = torch.linspace(dataset.X[:, variable_idx].min(), dataset.X[:, variable_idx].max(), n_values)
#     x_values = torch.round(diff_env_inputs * dataset.X_continuous_std[variable_idx] + dataset.X_continuous_mean[variable_idx], decimals=2)

#     pd.DataFrame(mean.numpy(), columns=dataset.Y_cols_species).to_csv(os.path.join(save_model_path, "mean.csv"), index=False)
#     pd.DataFrame(above.numpy(), columns=dataset.Y_cols_species).to_csv(os.path.join(save_model_path, "above.csv"), index=False)
#     pd.DataFrame(below.numpy(), columns=dataset.Y_cols_species).to_csv(os.path.join(save_model_path, "below.csv"), index=False)

#     pd.DataFrame(x_values.numpy()).to_csv(os.path.join(save_model_path, "x_values.csv"), index=False, header=False)

    # # Calculating "importance"
    # w = pyro.param("w_loc")
    # lengthscale = model.f.covar_module.base_kernel.lengthscale.squeeze()
    # outputscale = model.f.covar_module.outputscale
    # importance = ((w ** 2 * outputscale ** 2) @ (1 / lengthscale)).detach()

#     pd.DataFrame(importance.numpy(), columns=dataset.environmental).to_csv(os.path.join(save_model_path, "importance.csv"), index=False)
