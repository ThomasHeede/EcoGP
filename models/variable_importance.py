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

def _to_cpu_np(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else t

if __name__ == "__main__":
    # ---- inputs / knobs ----

    import importlib
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--save_model_path", type=str, help="Path to save the trained model.")

    args = parser.parse_args()
    save_model_path = args.save_model_path

    pyro.clear_param_store()
    dataset = torch.load(os.path.join(save_model_path, "dataset.pt"),
                         map_location="cpu", weights_only=False)

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
        importance = ((gamma ** 2 * outputscale ** 2) @ (1 / lengthscale)).detach()

        pd.DataFrame(_to_cpu_np(importance), columns=dataset.env_names, index=dataset.traits_names) \
            .to_csv(os.path.join(save_model_path, "env_traits_importance.csv"), index=True)

    else:
        w = pyro.param("w_loc").T

        lengthscale = model.f.covar_module.base_kernel.lengthscale.squeeze()
        outputscale = model.f.covar_module.outputscale

        importance = ((w ** 2 * outputscale ** 2) @ (1 / lengthscale)).detach()

        pd.DataFrame(_to_cpu_np(importance), columns=dataset.env_names, index=dataset.taxon_names) \
            .to_csv(os.path.join(save_model_path, "env_taxon_importance.csv"), index=True)
