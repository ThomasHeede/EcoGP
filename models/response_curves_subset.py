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


def _move_to_device(obj, device):
    # minimal "best effort" mover for dataset-like objects
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        t = [ _move_to_device(x, device) for x in obj ]
        return type(obj)(t)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj

def _worker_run_one_variable(args):
    (var_idx, n_samples, n_values, iter_list, batch_size, save_model_path, device_str) = args

    device = torch.device(device_str)
    torch.cuda.set_device(device)

    pyro.clear_param_store()

    model = torch.load(os.path.join(save_model_path, "model.pt"),
                    map_location=device, weights_only=False)
    dataset = torch.load(os.path.join(save_model_path, "dataset.pt"),
                        map_location=device, weights_only=False)
    state = torch.load(os.path.join(save_model_path, "param_store.pt"),
                    map_location="cpu", weights_only=False)

    if isinstance(state, dict) and "params" in state:
        state["params"] = {k: v.to(device) for k, v in state["params"].items()}

    pyro.get_param_store().set_state(state)

    # extra safety: ensure any already-initialized params are on device
    store = pyro.get_param_store()
    for k, v in list(store.items()):
        if torch.is_tensor(v) and v.device != device:
            store[k] = v.to(device)

    model = model.to(device)
    if hasattr(model, "f") and hasattr(model.f, "to"):
        model.f = model.f.to(device)

    model.spatial = False
    model.n_latents_spatial = None
    
    predictive = pyro.infer.Predictive(
        model.model,
        guide=model.guide,
        num_samples=n_samples,
        return_sites=("z",),
    )

    # x-grid on GPU
    diff_env_inputs = torch.linspace(
        dataset.X[:, var_idx].min(),
        dataset.X[:, var_idx].max(),
        n_values,
        device=device,
    )

    # Unstandardize x-values (your logic; kept as-is)
    A_ = dataset.X.to(torch.float32)
    cont = torch.isfinite(A_).all(0) & (A_ != torch.floor(A_)).all(0)
    kept = torch.nonzero(cont, as_tuple=True)[0]
    inv = torch.full((A_.size(1),), -1, dtype=torch.long, device=device)
    inv[kept] = torch.arange(kept.numel(), device=device)
    new_idx = inv[var_idx].item()

    x_values = torch.round(
        diff_env_inputs * dataset.X_continuous_std[new_idx] + dataset.X_continuous_mean[new_idx],
        decimals=2
    )

    n_species = dataset.Y.size(1)
    sum_mean  = torch.zeros((n_values, n_species), device=device)
    sum_above = torch.zeros((n_values, n_species), device=device)
    sum_below = torch.zeros((n_values, n_species), device=device)
    total_i = 0

    # Faster than no_grad; also avoids autograd overhead
    with torch.inference_mode():
        for start in range(0, len(iter_list), batch_size):
            idx = iter_list[start:start + batch_size]
            b = len(idx)
            total_i += b

            x0 = dataset.X[idx, :]                         # (b, n_env) on GPU
            X = x0.repeat_interleave(n_values, dim=0)      # (b*n_values, n_env)

            X[:, var_idx] = diff_env_inputs.repeat(b)

            z = predictive(
                X=X,
                Y=torch.ones(X.size(0), n_species, device=device),
                coords=None,
                traits=dataset.traits,
                training=False,
            )["z"].squeeze()                               # (S, b*n_values, Sp)

            z = z.view(n_samples, b, n_values, n_species)  # (S, b, V, Sp)

            logits_mean = z.mean(dim=0)                    # (b, V, Sp)
            logits_std  = z.std(dim=0)                     # (b, V, Sp)

            p_mean  = torch.sigmoid(logits_mean)
            p_above = torch.sigmoid(logits_mean + 2 * logits_std)
            p_below = torch.sigmoid(logits_mean - 2 * logits_std)

            sum_mean  += p_mean.sum(dim=0)
            sum_above += p_above.sum(dim=0)
            sum_below += p_below.sum(dim=0)

    mean  = (sum_mean  / total_i).detach().cpu().numpy()
    above = (sum_above / total_i).detach().cpu().numpy()
    below = (sum_below / total_i).detach().cpu().numpy()
    x_np  = x_values.detach().cpu().numpy().reshape(-1)

    species = list(dataset.taxon_names)
    df_one = pd.DataFrame({
        "env.variable": dataset.env_names[var_idx],
        "variable_idx": var_idx,
        "x_value":      np.repeat(x_np, len(species)),
        "species":      np.tile(species, len(x_np)),
        "mean":         mean.reshape(-1),
        "above":        above.reshape(-1),
        "below":        below.reshape(-1),
    })
    return df_one


if __name__ == "__main__":
    import os
    import argparse
    import multiprocessing as mp

    import numpy as np
    import pandas as pd
    import torch
    import pyro

    import os, subprocess, torch
    print("CUDA_VISIBLE_DEVICES =", os.getenv("CUDA_VISIBLE_DEVICES"))
    print("torch.version.cuda   =", torch.version.cuda)
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    print("torch.cuda.device_count() =", torch.cuda.device_count())

    try:
        subprocess.run(["nvidia-smi"], check=False)
    except FileNotFoundError:
        print("nvidia-smi not found in PATH")

    # ---- args ----
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_model_path",
        required=True,
        type=str,
        help="Path containing model.pt, dataset.pt, param_store.pt",
    )
    parser.add_argument(
        "--importance",
        required=True,
        type=str,
        help="Variable importance CSV with columns: tax_level,taxon,env.variable,importance.mean,importance.sd,rank",
    )
    args = parser.parse_args()
    save_model_path = args.save_model_path

    # ---- load dataset (CPU) ----
    pyro.clear_param_store()
    dataset = torch.load(os.path.join(save_model_path, "dataset.pt"),
                         map_location="cpu", weights_only=False)

    taxon_names = set(dataset.taxon_names)
    env_names = list(dataset.env_names)
    env_name_to_idx = {name: i for i, name in enumerate(env_names)}

    # ---- read + filter importance ----
    importance_df = (
        pd.read_csv(args.importance)
          .sort_values(["tax_level", "taxon", "rank"])
          .reset_index(drop=True)
    )

    # env.variable (string) -> env column index (int)
    importance_df["env_index"] = importance_df["env.variable"].map(env_name_to_idx)

    # keep only continuous env columns (your criterion)
    continuous_idxs = torch.where(
        torch.isfinite(dataset.X).all(0) & (dataset.X != torch.floor(dataset.X)).all(0)
    )[0].tolist()
    continuous_idxs = set(int(x) for x in continuous_idxs)

    filtered = importance_df[
        importance_df["env_index"].notna()
        & importance_df["env_index"].astype(int).isin(continuous_idxs)
        & importance_df["taxon"].isin(taxon_names)
    ].copy()
    filtered["env_index"] = filtered["env_index"].astype(int)

    # ---- UNIQUE selected variables from filtered importance ----
    selected_var_idxs = sorted(filtered["env_index"].drop_duplicates().tolist())
    if len(selected_var_idxs) == 0:
        raise RuntimeError("No variables left after filtering importance file vs dataset env/taxa/continuous columns.")

    # ---- compute response curves (GPU) ----
    n_samples = 100
    n_values  = 250

    K = 100
    iter_list = list(range(min(K, dataset.X.shape[0])))   # safe if dataset has < K rows
    batch_size = 10                                       # tune for VRAM (try 5 if OOM)

    n_gpus = torch.cuda.device_count()
    if n_gpus < 1:
        raise RuntimeError("No CUDA GPU visible to PyTorch. (torch.cuda.device_count()==0)")

    # One process per GPU (good default). Each process handles many variables sequentially.
    # Here: distribute variables across GPUs by assigning device per task.
    devices = [f"cuda:{i}" for i in range(n_gpus)]
    args_iterable = [
        (vidx, n_samples, n_values, iter_list, batch_size, save_model_path, devices[i % n_gpus])
        for i, vidx in enumerate(selected_var_idxs)
    ]

    mp.set_start_method("spawn", force=True)
    n_procs = min(n_gpus, len(selected_var_idxs))

    with mp.Pool(processes=n_procs) as pool:
        dfs = list(pool.imap_unordered(_worker_run_one_variable, args_iterable))

    out = pd.concat(dfs, ignore_index=True)
    out.to_csv(os.path.join(save_model_path, "responses_top_env_taxa.csv"), index=False)

    print(f"Done. Wrote {len(out):,} rows to responses_top_env_taxa.csv "
          f"for {len(selected_var_idxs)} variables across {n_gpus} GPU(s).")