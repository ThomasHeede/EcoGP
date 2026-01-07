import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tqdm

import wandb
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.EcoGP import EcoGP

from torch.utils.data import DataLoader, random_split
from models.DataSampler import DataSampler
from models.DataLoad import DataLoad
from models.BetaTraceELBO import BetaTraceELBO

from sklearn import metrics

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current device idx:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))

if __name__ == "__main__":
    import importlib
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class = argparse.RawTextHelpFormatter
        )

    # To override arguments from config

    ## Section "data" in the config
    io_group = parser.add_argument_group("I/O options")
    io_group.add_argument(
        "--config",
        type = str,
        metavar = "</configs/my_config>",
        default = "",  # TODO: Change config here or when running in terminal
        help = "Name of the config file (without .py extension, must be in configs/). If you use a config file here there is no need to se the other options.\n\n",
        )
    io_group.add_argument(
        '-x', '--x_mat',
        type = str,
        metavar = "x.csv",
        help = "Matrix of the environmental features (.csv).\nThe columns are variables and the rows are samples. Unique IDs are expected as column names for the variables and row names for the samples.\nIts presence enables the environmental part of the model - and it is mandatory."
        )
    io_group.add_argument(
        '-y', '--y_mat',
        type = str,
        metavar = "y.csv",
        help = "Matrix of the taxa observations (.csv).\nThe columns are taxa and the rows are samples. Unique IDs are expected as column names for the taxa and row names for the samples."
        )
    io_group.add_argument(
        '-c', '--coords',
        type = str,
        metavar = "coords.csv",
        help = "Matrix of the samples' coordinates (.csv).\nThe columns are longitude and latitude, the rows are sample. The samples have unique IDs as row names\nIts presence enables the spatial part of the model."
        )
    io_group.add_argument(
        '-t', '--traits',
        type = str,
        metavar = "traits.csv",
        default="",
        help = "Matrix of the traits (.csv).\nThe columns are different traits and the rows taxa. Unique trait names and taxa names are expected as column and row names.\nIts presence enables the traits inclusion in the model."
        )
    io_group.add_argument(
        '--normalize_X',
        type = bool,
        metavar = "[True|False]",
        help = "Does the environmental variable matrix need to be normalized? Default = True"
        )
    io_group.add_argument(
        '--presence_absence',
        type = bool,
        metavar = "[True|False]",
        help = "Run the analysis as Presence/Absence? Default = True."
        )
    io_group.add_argument(
        '-o', '--out',
        type = str,
        metavar = "</out_path/>",
        help = "Path to the output folder.\n\n"
        )

    ## Section "environmental" in the config
    env_group = parser.add_argument_group("Environmental component of the model")
    env_group.add_argument(
        '--n_latents_env',
        type = int,
        metavar = "N",
        help = "Number of latent variables to encode the environmental data. Default = 10.",
        default = 10
        )
    env_group.add_argument(
        '--n_inducing_points_env',
        type = int,
        metavar = "N",
        help = "Number of inducing point for learning the Gaussian Processes for the envirnomental component of the model. Default = 50.",
        default = 50
        )

    ## Section "spatial" in the config
    coords_group = parser.add_argument_group("Spatial component of the model")
    coords_group.add_argument(
        '--n_latents_spatial',
        type = int,
        metavar = "N",
        default = 5,
        help = "Number of latent variables to encode the spatial data. Default = 5."
        )
    coords_group.add_argument(
        '--n_inducing_points_spatial',
        type = int,
        metavar = "N",
        default = 50,
        help = "Number of inducing point for learning the Gaussian Processes for the spatial component of the model. Default = 50.")

    ## Section "general" in the config
    mod_group = parser.add_argument_group("Model options")
    mod_group.add_argument(
        '--likelihood',
        type = str,
        metavar = "[Dirichlet|Bernoulli]",
        default = "Bernoulli",
        help = "Choose the Likelihood: either Dirichlet (Multinomial) or Bernoulli. Default = Bernoulli.\n\n"
        )
    mod_group.add_argument(
        '--n_iter',
        type = int,
        metavar = "N",
        default = 500,
        help = "Number of iterations during training. Default = 500.\n\n"
        )
    mod_group.add_argument(
        '--n_particles',
        type = int,
        metavar = "N",
        default = 1,
        help = "Number of particles during training. Default = 1.\n\n"
        )
    mod_group.add_argument(
        '--lr',
        type = float,
        metavar = "N",
        default = 0.005,
        help = "Learning rate during training. The model can be very sensitive to this parameter and even crash for some values of the learnign rate. Default = 0.005.\n\n"
        )
    mod_group.add_argument(
        '--batch_size',
        type = int,
        metavar = "N",
        default = 512,
        help = "Number of data points processed together to infer the Gaussian Processes. Default = 512.\n\n"
        )
    mod_group.add_argument(
        "--split_pct",
        nargs = 3,
        type = float,
        metavar=("N1", "N2", "N3"),
        help = "Fractions of the data to be used for training (N1), testing (N2) and validation (N3) of the model. Default = 0.7 0.2 0.1.\n\n",
        )

    other_group = parser.add_argument_group("Other options")
    other_group.add_argument('--seed',
        type = int,
        metavar = "N",
        default = 123,
        help = "Set a seed to make the runs reproducible.\n\n"
        )
    other_group.add_argument(
        '--device',
        type = str,
        metavar = "[cpu|cuda]",
        default = "cpu",
        help = "Device to work, i.e., specify cuda if available. Defualt = cpu.\n\n"
        )
    other_group.add_argument(
        '--verbose',
        type = bool,
        metavar = "[True|False]",
        default = True,
        help = "Setting verbose mode on or off. Defualt = True.\n\n"
        )

    args = parser.parse_args()

    if args.config:
        print(f"Config File: {args.config}")

        config_module = importlib.import_module(f"configs.{args.config}")
        config = config_module.config  # Import the config module
    else:
        config = {}

    # Overrides config
    ## Section "addictive" in the config, inferred by the presence of the "data" section flags in the CLI
    if args.x_mat:
        config["additive"] = {}
        config["additive"]["environment"] = True
    if args.coords:
        config["additive"]["spatial"] = True
    else:
        config["additive"]["spatial"] = False
    if args.traits:
        config["additive"]["traits"] = True
    else:
        config["additive"]["traits"] = False
    
    ## Section "data" in the config
    if args.x_mat:
        config["data"] = {}
        config["data"]["X_path"] = args.x_mat
    if args.y_mat:
        config["data"]["Y_path"] = args.y_mat
    if args.coords:
        config["data"]["coords_path"] = args.coords
    if args.traits:
        config["data"]["traits_path"] = args.traits
    if args.normalize_X:
        config["data"]["normalize_X"] = args.normalize_X
    if args.presence_absence:
        config["data"]["presence_absence"] = args.presence_absence

    ## Section "environmental" in the config
    if args.n_latents_env:
        config["environmental"] = {}
        config["environmental"]["n_latents"] = args.n_latents_env
    if args.n_inducing_points_env:
        config["environmental"]["n_inducing_points"] = args.n_inducing_points_env

    ## Section "spatial" in the config
    if args.n_latents_spatial:
        config["spatial"] = {}
        config["spatial"]["n_latents"] = args.n_latents_spatial
    if args.n_inducing_points_spatial:
        config["spatial"]["n_inducing_points"] = args.n_inducing_points_spatial

    ## Section "general" in the config
    if any([args.seed, args.out, args.verbose, args.likelihood, args.n_iter, args.n_particles, args.lr, args.batch_size, args.split_pct]):
        config["general"] = {}
    if args.seed is not None:
        config["general"]["seed"] = args.seed
    if args.out:
        config["general"]["save_model_path"] = args.out
    if args.verbose:
        config["general"]["verbose"] = args.verbose
    if args.likelihood:
        config["general"]["likelihood"] = args.likelihood
    if args.n_iter:
        config["general"]["n_iter"] = args.n_iter
    if args.n_particles:
        config["general"]["n_particles"] = args.n_particles
    if args.lr:
        config["general"]["lr"] = args.lr
    if args.batch_size:
        config["general"]["batch_size"] = args.batch_size
    if args.split_pct:
        config["general"]["split_pct"] = args.split_pct
    if args.device:
        config["general"]["device"] = args.device
    
    

    # ARGUMENTS
    environment = config["additive"]["environment"]
    spatial = config["additive"]["spatial"]
    traits = config["additive"]["traits"]

    x_path = config["data"]["X_path"]
    y_path = config["data"]["Y_path"]
    coords_path = config["data"]["coords_path"]
    if "traits_path" in config["data"]:
        traits_path = config["data"]["traits_path"]
    else:
            traits_path = ""
    # total_counts_path = config["data"]["total_counts_path"]
    #hierarchy_path = config["data"]["hierarchy_path"]

    n_latents_env = config["environmental"]["n_latents"]
    n_latents_spatial = config["spatial"]["n_latents"]
    n_iter = config["general"]["n_iter"]
    n_particles = config["general"]["n_particles"]
    device = config["general"]["device"]
    lr = config["general"]["lr"]
    batch_size = config["general"]["batch_size"]
    split_pct = config["general"]["split_pct"]
    n_inducing_points_env = config["environmental"]["n_inducing_points"]
    n_inducing_points_spatial = config["spatial"]["n_inducing_points"]

    verbose = config["general"]["verbose"]
    presence_absence = config["data"]["presence_absence"]
    normalize_X = config["data"]["normalize_X"]
    likelihood = config["general"]["likelihood"]
    seed = config["general"]["seed"]

    # prevalence_threshold = config["data"]["prevalence_threshold"]

    save_model_path = config["general"]["save_model_path"]
    # STOP ARGUMENTS

    torch.manual_seed(seed)

    data = DataLoad(
        Y_path=y_path,
        X_path=x_path,
        coords_path=coords_path,
        traits_path=traits_path,
        device=device,
        normalize_X=normalize_X,
        #total_counts_path=total_counts_path,
        presence_absence_Y=presence_absence,
        verbose=verbose
    )

    dataset = DataSampler(data)

    if spatial:
        train_indices, test_indices, validation_indices = random_split(torch.arange(dataset.unique_coords.shape[0]),
                                                                       split_pct,
                                                                       generator=torch.Generator().manual_seed(seed))

        # Getting the spatial locations split into separate sets
        train_indices = dataset.coords_inverse_indicies[
            torch.isin(dataset.coords_inverse_indicies, torch.tensor(train_indices.indices))]
        test_indices = dataset.coords_inverse_indicies[
            torch.isin(dataset.coords_inverse_indicies, torch.tensor(test_indices.indices))]
        validation_indices = dataset.coords_inverse_indicies[
            torch.isin(dataset.coords_inverse_indicies, torch.tensor(validation_indices.indices))]

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        validation_dataset = torch.utils.data.Subset(dataset, validation_indices)
    else:
        train_dataset, test_dataset, validation_dataset = random_split(dataset, split_pct,
                                                                       generator=torch.Generator().manual_seed(seed))

    # Make sure at least 1 species obserservations are present all splits
    # Can't make predictions for a species not present in training
    keep_y = (dataset.Y[train_dataset.indices].sum(dim=0) >= split_pct[0] * 10) & (
                dataset.Y[test_dataset.indices].sum(dim=0) >= split_pct[1] * 10) & (
                dataset.Y[validation_dataset.indices].sum(dim=0) >= split_pct[2] * 10)
    dataset.Y = dataset.Y[:, keep_y]
    dataset.taxon_names = dataset.taxon_names[keep_y]
    dataset.n_species = dataset.Y.shape[1]
    if traits_path:
        dataset.traits = dataset.traits[keep_y, :]
    if verbose:
        print(f"Keeping {keep_y.sum().item()} taxons with at least {split_pct} * 10 "
              f"observations per split, respectively.")

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    n_tasks = dataset.n_species
    n_variables = dataset.n_env
    # n_traits = dataset.n_traits
    unique_coordinates = dataset.unique_coords[
        dataset.get_dist_idx_reverse(train_dataset.indices)[0]] if spatial else None

    model = EcoGP(
        n_latents_env,
        n_variables,
        n_inducing_points_env,
        n_latents_spatial,
        n_inducing_points_spatial,
        unique_coordinates,
        environment=environment,
        spatial=spatial,
        traits=traits,
        likelihood=likelihood
    ).to(device)

    optimizer = pyro.optim.Adam({"lr": lr})
    # elbo = pyro.infer.Trace_ELBO(num_particles=n_particles, vectorize_particles=True, retain_graph=True)

    elbo = BetaTraceELBO(beta=.5, num_particles=n_particles, vectorize_particles=True, retain_graph=True)

    svi = pyro.infer.SVI(model.model, model.guide, optimizer, elbo)

    model.train()

    losses = []

    iterator = tqdm.tqdm(range(n_iter))
    for i in iterator:
        loss = 0
        for idx in train_dataloader:
            batch = train_dataset.dataset.get_batch_data(idx)
            loss += svi.step(batch) / batch.get("Y").nelement()

        iterator.set_postfix(loss=loss)
        losses.append(loss)

    plt.plot(list(range(n_iter)), losses)
    plt.show()

    # Save model
    if save_model_path:
        torch.save(model, os.path.join(save_model_path, "model.pt"))
        pyro.get_param_store().save(os.path.join(save_model_path, "param_store.pt"))
        torch.save(dataset, os.path.join(save_model_path, "dataset.pt"))

        # Save config
        import pprint

        with open(os.path.join(save_model_path, 'config.txt'), 'w') as f:
            # Create a PrettyPrinter object that writes to the file
            pp = pprint.PrettyPrinter(stream=f)
            pp.pprint(config)

        # Testing
        test_dataloader = DataLoader(dataset=test_dataset,
                                     batch_size=batch_size,
                                     shuffle=True)

    prob_list = []
    y_test_list = []
    for idx in test_dataloader:
        batch = test_dataset.dataset.get_batch_data(idx)
        res = model.forward(batch).detach()

        prob_list.append(res)
        y_test_list.append(batch.get("Y") / (dataset.total_counts[idx] if dataset.using_total_counts else 1))

    prob = torch.concat(prob_list)
    test_Y = torch.concat(y_test_list)
    del prob_list, y_test_list

    if save_model_path:
        import pandas as pd

        pd.DataFrame(prob, columns=dataset.taxon_names, index=dataset.site_names[test_dataset.indices]).to_csv(os.path.join(save_model_path, "Y_pred.csv"))
        pd.DataFrame(test_Y, columns=dataset.taxon_names, index=dataset.site_names[test_dataset.indices]).to_csv(os.path.join(save_model_path, "Y_true.csv"))

    from models.misc.calculate_metrics import calculate_metrics
    from models.misc.calculate_metrics import calculate_metric_averages

    metrics_per_species = calculate_metrics(test_Y, prob)
    
    if save_model_path:
        import pandas as pd

        pd.DataFrame(metrics_per_species, columns=metrics_per_species.keys(), index=dataset.taxon_names).to_csv(os.path.join(save_model_path, "metrics_per_taxon.csv"))

    metrics = calculate_metric_averages(metrics_per_species)
    print(metrics)

    # # Validation
    # validation_dataloader = DataLoader(dataset=validation_dataset,
    #                              batch_size=batch_size,
    #                              shuffle=True)
    #
    # prob_list = []
    # y_validation_list = []
    # for idx in validation_dataloader:
    #     batch = test_dataset.dataset.get_batch_data(idx)
    #     res = model.forward(batch).detach()
    #
    #     prob_list.append(res)
    #     y_validation_list.append(batch.get("Y") / (dataset.total_counts[idx] if dataset.using_total_counts else 1))
    #
    # prob = torch.concat(prob_list)
    # validation_Y = torch.concat(y_validation_list)
    # del prob_list, y_validation_list
    #
    # torch.save(prob, os.path.join(save_model_path, "Y_pred_validation.pt"))
    # torch.save(validation_Y, os.path.join(save_model_path, "Y_true_validation.pt"))

    print("Done")
