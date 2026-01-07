import torch
import pyro
import pyro.distributions as dist
import gpytorch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tqdm

import wandb
import sys
import os

from MultitaskVariationalStrategy import MultitaskVariationalStrategy
from likelihoods import DirichletMultinomialLikelihood, BernoulliLikelihood


class EcoGP(pyro.nn.PyroModule):
    def __init__(self,
                 n_latents_env=None,
                 n_variables=None,
                 n_inducing_points_env=None,
                 n_latents_spatial=None,
                 n_inducing_points_spatial=None,
                 unique_coordinates=None,
                 environment=True,
                 spatial=True,
                 traits=True,
                 likelihood=None):
        super().__init__()

        if likelihood == "Bernoulli":
            self.likelihood = BernoulliLikelihood
        elif likelihood == "Dirichlet":
            self.likelihood = DirichletMultinomialLikelihood

        self.environment = environment
        self.spatial = spatial
        self.traits = traits

        assert self.environment + self.spatial + self.traits, f"Model cannot run without any components! {self.environment=}, {self.spatial =}, {self.traits=}"
        print(f"Running with components: {self.environment=}, {self.spatial=}, {self.traits=}")

        if self.environment:
            self.n_latents_env = n_latents_env
            self.f = EnvironmentGP(n_latents=n_latents_env, n_variables=n_variables,
                                   n_inducing_points=n_inducing_points_env)

        if self.spatial:
            self.n_latents_spatial = n_latents_spatial
            self.g = SpatialGP(n_latents=n_latents_spatial, unique_coordinates=unique_coordinates,
                               n_inducing_points=n_inducing_points_spatial)

    def model(self, batch):
        pyro.module("model", self)

        n_samples = batch.get("n_samples_batch")
        n_species = batch.get("n_species")
        n_traits = batch.get("n_traits")

        samples_plate = pyro.plate(name="samples_plate", size=n_samples, dim=-2)
        species_plate = pyro.plate(name="species_plate", size=n_species, dim=-1)
        latent_env_plate = pyro.plate("env_latents_plate_w", self.n_latents_env, dim=-2)
        traits_plate = pyro.plate(name="traits_plate", size=n_traits, dim=-1)

        z = 0

        if self.environment:
            f_dist = self.f.pyro_model(batch.get("X"), name_prefix="f_GP")

            # Use a plate here to mark conditional independencies
            with pyro.plate("L_plate", dim=-1):
                # Sample from latent function distribution
                f_samples = pyro.sample(".f(x)", f_dist)

            f_samples = f_samples if f_samples.shape == torch.Size([n_samples, self.n_latents_env]) else f_samples.mean(
                dim=0).reshape(n_samples, self.n_latents_env)

            if self.traits:
                with traits_plate, latent_env_plate:
                    gamma = pyro.sample("gamma", dist.Normal(loc=torch.zeros(self.n_latents_env, n_traits), scale=torch.ones(self.n_latents_env, n_traits)))
                w_loc = pyro.deterministic("w_loc", (batch.get("traits") @ gamma.T).T)
            else:
                w_loc = torch.zeros(self.n_latents_env, n_species)
            
            w_scale = torch.ones(self.n_latents_env, n_species)

            with species_plate, latent_env_plate:
                w = pyro.sample("w", dist.Normal(loc=w_loc, scale=w_scale))
            z = z + f_samples @ w

            # if self.traits:
            #     print("Traits not completed")

            # w_loc = torch.zeros(self.n_latents_env, n_species)
            # w_scale = torch.ones(self.n_latents_env, n_species)
            # with species_plate, pyro.plate("env_latents_plate_w", self.n_latents_env, dim=-2):
            #     w = pyro.sample("w", dist.Normal(loc=w_loc, scale=w_scale))

            # z = z + f_samples @ w

        if self.spatial:
            g_dist = self.g.pyro_model(batch.get("coords"), name_prefix="g_GP")

            with pyro.plate("M_plate", dim=-1):
                # Sample from latent function distribution
                g_samples = pyro.sample(".g(coords)", g_dist)

            g_samples = g_samples if g_samples.shape == torch.Size(
                [n_samples, self.n_latents_spatial]) else g_samples.mean(dim=0).reshape(
                n_samples, self.n_latents_spatial)
            # g_samples = g_samples if g_samples.shape == torch.Size(
            #     [batch["n_locs_batch"], self.n_latents_spatial]) else g_samples.mean(dim=0).reshape(
            #     batch["n_locs_batch"], self.n_latents_spatial)
            # g_samples = g_samples[batch["batch_inverse"]]

            # v = pyro.param("v", torch.randn(self.n_latents_spatial, n_species))
            v_loc = torch.zeros(self.n_latents_spatial, n_species)
            v_scale = torch.ones(self.n_latents_spatial, n_species)
            with species_plate, pyro.plate("spatial_latents_plate_v", self.n_latents_spatial, dim=-2):
                v = pyro.sample("v", dist.Normal(loc=v_loc, scale=v_scale))

            z = z + g_samples @ v

        with species_plate:
            bias = pyro.sample("b", dist.Normal(loc=torch.zeros(n_species), scale=torch.ones(n_species)))

        z = z + bias

        self.likelihood(z, batch, samples_plate, species_plate)

    def guide(self, batch):
        n_species = batch.get("n_species")
        n_traits = batch.get("n_traits")

        species_plate = pyro.plate(name="species_plate", size=n_species, dim=-1)
        latent_env_plate = pyro.plate("env_latents_plate_w", self.n_latents_env, dim=-2)
        traits_plate = pyro.plate(name="traits_plate", size=n_traits, dim=-1)

        if self.environment:
            # w_loc = pyro.param(
            #     "w_loc",
            #     torch.zeros(n_species, self.n_latents_env)
            # )
            #
            # # Shape: [n_species, n_latents_env, n_latents_env]
            # w_scale_tril = pyro.param(
            #     "w_scale_tril",
            #     0.1 * torch.eye(self.n_latents_env)
            #     .expand(n_species, self.n_latents_env, self.n_latents_env)
            #     .clone(),
            #     constraint=dist.constraints.lower_cholesky
            # )
            #
            # # -- CRITICAL PART: set dim=-1 so that species is the RIGHTMOST dimension.
            # with species_plate:
            #     # By default, MultivariateNormal(...):
            #     #   - batch shape = [n_species]
            #     #   - event shape = [n_latents_env]
            #     #
            #     # Placing the plate at dim=-1 forces the "event dimension" to be -2,
            #     # so physically the sample comes out [n_latents_env, n_species].
            #     w = pyro.sample(
            #         "w",
            #         dist.MultivariateNormal(w_loc, scale_tril=w_scale_tril)
            #     )
            
            if self.traits:
                gamma_loc = pyro.param("gamma_loc", torch.zeros(self.n_latents_env, n_traits))
                gamma_scale = pyro.param("gamma_scale", 0.1 * torch.ones(self.n_latents_env, n_traits),
                                    constraint=dist.constraints.positive)
                with traits_plate, latent_env_plate:
                    #gamma = pyro.sample("gamma", dist.Normal(loc=torch.zeros(self.n_latents_env, n_traits), scale=torch.ones(self.n_latents_env, n_traits)))
                    gamma = pyro.sample("gamma", dist.Normal(loc=gamma_loc, scale=gamma_scale))
                w_loc = pyro.deterministic("w_loc", (batch.get("traits") @ gamma.T).T)
            else:
                w_loc = pyro.param("w_loc", torch.zeros(self.n_latents_env, n_species))

            #w_loc = pyro.param("w_loc", torch.zeros(self.n_latents_env, n_species))
            w_scale = pyro.param("w_scale", 0.1 * torch.ones(self.n_latents_env, n_species),
                                 constraint=dist.constraints.positive)

            with species_plate, latent_env_plate:
                w = pyro.sample("w", dist.Normal(loc=w_loc, scale=w_scale))

            # pyro.module(self.name_prefixes[i], self.gp_models[i])
            f_dist = self.f.pyro_guide(batch.get("X"), name_prefix="f_GP")
            # Use a plate here to mark conditional independencies
            with pyro.plate("L_plate", dim=-1):
                # Sample from latent function distribution
                f_samples = pyro.sample(".f(x)", f_dist)

        if self.spatial:
            g_dist = self.g.pyro_guide(batch.get("coords"), name_prefix="g_GP")  # TODO: BREAKER
            # Use a plate here to mark conditional independencies
            with pyro.plate("M_plate", dim=-1):
                # Sample from latent function distribution
                g_samples = pyro.sample(".g(coords)", g_dist)

            v_loc = pyro.param("v_loc", torch.zeros(self.n_latents_spatial, n_species))
            v_scale = pyro.param(
                "v_scale",
                0.1 * torch.ones(self.n_latents_spatial, n_species),
                constraint=dist.constraints.positive
            )

            with species_plate, pyro.plate("spatial_latents_plate_v", self.n_latents_spatial, dim=-2):
                v = pyro.sample("v", dist.Normal(loc=v_loc, scale=v_scale))

        # if self.traits:
        #     bias_loc = pyro.param("bias_loc", torch.zeros(n_species))
        #     bias_scale = pyro.param("bias_scale", torch.ones(n_species), constraint=dist.constraints.positive)
        #
        #     with species_plate:
        #         bias = pyro.sample("b", dist.Normal(loc=bias_loc, scale=bias_scale))

        bias_loc = pyro.param("bias_loc", torch.zeros(n_species))
        bias_scale = pyro.param("bias_scale", torch.ones(n_species), constraint=dist.constraints.positive)

        with species_plate:
            bias = pyro.sample("b", dist.Normal(loc=bias_loc, scale=bias_scale))

    def forward(self, batch):
        # Point prediction
        z = 0

        if self.environment:
            f_samples = self.f.pyro_guide(batch.get("X"), name_prefix="f_GP").mean
            if self.traits:
                w = (batch.get("traits") @ pyro.param("gamma_loc").T).T
            else:
                w = pyro.param("w_loc")

            z = z + f_samples @ w

        if self.spatial:
            g_samples = self.g.pyro_guide(batch.get("coords"), name_prefix="g_GP").mean
            v = pyro.param("v_loc")

            z = z + g_samples @ v

        bias = pyro.param("bias_loc")

        z = z + bias

        if isinstance(self.likelihood, type(BernoulliLikelihood)):
            return dist.Bernoulli(logits=z).mean

        if isinstance(self.likelihood, type(DirichletMultinomialLikelihood)):
            return dist.Dirichlet(concentration=z).mean


class EnvironmentGP(gpytorch.models.ApproximateGP):
    def __init__(self, n_latents, n_variables, n_inducing_points):
        self.n_latents = n_latents
        # Let's use a different set of inducing points for each latent function
        inducing_points = torch.randn(n_latents, n_inducing_points, n_variables)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([n_latents])
        )

        variational_strategy = MultitaskVariationalStrategy(  # CustomVariationalStrategy
            base_variational_strategy=gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch, so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([n_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.GammaPrior(rate=1, concentration=5),
                batch_shape=torch.Size([n_latents]),
                ard_num_dims=n_variables,
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(rate=1, concentration=2),
            batch_shape=torch.Size([n_latents])
        )

        # self.covar_module.base_kernel.lengthscale = torch.rand(n_latents, 1, n_variables)
        # self.covar_module.outputscale = torch.rand(n_latents, 1, 1)

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class HaversineRBFKernel(gpytorch.kernels.Kernel):
    """A GPyTorch kernel that computes the Haversine distance and applies an RBF transformation."""

    has_lengthscale = True  # Allows GPyTorch to learn the lengthscale

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x1, x2, diag=False, **params):
        """Compute the kernel matrix using Haversine distance with RBF transformation."""
        if diag:
            return torch.ones(1, x1.shape[-2])
        # Convert degrees to radians
        RADIUS = 6373  # Approximate radius of Earth in km

        # Convert degrees to radians
        lon1, lat1, lon2, lat2 = map(torch.deg2rad, (x1[:, :, 0], x1[:, :, 1], x2[:, :, 0], x2[:, :, 1]))

        # Compute differences
        dlon = lon2.unsqueeze(1) - lon1.unsqueeze(2)
        dlat = lat2.unsqueeze(1) - lat1.unsqueeze(2)

        # Haversine formula
        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1.unsqueeze(2)) * torch.cos(lat2.unsqueeze(1)) * torch.sin(
            dlon / 2) ** 2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

        haversine_dist = RADIUS * c

        # Apply the RBF kernel
        rbf_kernel = torch.exp(-0.5 * (haversine_dist / self.lengthscale) ** 2)

        return rbf_kernel


class SpatialGP(gpytorch.models.ApproximateGP):
    def __init__(self, n_latents, unique_coordinates, n_inducing_points):
        self.n_latents = n_latents
        num_coords = unique_coordinates.size(0)

        inducing_points = unique_coordinates[
                          torch.stack([torch.randperm(num_coords)[:n_inducing_points] for _ in range(self.n_latents)]),
                          :]

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([n_latents])
        )

        variational_strategy = MultitaskVariationalStrategy(  # CustomVariationalStrategy
            base_variational_strategy=gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=False
            ),
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch, so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([n_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.GammaPrior(rate=1, concentration=5),
                batch_shape=torch.Size([n_latents]),
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(rate=1, concentration=2),
            batch_shape=torch.Size([n_latents])
        )
        # self.covar_module.base_kernel.lengthscale = torch.rand(n_latents, 1, 1) * 5
        # self.covar_module.base_kernel.lengthscale = torch.ones(n_latents, 1, 1, requires_grad=False) * 3
        # self.covar_module.outputscale = torch.rand(n_latents, 1, 1)

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


