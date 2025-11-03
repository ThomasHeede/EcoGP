import torch
import pyro
import pyro.distributions as dist

from models.DirichletMultinomial import DirichletMultinomial


def DirichletMultinomialLikelihood(z, batch, samples_plate, species_plate):
    z = torch.nn.functional.softplus(z) + 1e-6

    pyro.deterministic("z", z)

    if batch.get("training", True):
        with samples_plate:
            # IMPORTANT: no species_plate here; species is the Dirichlet EVENT dim
            pyro.sample("y", DirichletMultinomial(concentration=z, total_count=batch.get("Y").sum(dim=1), is_sparse=True),
                        obs=batch.get("Y"))
    else:
        # No plate for predictive !!!
        pyro.sample("y", dist.Dirichlet(concentration=z), obs=None)


def BernoulliLikelihood(z, batch, samples_plate, species_plate):
    pyro.deterministic("z", z)

    with samples_plate, species_plate:
        pyro.sample("y", dist.Bernoulli(logits=z),
                    obs=batch.get("Y").bool().float() if batch.get("training", True) else None)