from typing import Optional, Tuple

import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader
from torch_bayesian.vi import VIModule
from torch_bayesian.vi.predictive_distributions import CategoricalPredictiveDistribution as Categorical


def classification_calibration(
    loader: DataLoader,
    model: VIModule,
    bins: int = 20,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Tensor:
    """Calculate the calibration of a classification model."""
    total = torch.zeros(bins)
    positive = torch.zeros(bins)
    dist = Categorical()
    # counter = 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device=device, dtype=dtype))[0].permute(1, 0, 2)
            out = dist.predictive_parameters_from_samples(out)
            total += torch.histogram(out.to(device="cpu"), bins=bins, range=[0, 1]).hist
            positive += torch.histogram(
                torch.gather(out.to(device="cpu"), 1, y.unsqueeze(-1)),
                bins=bins,
                range=[0, 1],
            ).hist
            # if (counter := counter + 1) == 50:
            #    break

    calibration = positive / total
    return calibration


def mean_absolute_calibration_error(calibration: Tensor) -> float:
    """Calculate the mean absolute calibration error from the calibration line."""
    bins = len(calibration)
    ref = torch.linspace(0.0, 1.0, bins + 1)[:-1] + 1 / (2 * bins)
    return ((ref - calibration).abs() / bins).sum().item()


def root_mean_square_calibration_error(calibration: Tensor) -> float:
    """Calculate the root mean squared calibration error from the calibration line."""
    bins = len(calibration)
    ref = torch.linspace(0.0, 1.0, bins + 1)[:-1] + 1 / (2 * bins)
    return ((ref - calibration).pow(2) / bins).sum().sqrt().item()


def plot_calibration(
    calibration: Tensor, std: Optional[Tensor] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the calibration line."""
    bins = len(calibration)
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], c=(0.7, 0.7, 0.7))
    x_values = torch.linspace(0.0, 1.0, bins + 1)[:-1] + 1 / (2 * bins)
    ax.plot(x_values, calibration)

    if std is not None:
        ax.fill_between(x_values, calibration - std, calibration + std, alpha=0.3)

    return fig, ax