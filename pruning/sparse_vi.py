# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import numpy as np

from foundations import hparams
from pruning import base
from pruning.mask import Mask

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from models import base as models_base


@dataclasses.dataclass
class PruningHparams(hparams.PruningHparams):
    pruning_fraction: float = 0.2
    pruning_layers_to_ignore: str = None
    prune_by: str = "signal_to_noise"  # "signal_to_noise", "mu", "square_sum"
    
    _name = 'Hyperparameters for Sparse Global BNN Pruning'
    _description = 'Hyperparameters that modify the way pruning occurs.'
    _pruning_fraction = 'The fraction of additional weights to prune from the network.'
    _layers_to_ignore = 'A comma-separated list of addititonal tensors that should not be pruned.'
    _prune_by = 'The method to use for pruning. Options are "signal_to_noise", "mu", "square_sum".'


class Strategy(base.Strategy):
    @staticmethod
    def get_pruning_hparams() -> type:
        return PruningHparams

    @staticmethod
    def prune(pruning_hparams: PruningHparams, trained_model: 'models_base.Model', current_mask: Mask = None):
        current_mask = Mask.ones_like(trained_model).numpy() if current_mask is None else current_mask.numpy()

        # Determine the number of weights that need to be pruned.
        number_of_remaining_weights = np.sum([np.sum(v) for v in current_mask.values()])
        number_of_weights_to_prune = np.ceil(
            pruning_hparams.pruning_fraction * number_of_remaining_weights).astype(int)

        # Determine which layers can be pruned.
        prunable_tensors = set(trained_model.prunable_layer_names)
        if pruning_hparams.pruning_layers_to_ignore:
            prunable_tensors -= set(pruning_hparams.pruning_layers_to_ignore.split(','))

        # Get the model weights.
        weights = {k: v.clone().cpu().detach().numpy()
                   for k, v in trained_model.state_dict().items()
                   if k.removesuffix("_mean").removesuffix("_log_std") in prunable_tensors}
        
        # we can finally be free! this is VIMod stuff only. prune based on mu/sigma?
        # basically combine things with equal name but _mean and _log_std into a single weight
        weights_combined = {}
        for name in weights.keys():
            if name.endswith("_mean"):
                # compute scoring function
                if pruning_hparams.prune_by == "signal_to_noise":
                    weights_combined[name.removesuffix("_mean")] = weights[name] / (np.exp(weights[name.removesuffix("_mean") + "_log_std"]) ** 2 + 1e-6)  # pruned based on signal-to-noise ratio
                elif pruning_hparams.prune_by == "mu":
                    weights_combined[name.removesuffix("_mean")] = weights[name]  # prune based on only mu
                elif pruning_hparams.prune_by == "square_sum":
                    weights_combined[name.removesuffix("_mean")] = (weights[name] ** 2) + (weights[name.removesuffix("_mean") + "_log_std"] ** 2)  # prune based on square sum of sigma and mu
                else:
                    raise ValueError(f"Unknown pruning method: {pruning_hparams.prune_by}")
            elif name.endswith("_log_std"):
                pass
            else:
                weights_combined[name] = weights[name]
        weights = weights_combined
        # Create a vector of all the unpruned weights in the model.
        weight_vector = np.concatenate([v[current_mask[k] == 1] for k, v in weights.items()])
        threshold = np.sort(np.abs(weight_vector))[number_of_weights_to_prune]

        new_mask = Mask({k: np.where(np.abs(v) > threshold, current_mask[k], np.zeros_like(v))
                         for k, v in weights.items()})
        for k in current_mask:
            if k not in new_mask:
                new_mask[k] = current_mask[k]
        for k in new_mask:
            if new_mask[k].sum() == 0.:
                print(f'Mask {k} is only pruned weights.')
        return new_mask
