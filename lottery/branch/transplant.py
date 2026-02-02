import torch
from lottery.branch import base
import models.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train
from datasets import registry as datasets_registry
from copy import deepcopy
import numpy as np


class Branch(base.Branch):
    def branch_function(self):
        non_vi_weights = models.registry.load(self.lottery_desc.run_path(self.replicate, 0), self.lottery_desc.str_to_step('0ep'), self.lottery_desc.model_hparams, map_location=torch.device("cpu"))
        non_vi_mask = Mask.load(self.level_root)
        
        # get an equal bayesian model, init, weight doesnt matter
        b_params = deepcopy(self.lottery_desc.model_hparams)
        # go from cifar_resnet_x to cifar_viresnet_x
        # go from cifar_vgg_x to cifar_vivgg_x
        vi_name = b_params.model_name.split("_")
        b_params.model_name = f"{vi_name[0]}_vi{vi_name[1]}_{vi_name[2]}"
        b_model = models.registry.get(b_params, datasets_registry.num_classes(self.lottery_desc.dataset_hparams))
        # instantiate bayesian model mask
        vi_mask = Mask.ones_like(b_model)
        # transfer weights to bayesian model, create a new state_dict to match names in b_model, and load state_dict
        # then overwrite mask per layer with nb mask, set sigma equal to mu mask
        # resnet non vi model has additional one vector for output linear layer
        # vgg vi models have additional masking of bias
        # svit vi models have additional masking of bias
        old_mask = []
        for (key, value) in non_vi_mask.items():
            if "bn" in key:continue
            if "running" in key or "num" in key: continue
            if key == "fc.weight" and "resnet" in b_params.model_name: continue
            old_mask.append(value)

        new_mask = {}
        idx = 0
        for key in vi_mask.keys():
            if key.endswith("_log_std"): continue
            if "bn" in key: continue
            if "_bias" in key: continue
            new_mask[key] = old_mask[idx]
            idx += 1

        old_weights = []
        for key, val in non_vi_weights.state_dict().items():
            if "running" in key: continue
            if "num" in key: continue
            old_weights.append(val)
        
        new_state = {}
        idx = 0
        for key in b_model.state_dict().keys():
            if key.endswith("_log_std"):continue
            new_state[key] = old_weights[idx]
            if key.endswith("_mean"):
                new_state[key.replace("_mean", "_log_std")] = torch.ones_like(old_weights[idx]) * np.log(0.01)
            idx += 1
        
        b_model.load_state_dict(new_state)
        b_model = PrunedModel(b_model, new_mask)
        # train
        train.standard_train(b_model, self.branch_root, self.lottery_desc.dataset_hparams,
                             self.lottery_desc.training_hparams, start_step=None, verbose=self.verbose)

    @staticmethod
    def description():
        return "Finetune a non-bayesian model bayesian."

    @staticmethod
    def name():
        return 'transplant'
