# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from models.base import Model
from pruning.mask import Mask

import numpy as np


class PrunedModel(Model):
    @staticmethod
    def to_mask_name(name):
        return 'mask_' + name.replace('.', '___')

    def __init__(self, model: Model, mask: Mask):
        if isinstance(model, PrunedModel): raise ValueError('Cannot nest pruned models.')
        super(PrunedModel, self).__init__()
        self.model = model

        for k in self.model.prunable_layer_names:
            if k not in mask:
                raise ValueError('Missing mask value {}.'.format(k))

            kname = k
            if hasattr(model, "is_vi_model"):
                kname = k + "_mean"
            weight_k = self.model.state_dict()[kname]
            if not np.array_equal(mask[k].shape, np.array(weight_k.shape)):
                raise ValueError('Incorrect mask shape {} for tensor {}.'.format(mask[k].shape, k))

        for k in mask:
            if k not in self.model.prunable_layer_names:
                raise ValueError('Key {} found in mask but is not a valid model tensor.'.format(k))

        for k, v in mask.items():
            gen_name = PrunedModel.to_mask_name(k)
            self.register_buffer(gen_name, v.float())
        self._apply_mask()

    # todo: Vi models behave differently here
    def _apply_mask(self):
        for name, param in self.model.named_parameters():
            gen_name = PrunedModel.to_mask_name(name).removesuffix("_mean").removesuffix("_log_std")
            if hasattr(self, gen_name):
                param.data *= getattr(self, gen_name)

    def forward(self, x):
        self._apply_mask()
        return self.model.forward(x)

    @property
    def prunable_layer_names(self):
        return self.model.prunable_layer_names

    @property
    def output_layer_names(self):
        return self.model.output_layer_names

    @property
    def loss_criterion(self):
        return self.model.loss_criterion

    def save(self, save_location, save_step):
        self.model.save(save_location, save_step)

    @staticmethod
    def default_hparams(): raise NotImplementedError()
    @staticmethod
    def is_valid_model_name(model_name): raise NotImplementedError()
    @staticmethod
    def get_model_from_name(model_name, outputs, initializer): raise NotImplementedError()
