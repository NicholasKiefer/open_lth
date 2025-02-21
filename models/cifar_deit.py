# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.deit import _create_deit

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global


class Model(base.Model):
    """A DeiT as originally designed for Imagenet."""

    def __init__(self, plan, initializer, outputs=None):
        super(Model, self).__init__()
        outputs = outputs or 10

        embed_dim = {"tiny": 192, "small": 384, "base": 768}[plan]
        model_args = dict(patch_size=16, embed_dim=embed_dim, depth=12, num_heads=3, img_size=32, num_classes=outputs)
        self.deit = _create_deit(
        f'deit_{plan}_patch16_224', pretrained=False, distilled=False, **dict(model_args,))
        self.criterion = nn.CrossEntropyLoss()

        # Initialize.
        self.apply(initializer)

    def forward(self, x):
        out = self.deit(x)
        return out

    @property
    def output_layer_names(self):
        return ["head.weight", "head.bias"]

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('cifar_deit_') and model_name.split("_")[2] in ["tiny", "small", "base"])

    @staticmethod
    def get_model_from_name(model_name, initializer,  outputs=10):
        """
        Naming scheme is cifar_resnet_size, where size is in [tiny, small, base]
        """

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        name = model_name.split('_')
        plan = name[2]

        return Model(plan, initializer, outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='cifar_deit_tiny',
            model_init='kaiming_normal',
            batchnorm_init='uniform',
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='cifar10',
            batch_size=1024,
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='adamw',
            momentum=0.9,
            milestone_steps='80ep,120ep',
            lr=0.001,
            gamma=0.1,
            weight_decay=5e-2,
            training_steps='160ep',
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)
