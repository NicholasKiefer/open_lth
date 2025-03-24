import torch
import torch.nn as nn
import torch.nn.functional as F

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_vi
from pruning import sparse_global, sparse_vi
from functools import partial
from torch_bayesian.vi import VILinear, VIConv2d, VISequential, VIResidualConnection, VIModule, KullbackLeiblerLoss
from torch_bayesian.vi.predictive_distributions import CategoricalPredictiveDistribution

class Model(base.Model):
    """A residual neural network as originally designed for CIFAR-10."""
    is_vi_model = True
    dataset_size_train = 50000  # size of cifar
    dataset_size_test = 10000  # size of cifar

    def __init__(self, initializer):
        super(Model, self).__init__()
        outputs = 10

        mods = []
        # Initial convolution.
        current_filters = 16
        mods.append(VIConv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False))
        mods.append(nn.BatchNorm2d(current_filters, track_running_stats=False))
        # mods.append(nn.BatchNorm2d(current_filters, ))
        # mods.append(nn.ReLU())

        # Final fc layer. Size = number of filters in last segment.
        mods.append(nn.AvgPool2d(8,))  # pool by what? todo:
        mods.append(nn.Flatten())
        # mods.append(VILinear(32 ** 2 * 3, 256))
        mods.append(nn.ReLU())
        mods.append(VILinear(256, outputs))

        pred_distr = CategoricalPredictiveDistribution()
        self.criterion = KullbackLeiblerLoss(pred_distr, self.dataset_size_train, heat=1., track=True)

        # bunch everything together for VI stuff
        self.vi = VISequential(*mods)

        # Initialize.
        self.apply(initializer)


    def forward(self, x):
        x, log_probs = self.vi(x)
        if self.training:
            if torch.any(torch.isnan(x)):
                raise ValueError("rip")
        return x, log_probs

    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']
    
    # return [name + '.weight' for name, module in self.named_modules()
    @property
    def prunable_layer_names(self):
        names = []
        for name, layer in self.named_modules():
            if isinstance(layer, (VILinear, VIConv2d)):
                for w in layer.random_variables:
                    names.append(f"{name}._{w}")
        return names

    @staticmethod
    def is_valid_model_name(model_name):
        return model_name == "cifar_vicnn"

    @staticmethod
    def get_model_from_name(model_name, initializer,  outputs=10):
        if not Model.is_valid_model_name(model_name):
            raise ValueError("invalid model name")

        return Model(initializer)

    @property
    def loss_criterion(self):
        if not self.training:
            return partial(self.criterion, dataset_size=self.dataset_size_test)
        return partial(self.criterion, dataset_size=self.dataset_size_train)

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='cifar_vicnn',
            model_init='vi',
            batchnorm_init='uniform',
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='cifar10',
            batch_size=128,
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='sgd',
            momentum=0.9,
            milestone_steps='80ep,120ep',
            lr=0.1,
            gamma=0.1,
            weight_decay=1e-4,
            training_steps='160ep',
        )

        pruning_hparams = sparse_vi.PruningHparams(
            pruning_strategy='sparse_vi',
            pruning_fraction=0.2
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)
