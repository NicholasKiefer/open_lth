import torch
import torch.nn as nn
import torch.nn.functional as F

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from functools import partial
from pruning import sparse_global, sparse_vi
from torch_bayesian.vi import VILinear, VIConv2d, VISequential, VIResidualConnection, VIModule, KullbackLeiblerLoss
from torch_bayesian.vi.predictive_distributions import CategoricalPredictiveDistribution
from torch_bayesian.vi.variational_distributions import MeanFieldNormalVarDist
from torch_bayesian.vi.priors import MeanFieldNormalPrior


class Model(base.Model):
    """A VGG-style neural network designed for CIFAR-10."""
    is_vi_model = True
    dataset_size_train = 50000  # size of cifar
    dataset_size_test = 10000  # size of cifar

    class ConvModule(VIModule):
        """A single convolutional module in a VGG network."""

        def __init__(self, in_filters, out_filters):
            super(Model.ConvModule, self).__init__()
            init = MeanFieldNormalVarDist(initial_std=0.01)
            prior = MeanFieldNormalPrior(0, 1)
            self.conv = VIConv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1, variational_distribution=init, prior=prior)
            self.bn = nn.BatchNorm2d(out_filters, track_running_stats=False)

        def forward(self, x):
            x, log_prob = self.conv(x)
            x = F.relu(self.bn(x))
            return x, log_prob

    def __init__(self, plan, initializer, outputs=10):
        super(Model, self).__init__()

        layers = []
        filters = 3

        for spec in plan:
            if spec == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(Model.ConvModule(filters, spec))
                filters = spec

        layers.append(nn.AvgPool2d(2))
        layers.append(nn.Flatten())
        layers.append(VILinear(512, outputs))
        self.layers = VISequential(*layers)
        
        pred_distr = CategoricalPredictiveDistribution()
        self.criterion = KullbackLeiblerLoss(pred_distr, self.dataset_size_train, heat=.1, track=True)

        self.apply(initializer)

    def forward(self, x):
        x, log_probs = self.layers(x)
        if self.training:
            if torch.any(torch.isnan(x)):
                raise ValueError("rip")
        return x, log_probs
    
    def return_log_probs(self):
        self.layers.return_log_probs()

    @property
    def output_layer_names(self):
        name = list(reversed(list(self.layers._modules.items())))[0]
        return [f'layers.{name}._weight', f'layers.{name}._bias']
    
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
        return (model_name.startswith('cifar_vivgg_') and
                len(model_name.split('_')) == 3 and
                model_name.split('_')[2].isdigit() and
                int(model_name.split('_')[2]) in [11, 13, 16, 19])

    @staticmethod
    def get_model_from_name(model_name, initializer, outputs=10):
        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        outputs = outputs or 10

        num = int(model_name.split('_')[2])
        if num == 11:
            plan = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
        elif num == 13:
            plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
        elif num == 16:
            plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        elif num == 19:
            plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
        else:
            raise ValueError('Unknown VGG model: {}'.format(model_name))

        return Model(plan, initializer, outputs)

    @property
    def loss_criterion(self):
        if not self.training:
            return partial(self.criterion, dataset_size=self.dataset_size_test)
        return partial(self.criterion, dataset_size=self.dataset_size_train)

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='cifar_vivgg_16',
            model_init='vi',
            batchnorm_init='uniform',
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='cifar10',
            batch_size=128
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='adam',
            milestone_steps='80ep,120ep',
            lr=1e-3,
            gamma=0.1,
            weight_decay=1e-4,
            training_steps='160ep'
        )

        pruning_hparams = sparse_vi.PruningHparams(
            pruning_strategy='sparse_vi',
            pruning_fraction=0.2,
            prune_by="signal_to_noise",
            pruning_layers_to_ignore='fc._weight,fc._bias',
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)
