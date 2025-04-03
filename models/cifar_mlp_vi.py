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
from torch_bayesian.vi.variational_distributions import MeanFieldNormalVarDist
from torch_bayesian.vi.priors import MeanFieldNormalPrior


class Model(base.Model):
    """A residual neural network as originally designed for CIFAR-10."""
    is_vi_model = True
    dataset_size_train = 50000  # size of cifar
    dataset_size_test = 10000  # size of cifar

    def __init__(self, initializer, outputs=None, kl_loss_scale=1.0):
        super(Model, self).__init__()
        outputs = outputs or 10

        prior = MeanFieldNormalPrior(0, 1)
        init = MeanFieldNormalVarDist(initial_std=0.05)
        mods = []
        mods.append(nn.Flatten())
        mods.append(VILinear(3072, 256, init, prior, ))
        mods.append(nn.BatchNorm1d(256, track_running_stats=False))
        mods.append(nn.ReLU())
        mods.append(VILinear(256, 256, init, prior, ))
        mods.append(nn.BatchNorm1d(256, track_running_stats=False))
        mods.append(nn.ReLU())
        mods.append(VILinear(256, outputs, init, prior))


        # kl_loss = 1 / len(loader) * kl_div(model) * 4e-3
        self.kl_loss_scale = kl_loss_scale
        pred_distr = CategoricalPredictiveDistribution()
        self.criterion = KullbackLeiblerLoss(pred_distr, self.dataset_size_train, heat=1, track=True)

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
        return model_name == "cifar_mlp_vi"

    @staticmethod
    def get_model_from_name(model_name, initializer,  outputs=10):
        """The naming scheme for a ResNet is 'cifar_resnet_N[_W]'.

        The ResNet is structured as an initial convolutional layer followed by three "segments"
        and a linear output layer. Each segment consists of D blocks. Each block is two
        convolutional layers surrounded by a residual connection. Each layer in the first segment
        has W filters, each layer in the second segment has 32W filters, and each layer in the
        third segment has 64W filters.

        The name of a ResNet is 'cifar_resnet_N[_W]', where W is as described above.
        N is the total number of layers in the network: 2 + 6D.
        The default value of W is 16 if it isn't provided.

        For example, ResNet-20 has 20 layers. Exclusing the first convolutional layer and the final
        linear layer, there are 18 convolutional layers in the blocks. That means there are nine
        blocks, meaning there are three blocks per segment. Hence, D = 3.
        The name of the network would be 'cifar_resnet_20' or 'cifar_resnet_20_16'.
        """

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        return Model(initializer, outputs, 1.)

    @property
    def loss_criterion(self):
        if not self.training:
            return partial(self.criterion, dataset_size=self.dataset_size_test)
        return partial(self.criterion, dataset_size=self.dataset_size_train)

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='cifar_mlp_vi',
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
