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
from torch_bayesian.vi.priors import MeanFieldNormalPrior, GaussianMixturePrior


class Model(base.Model):
    """A residual neural network as originally designed for CIFAR-10."""
    is_vi_model = True
    dataset_size_train = 50000  # size of cifar
    dataset_size_test = 10000  # size of cifar
    class Block(VIModule):
        """A ResNet block."""

        def __init__(self, f_in: int, f_out: int, downsample=False):
            super(Model.Block, self).__init__()
            init = MeanFieldNormalVarDist(initial_std=0.01)
            prior = MeanFieldNormalPrior(0, 1)
            # prior = GaussianMixturePrior(0, 0.01, 0, 1, pi=0.5)
            stride = 2 if downsample else 1
            self.conv1 = VIConv2d(f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False, variational_distribution=init, prior=prior)
            # self.bn1 = BayesianBatchNorm2d(f_out)
            self.bn1 = nn.BatchNorm2d(f_out, track_running_stats=False)
            self.conv2 = VIConv2d(f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False, variational_distribution=init, prior=prior)
            # self.bn2 = BayesianBatchNorm2d(f_out)
            self.bn2 = nn.BatchNorm2d(f_out, track_running_stats=False)

            # No parameters for shortcut connections.
            if downsample or f_in != f_out:
                self.shortcut = VISequential(
                    VIConv2d(f_in, f_out, kernel_size=1, stride=2, bias=False, variational_distribution=init, prior=prior),
                    # BayesianBatchNorm2d(f_out)
                    nn.BatchNorm2d(f_out, track_running_stats=False)
                )
            else:
                self.shortcut = VISequential()

        def forward(self, x):
            out, log_prob1 = self.conv1(x)
            out = F.relu(self.bn1(out))
            out, log_prob2 = self.conv2(out)
            out = self.bn2(out)
            shortcut, log_prob3 = self.shortcut(x)
            out = out + shortcut
            log_prob = log_prob1 + log_prob2 + log_prob3
            return F.relu(out), log_prob

    def __init__(self, plan, initializer, outputs=None, kl_loss_scale=1.0):
        super(Model, self).__init__()
        outputs = outputs or 10

        prior = MeanFieldNormalPrior(0, 1)
        # prior = GaussianMixturePrior(0, 0.01, 0, 1, pi=0.5)
        init = MeanFieldNormalVarDist(initial_std=0.01)
        mods = []
        # Initial convolution.
        current_filters = plan[0][0]
        mods.append(VIConv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False, variational_distribution=init, prior=prior))
        mods.append(nn.BatchNorm2d(current_filters, track_running_stats=False))
        mods.append(nn.ReLU())

        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(Model.Block(current_filters, filters, downsample))
                current_filters = filters

        mods.extend(blocks)

        # Final fc layer. Size = number of filters in last segment.
        mods.append(nn.AvgPool2d(8,))  # pool by what? todo:
        mods.append(nn.Flatten())
        mods.append(VILinear(plan[-1][0], outputs))


        # kl_loss = 1 / len(loader) * kl_div(model) * 4e-3
        self.kl_loss_scale = kl_loss_scale
        pred_distr = CategoricalPredictiveDistribution()
        self.criterion = KullbackLeiblerLoss(pred_distr, self.dataset_size_train, heat=.1, track=True)

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
        valid = model_name.startswith("cifar_viresnet_")
        if not valid: return False;
        valid = (5 > len(model_name.split('_')) > 2 
                and all([x.isdigit() 
                and int(x) > 0 for x in model_name.split('_')[2:]]) 
                and (int(model_name.split('_')[2]) - 2) % 6 == 0 
                and int(model_name.split('_')[2]) > 2)
        return valid

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
        name = model_name.split('_')
        W = 16 if len(name) == 3 else int(name[3])
        D = int(name[2])
        if (D - 2) % 3 != 0:
            raise ValueError('Invalid ResNet depth: {}'.format(D))
        D = (D - 2) // 6
        plan = [(W, D), (2*W, D), (4*W, D)]

        return Model(plan, initializer, outputs, 1.)

    @property
    def loss_criterion(self):
        if not self.training:
            return partial(self.criterion, dataset_size=self.dataset_size_test)
        return partial(self.criterion, dataset_size=self.dataset_size_train)

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='cifar_viresnet_20',
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
            pruning_fraction=0.2,
            prune_by='signal_to_noise',
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)
