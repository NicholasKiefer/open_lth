# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F
import timm

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global, sparse_vi

from functools import partial
import torch
from torch_bayesian.vi import VILinear, VIConv2d, VISequential, VIResidualConnection, VIModule, KullbackLeiblerLoss
from torch_bayesian.vi.predictive_distributions import CategoricalPredictiveDistribution
from torch_bayesian.vi.variational_distributions import MeanFieldNormalVarDist
from torch_bayesian.vi.priors import MeanFieldNormalPrior

from einops import rearrange
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = VISequential(
            nn.LayerNorm(dim),
            VILinear(dim, hidden_dim, variational_distribution=MeanFieldNormalVarDist(initial_std=0.01)),
            nn.GELU(),
            VILinear(hidden_dim, dim, variational_distribution=MeanFieldNormalVarDist(initial_std=0.01)),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = VILinear(dim, inner_dim * 3, bias = False, variational_distribution=MeanFieldNormalVarDist(initial_std=0.01))
        self.to_out = VILinear(inner_dim, dim, bias = False, variational_distribution=MeanFieldNormalVarDist(initial_std=0.01))

    def forward(self, x):
        x = self.norm(x)

        qkv, log1 = self.to_qkv(x)
        qkv = qkv.chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out, log2 = self.to_out(out)
        log_ = log1 + log2
        return out, log_

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        layers = []
        for _ in range(depth):
            layers.extend([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ])
        self.layers = VISequential(*layers)
    def forward(self, x):
        total_log_probs = torch.tensor([0.0, 0.0], device=x.device)
        for mod in self.layers:
            out, log = mod(x)
            x = x + out
            total_log_probs = total_log_probs + log
        out = self.norm(x)
        return out, total_log_probs

class SimpleViT(VIModule):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = VISequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            VILinear(patch_dim, dim, variational_distribution=MeanFieldNormalVarDist(initial_std=0.01)),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        # self.to_latent = nn.Identity()

        self.linear_head = VILinear(dim, num_classes, variational_distribution=MeanFieldNormalVarDist(initial_std=0.01))

    def forward(self, img):
        device = img.device
        total_log_probs = torch.tensor([0.0, 0.0], device=device)
        x, log = self.to_patch_embedding(img)
        total_log_probs = total_log_probs + log
        x = x + self.pos_embedding.to(device, dtype=x.dtype)

        x, log = self.transformer(x)
        total_log_probs = total_log_probs + log
        x = x.mean(dim = 1)

        # x = self.to_latent(x)
        out, log = self.linear_head(x)
        total_log_probs = total_log_probs + log
        return out, total_log_probs


class Model(base.Model):
    """A Simple ViT."""
    is_vi_model = True
    dataset_size_train = 50000  # size of cifar
    dataset_size_test = 10000  # size of cifar
    
    def __init__(self, plan, initializer, outputs=None):
        super(Model, self).__init__()
        outputs = outputs or 10

        embed_dim = {"tiny": 192, "small": 384, "base": 768}[plan]
        # model_args = dict(patch_size=2, embed_dim=embed_dim, depth=12, num_heads=3, img_size=32, num_classes=outputs, global_pool="avg")
        self.vit = SimpleViT(dim=embed_dim, image_size=32, patch_size=4, num_classes=outputs, depth=6, heads=3, dim_head=embed_dim // 3, mlp_dim=4 * embed_dim)
        pred_distr = CategoricalPredictiveDistribution()
        self.criterion = KullbackLeiblerLoss(pred_distr, self.dataset_size_train, heat=.1, track=True)

        # Initialize.
        self.apply(initializer)

    def forward(self, x):
        out = self.vit(x)
        return out

    def return_log_probs(self):
        self.vit.return_log_probs()

    @property
    def prunable_layer_names(self):
        names = []
        for name, layer in self.named_modules():
            if isinstance(layer, (VILinear, VIConv2d)):
                for w in layer.random_variables:
                    names.append(f"{name}._{w}")
        return names

    @property
    def output_layer_names(self):
        return ["vit.linear_head.weight", "vit.linear_head.bias"]

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('cifar_visvit_') and model_name.split("_")[2] in ["tiny", "small", "base"])

    @staticmethod
    def get_model_from_name(model_name, initializer,  outputs=10):
        """
        Naming scheme is cifar_svit_size, where size is in [tiny, small, base]
        """

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        name = model_name.split('_')
        plan = name[2]

        return Model(plan, initializer, outputs)

    @property
    def loss_criterion(self):
        if not self.training:
            return partial(self.criterion, dataset_size=self.dataset_size_test)
        return partial(self.criterion, dataset_size=self.dataset_size_train)

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='cifar_visvit_tiny',
            model_init='kaiming_normal',
            batchnorm_init='uniform',
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='cifar10',
            batch_size=1024,
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='adam',
            milestone_steps='cosine',
            lr=0.001,
            weight_decay=1e-4,
            training_steps='160ep',
            warmup_steps="80ep",
        )

        pruning_hparams = sparse_vi.PruningHparams(
            pruning_strategy='sparse_vi',
            pruning_fraction=0.2
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)
