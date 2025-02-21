# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torch.distributed
import pathlib

from platforms import base


class Platform(base.Platform):
    @property
    def root(self):
        return pathlib.Path("/hkfs/work/workspace_haic/scratch/vq6575-gen_param/open_lth/data/")

    @property
    def dataset_root(self):
        return pathlib.Path("/hkfs/work/workspace_haic/scratch/vq6575-gen_param/open_lth/data/")

    @property
    def imagenet_root(self):
        raise NotImplementedError


class haicore(Platform):
    @property
    def device_str(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    def torch_device(self):
        return torch.device(self.device_str)

    @property
    def is_parallel(self):
        return torch.cuda.is_available() and torch.cuda.device_count() > 1

    @property
    def is_distributed(self):
        return False

    @property
    def rank(self):
        return torch.distributed.get_rank()

    @property
    def world_size(self):
        return torch.cuda.device_count()

    @property
    def is_primary_process(self):
        return not self.is_distributed or self.rank == 0

    def barrier(self):
        pass