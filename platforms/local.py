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
        return pathlib.Path("/hkfs/work/workspace_haic/scratch/vq6575-openlth/open_lth/data/")

    @property
    def dataset_root(self):
        return pathlib.Path("/hkfs/work/workspace_haic/scratch/vq6575-openlth/open_lth/data/")

    @property
    def imagenet_root(self):
        return pathlib.Path("/hkfs/home/dataset/datasets/imagenet-2012/original/imagenet-raw/ILSVRC/Data/CLS-LOC/")


class Haicore(Platform):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_workers = 4
        self._initialize_distributed()

    def _initialize_distributed(self):
        """Initializes the distributed environment if applicable."""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            torch.distributed.init_process_group(backend="nccl")
            self._distributed = True
            torch.cuda.set_device(self.rank)
            print(f"initialized on rank {self.rank}")
        else:
            self._distributed = False
            print("did not initialize distributed, RANK is not in os.environ")

    @property
    def device_str(self):
        return f'cuda:{self.rank}' if self.is_distributed else 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    def torch_device(self):
        return torch.device(self.device_str)

    @property
    def is_parallel(self):
        return self.is_distributed  # DDP inherently means parallel execution

    @property
    def is_distributed(self):
        return self._distributed

    @property
    def rank(self):
        return torch.distributed.get_rank() if self.is_distributed else 0

    @property
    def world_size(self):
        return torch.distributed.get_world_size() if self.is_distributed else 1

    @property
    def is_primary_process(self):
        return self.rank == 0  # Rank 0 is the primary process

    def barrier(self):
        if self.is_distributed:
            torch.distributed.barrier()