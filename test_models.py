import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
import torch_bayesian
import sys
import os
import argparse
import re
sys.path.append(os.path.abspath(".."))

import platforms.registry
import datasets
import models.registry
import foundations.step
from foundations import hparams as hparams_lib
from foundations import paths
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
import torchmetrics
import numpy as np


def get_hashname(exp_line: int):
    with open(f"logs/gen/slurm_output_{exp_line:03d}.out", "r") as file:
        lines = file.readlines()
        outputline = [i for i in lines if i.startswith("Output Location: ")]
        if outputline:
            return outputline[0].removeprefix("Output Location: ").strip("\n").replace("level_0", "level_x").replace("vq6575-openlth", "vq6575-bnn_lth")
    raise ValueError(f"Could not find Output Location in slurm_output_{exp_line:03d}.out")


def get_model_name(exp_line: int):
    """Extract model name from slurm output file."""
    with open(f"logs/gen/slurm_output_{exp_line:03d}.out", "r") as file:
        lines = file.readlines()
        for line in lines:
            if "model_name =>" in line:
                # Extract model name from line like "    * model_name => cifar_viresnet_20"
                match = re.search(r'model_name => (\S+)', line)
                if match:
                    return match.group(1)
    raise ValueError(f"Could not find model_name in slurm_output_{exp_line:03d}.out")


def compute_test_accuracy(exp_line: int, level: int = 20):
    """
    Compute calibration and MACE for a given experiment and pruning level.
    Returns tuple of (calibration tensor, MACE float).
    """
    base_dir = get_hashname(exp_line)
    dir_path = base_dir.replace("level_x", f"level_{level}")
    
    # Get model name from slurm output
    modelname = get_model_name(exp_line)
    
    # Recreate the dataset and model hyperparameters for this run.
    dataset_hparams = hparams_lib.DatasetHparams(
        dataset_name="cifar10",
        batch_size=128,
    )

    model_hparams = hparams_lib.ModelHparams(
        model_name=modelname,
        model_init="vi",
        batchnorm_init="uniform",
    )

    # Set up platform
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args([])
    args.num_workers = 0
    platform = platforms.registry.get("haicore").create_from_args(args)
    platforms.platform._PLATFORM = platform

    # Build the underlying VI model and wrap it with the pruning mask at this level.
    # Load the trained weights from the final checkpoint (160 epochs for this run).
    base_model = models.registry.get(model_hparams)
    
    # Use GPU for loading and computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_path = os.path.join(dir_path, "model_ep160_it0.pth")
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    base_model = base_model.to(device)
    base_model.load_state_dict(state_dict)
    
    mask = Mask.load(dir_path)
    for k, v in mask.items():
        mask[k] = v.to(device)
    model = PrunedModel(base_model, mask)

    # Build the test data loader.
    loader = datasets.registry.get(dataset_hparams, train=False)

    if "resnet" in modelname:
        model.model.vi.return_log_probs(mode=True)
    elif "vgg" in modelname:
        model.model.layers.return_log_probs(mode=True)
    elif "vit" in modelname:
        model.model.vit.return_log_probs(mode=True)
    model.eval()
    
    # compute test accuracy (using GPU)
    accs = []
    for _ in range(5):
        accs.append(test_model(loader, model, device))
    return accs

def test_model(loader, model, device):
    metric_acc = torchmetrics.Accuracy("multiclass", num_classes=10,).to(device)
    with torch.no_grad():
        for (x, y) in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if isinstance(model, (DataParallel, DistributedDataParallel)):
                if hasattr(model.module.model, "is_vi_model"):
                    out = model.module.model.criterion.predictive_distribution.predictive_parameters_from_samples(out[0].permute(1, 0, 2))
            elif hasattr(model.model, "is_vi_model"):
                    out = model.model.criterion.predictive_distribution.predictive_parameters_from_samples(out[0].permute(1, 0, 2))
            metric_acc.update(out, y)
    return metric_acc.compute()


def main():
    parser = argparse.ArgumentParser(description="Measure test acc for a given experiment")
    parser.add_argument("exp_line", type=int, help="Experiment line number (1-61)")
    parser.add_argument("--level", type=int, default=20, help="Pruning level to evaluate")
    args = parser.parse_args()
    
    try:
        print(f"Processing experiment {args.exp_line}, level {args.level}")
        test_acc = compute_test_accuracy(args.exp_line, args.level)
        test_acc = [i.cpu() for i in test_acc]
        # Get the directory path where we should save results
        base_dir = get_hashname(args.exp_line)
        dir_path = base_dir.replace("level_x", f"level_{args.level}")
        
        # Save calibration tensor and MACE
        test_acc_path = os.path.join(dir_path, "accs")
        
        # Move calibration to CPU before saving
        test_acc = np.array(test_acc)
        np.save(test_acc_path, test_acc)
        # with open(test_acc_path, "a") as f:
            # f.write(f"{test_acc.cpu().item()}\n")
        
        print(f"test acc saved to: {test_acc_path}")
        print(f"acc: {test_acc}")
        
    except Exception as e:
        print(f"Error processing experiment {args.exp_line}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
