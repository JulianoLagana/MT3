import argparse

import matplotlib.pyplot as plt
import torch
import numpy as np

from src.data_generation.data_generator import DataGenerator
from src.util.load_config_files import load_yaml_into_dotdict, dotdict
from src.util.misc import factor_int

batch_size = 4

# Load hyperparameters from yaml file specified via CLI
parser = argparse.ArgumentParser()
parser.add_argument('-tp', '--task_params', help='filepath to configuration yaml file defining the task', required=True)
args = parser.parse_args()


params = load_yaml_into_dotdict(args.task_params)
params.training = dotdict()
params.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
params.training.batch_size = batch_size

datagen = DataGenerator(params)

for j in range(10):
    nrows, ncols = factor_int(batch_size)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 10))
    training_nested_tensor, labels, unique_ids, trajectories = datagen.get_batch()
    for i in range(batch_size):
        idx_mask_starts = np.argmax(training_nested_tensor.mask[i])
        max_idx = idx_mask_starts if idx_mask_starts != 0 else -1
        meas = training_nested_tensor.tensors[i]

        if isinstance(ax, np.ndarray):
            ax_ = ax.flatten()[i]
        else:
            ax_ = ax
        ax_.scatter(meas[:max_idx, 0], meas[:max_idx, 1], marker='x', c=unique_ids[i][:max_idx])
        ax_.scatter(labels[i][:, 0], labels[i][:, 1], color='red', s=100, alpha=0.5)

        ax_.set_xlim([params.data_generation.field_of_view_lb-1, params.data_generation.field_of_view_ub+1])
        ax_.set_ylim([params.data_generation.field_of_view_lb-1, params.data_generation.field_of_view_ub+1])
        ax_.grid()
    plt.show()





