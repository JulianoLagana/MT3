import sys
import argparse

from scipy.io import savemat
import torch
import numpy as np

from src.data_generation.data_generator import DataGenerator
from src.util.load_config_files import load_yaml_into_dotdict, dotdict


parser = argparse.ArgumentParser()
parser.add_argument('-fp', '--filepath', help='filepath to configuration yaml file', required=True)
args = parser.parse_args()

seed = 1
n_samples = 1000
print_n_avg_objects = True

# Create data generator
params = load_yaml_into_dotdict(args.filepath)
params.training = dotdict()
params.training.batch_size = 1
params.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
params.data_generation.seed = seed
data_gen = DataGenerator(params)

measurements = []
gt_datas = []
lens = []
for i_sample in range(n_samples):
    training_tensor, labels, unique_ids, trajectories = data_gen.get_batch()

    if print_n_avg_objects:
        lens.append(len(labels[0]))

    measurements.append(training_tensor.tensors[0].numpy())
    gt_datas.append(labels[0].numpy())

    if i_sample % 100 == 0:
        print(f"{i_sample}/{n_samples} samples computed")

if print_n_avg_objects:
    print(f"Average number of objects in {n_samples} samples: {np.mean(lens)}")

savedict = \
   {
       'measurements': measurements,
       'ground_truths': gt_datas,
       'hyperparam_n_timesteps': params.data_generation.n_timesteps,
       'p_add': params.data_generation.p_add,
       'p_remove': params.data_generation.p_remove,
       'p_meas': params.data_generation.p_meas,
       'sigma_q': params.data_generation.sigma_q,
       'sigma_y': params.data_generation.sigma_y,
       'n_avg_false_measurments': params.data_generation.n_avg_false_measurements,
       'n_avg_starting_objects': params.data_generation.n_avg_starting_objects,
       'field_of_view_lb': params.data_generation.field_of_view_lb,
       'field_of_view_ub': params.data_generation.field_of_view_ub,
       'mu_x0': params.data_generation.mu_x0,
       'std_x0': params.data_generation.std_x0,
       'mu_v0': params.data_generation.mu_v0,
       'std_v0': params.data_generation.std_v0,
   }

task_name = sys.argv[-1].split('/')[-1].split('.')[0]
savemat(f'{task_name}-{n_samples}samples-seed{seed}.mat', savedict)
