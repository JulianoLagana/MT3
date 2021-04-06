from util.misc import super_load
from util.load_config_files import load_yaml_into_dotdict
import argparse
import warnings


# Parse arguments and load the model, before doing anything else (important, reduces possibility of weird bugs)
parser = argparse.ArgumentParser()
parser.add_argument('-rp', '--result_filepath', help='filepath to result folder for trained model', required=True)
parser.add_argument('-tp', '--task_params', help='filepath to configuration yaml file defining the task', required=False)
args = parser.parse_args()
print(f'Evaluating results from folder: {args.result_filepath}...')

model, params = super_load(args.result_filepath, verbose=True)

# Test that the model was trained in the task chosen for evaluation
if args.task_params is not None:
    task_params = load_yaml_into_dotdict(args.task_params)
    for k, v in task_params.data_generation.items():
        if k not in params.data_generation:
            warnings.warn(f"Key '{k}' not found in trained model's hyperparameters")
        elif params.data_generation[k] != v:
            warnings.warn(f"Different values for key '{k}'. Task: {v}\tTrained: {params.data_generation[k]}")
    # Use task params, not the ones from the trained model
    params.recursive_update(task_params)  # note: parameters specified only on trained model will remain untouched
else:
    warnings.warn('Evaluation task was not specified; inferring it from the task specified in the results folder.')



import os
import pickle

import numpy as np
from data_generation.data_generator import DataGenerator

from modules.loss import MotLoss
from modules import evaluator


# Read evaluation hyperparameters and overwrite `params` with them
eval_params = load_yaml_into_dotdict('configs/eval/default.yaml')
params.recursive_update(eval_params)

data_generator = DataGenerator(params)
mot_loss = MotLoss(params)

og, pg, d = evaluator.evaluate_metrics(data_generator, model, params, mot_loss,  num_eval=1000, verbose=False)

print("\nFinished running evaluation... please paste this in the spread-sheet")
# Print GOSPA scores
print('GOSPA metric:')
for method, values in og.items():
    print(f"\t{method}: ")
    for value_name, value in values.items():
        print(f"\t\t{value_name:<13}: {np.mean(value):<6.4} ({np.var(value):<5.4})")

# Print other metrics
for metric, metric_name in zip([pg, d], ['Prob-GOSPA', 'DETR']):
    print(f"{metric_name} metric:")
    for method, values in metric.items():
        print(f"\t{method:<11}: {np.mean(values):<6.4} ({np.var(values):5.4})")

os.makedirs(os.path.join(args.result_filepath, 'eval'), exist_ok=True)
pickle.dump(og, open(os.path.join(args.result_filepath, 'eval', 'original_gospa.p'), "wb"))
pickle.dump(pg, open(os.path.join(args.result_filepath, 'eval', 'prob_gospa.p'), "wb"))
pickle.dump(d, open(os.path.join(args.result_filepath, 'eval', 'detr.p'), "wb"))
