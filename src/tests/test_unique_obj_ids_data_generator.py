import numpy as np
import matplotlib.pyplot as plt
import torch

from src.data_generation.data_generator import MotDataGenerator
from src.training import get_params, convert_to_dot_dict


params = get_params()
params = convert_to_dot_dict(params)
data_gen_params = params.data_generation_params
data_gen_params.sigma_y = 0.0001

data_gen = MotDataGenerator(data_gen_params)
seed = 1

np.random.seed(seed)

for i in range(50):
    data_gen.step()
data_gen.finish()

# Visually inspect if object ids reasonably match each trajectory (and don't repeat ever)
plt.scatter(data_gen.measurements[:, 0], data_gen.measurements[:, 1], c=data_gen.unique_ids, cmap='tab20b')
plt.show()