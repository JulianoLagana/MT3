import multiprocessing

import numpy as np
from numpy.random import SeedSequence, default_rng
import torch
from torch import Tensor

from data_generation.mot_data_generation import MotDataGenerator
from util.misc import NestedTensor


class DataGenerator:
    def __init__(self, params):
        self.params = params
        assert 0 <= params.data_generation.n_prediction_lag <= params.data_generation.n_timesteps, "Prediction lag has to be smaller than the total number of time-steps."
        self.device = params.training.device
        self.n_timesteps = params.data_generation.n_timesteps

        self.pool = multiprocessing.Pool()

        # Create `batch_size` data generators, each with its own independent (to a high probability) RNG
        ss = SeedSequence(params.data_generation.seed)
        rngs = [default_rng(s) for s in ss.spawn(params.training.batch_size)]
        self.datagens = [MotDataGenerator(params, rng=rng) for rng in rngs]

    def get_batch(self):
        results = self.pool.starmap(get_single_training_example, zip(self.datagens, [self.n_timesteps]*len(self.datagens)))

        # Unpack results
        training_data, labels, unique_ids, trajectories, new_rngs = tuple(zip(*results))
        labels = [Tensor(l).to(torch.device(self.device)) for l in labels]
        trajectories = list(trajectories)
        unique_ids = [list(u) for u in unique_ids]

        # Update the RNGs of all the datagens for next call
        for datagen, new_rng in zip(self.datagens, new_rngs):
            datagen.rng = new_rng

        # Pad training data
        max_len = max(list(map(len, training_data)))
        training_data, mask = pad_to_batch_max(training_data, max_len)

        # Pad unique ids
        for i in range(len(unique_ids)):
            unique_id = unique_ids[i]
            n_items_to_add = max_len - len(unique_id)
            unique_ids[i] = np.concatenate([unique_id, [-2] * n_items_to_add])[None, :]
        unique_ids = np.concatenate(unique_ids)

        training_nested_tensor = NestedTensor(Tensor(training_data).to(torch.device(self.device)),
                                              Tensor(mask).bool().to(torch.device(self.device)))
        unique_ids = Tensor(unique_ids).to(self.device)

        return training_nested_tensor, labels, unique_ids, trajectories


def pad_to_batch_max(training_data, max_len):
    batch_size = len(training_data)
    d_meas = training_data[0].shape[1]
    training_data_padded = np.zeros((batch_size, max_len, d_meas))
    mask = np.ones((batch_size, max_len))
    for i, ex in enumerate(training_data):
        training_data_padded[i,:len(ex),:] = ex
        mask[i,:len(ex)] = 0

    return training_data_padded, mask


def get_single_training_example(data_generator, n_timesteps):
    """Generates a single training example

    Returns:
        training_data   : A single training example
        true_data       : Ground truth for example
    """

    data_generator.reset()
    label_data = []

    while len(label_data) == 0 or len(data_generator.measurements) == 0:

        # Generate n_timesteps of data, from scratch
        data_generator.reset()
        for i in range(n_timesteps - 1):
            add_new_objects_flag = i < n_timesteps -3  # don't add new objects in the last two timesteps of generation, for cleaner training labels
            data_generator.step(add_new_objects=add_new_objects_flag)
        data_generator.finish()

        # -1 is applied because we count t=0 as one time-step
        for traj_id in data_generator.trajectories:
            traj = data_generator.trajectories[traj_id]
            if round(traj[-1][-1] / data_generator.delta_t) == n_timesteps - 1: #last state of trajectory, time
                pos = traj[-1][:data_generator.dim].copy()
                label_data.append(pos)

    training_data = np.array(data_generator.measurements.copy())
    unique_measurement_ids = data_generator.unique_ids.copy()
    new_rng = data_generator.rng

    return training_data, np.array(label_data), unique_measurement_ids, data_generator.trajectories.copy(), new_rng


