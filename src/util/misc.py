from typing import Optional, List
import math
import os
import sys

import torch
from torch import Tensor

from util.load_config_files import load_yaml_into_dotdict, dotdict


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(meas.shape) for meas in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for meas, pad_meas, m in zip(tensor_list, tensor, mask):
            pad_meas[: meas.shape[0], : meas.shape[1],
                     : meas.shape[2]].copy_(meas)
            m[: meas.shape[1], :meas.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def save_checkpoint(folder, filename, model, optimizer, scheduler):
    print(f"[INFO] Saving checkpoint in {folder}/{filename}")
    if not os.path.isdir(folder):
        os.makedirs(folder)
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
                }, os.path.join(folder, filename))


def factor_int(n):
    """
    Given an integer n, compute a factorization into two integers such that they're close as possible to each other (like
    a square root). E.g. factor_int(16)=(4,4), but factor_int(15)=(3,5).
    """
    nsqrt = math.ceil(math.sqrt(n))
    solution = False
    val = nsqrt
    while not solution:
        val2 = int(n/val)
        if val2 * val == float(n):
            solution = True
        else:
            val-=1
    return val, val2


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def super_load(experiment_path, checkpoint_filename=None, src_path_suffix=None, verbose=False):
    if src_path_suffix is None:
        src_path = os.path.join(experiment_path, 'code_used', 'MT3', 'src')
    else:
        src_path = os.path.join(experiment_path, src_path_suffix)
    params_path = os.path.join(experiment_path, 'code_used')
    checkpoints_path = os.path.join(experiment_path, 'checkpoints')

    # Load hyperparam files
    params = load_yaml_into_dotdict(os.path.join(params_path, 'model_params.yaml'))
    params.update(load_yaml_into_dotdict(os.path.join(params_path, 'task_params.yaml')))

    if params.training.device == 'auto':
        params.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Temporarily change from where python should import things
    old_paths = sys.path
    sys.path = [src_path]

    # Load the correct model definition from the src/ folder inside the experiment
    from modules.models.mt3.mt3 import MOTT

    # Create the model
    model = MOTT(params)
    model.to(torch.device(params.training.device))

    # Load the latest (or user-provided) checkpoint
    if checkpoint_filename is None:
        checkpoints = os.listdir(checkpoints_path)
        checkpoints.sort(key=lambda x: int(x.split('_')[-1]))
        weight_filename = os.path.join(checkpoints_path, checkpoints[-1])
    else:
        weight_filename = os.path.join(checkpoints_path, checkpoint_filename)
    if verbose:
        print(f'[INFO]: Loading weights {weight_filename}')
    t = torch.load(weight_filename, map_location=torch.device(params.training.device))
    model.load_state_dict(t['model_state_dict'])

    model.eval()

    # Restore sys.path contents
    sys.path = old_paths

    return model, params


@torch.no_grad()
def split_batch(batch, unique_ids, params):
    bs = batch.tensors.shape[0]
    batch = batch.tensors
    first_batch = []
    first_ids = []
    second_batch = []
    second_ids = []

    mapped_time_idx = batch[:,:,-1] / params.data_generation.dt

    
    for i in range(bs):
        # Take out all measurements that are in the first batch that are not padded
        first_batch_idx = mapped_time_idx[i] < params.data_generation.n_timesteps
        first_batch_idx = torch.logical_and(first_batch_idx, unique_ids[i]!=-2)

        second_batch_idx = 1 <= mapped_time_idx[i]
        second_batch_idx = torch.logical_and(second_batch_idx, unique_ids[i]!=-2)

        first_batch.append(batch[i][first_batch_idx])
        first_ids.append(unique_ids[i][first_batch_idx])

        second_batch.append(batch[i][second_batch_idx])
        second_ids.append(unique_ids[i][second_batch_idx])

        # Shift timestep
        second_batch[i][:,-1] = second_batch[i][:,-1] - params.data_generation.dt
    
    
    first, first_ids = pad_and_nest(first_batch, first_ids)
    second, second_ids = pad_and_nest(second_batch, second_ids)
                   
    return first, second, first_ids, second_ids

def pad_and_nest(batch, ids):
    max_len = max(list(map(len, batch)))
    batch, mask, ids = pad_to_batch_max(batch, ids, max_len)
    nested = NestedTensor(batch, mask.bool())

    return nested, ids

def pad_to_batch_max(batch, unique_ids, max_len):
    batch_size = len(batch)
    dev = batch[0].device
    d_meas = batch[0].shape[1]
    training_data_padded = torch.zeros((batch_size, max_len, d_meas), device=dev)
    mask = torch.ones((batch_size, max_len), device=dev)
    ids = -2 * torch.ones((batch_size, max_len), device=dev)
    for i, ex in enumerate(batch):
        training_data_padded[i,:len(ex),:] = ex
        mask[i,:len(ex)] = 0
        ids[i,:len(ex)] = unique_ids[i]

    return training_data_padded, mask, ids

def extract_batch(batch,unique_ids, lower_time_idx, upper_time_idx, dt,  batch_id=0):
    bt = batch.tensors.clone().detach()
    bm = batch.mask.clone().detach()
    u = unique_ids.clone().detach()
    b = NestedTensor(bt,bm)
    times = torch.round(b.tensors[batch_id,:,-1] / dt)
    idx = torch.logical_and(lower_time_idx <= times, times < upper_time_idx)
    b.tensors = b.tensors[batch_id, idx].unsqueeze(0)
    b.tensors[:,:,-1] = b.tensors[:,:,-1] - lower_time_idx*dt
    b.mask = batch.mask[batch_id, idx].unsqueeze(0)
    u = u[:,idx]

    return b, u