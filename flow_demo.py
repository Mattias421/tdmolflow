import tqdm
import json
import pickle
import numpy as np
import torch
import dnnlib
from pathlib import Path
from torch_utils import distributed as dist
from torch_utils.misc import modify_network_pkl
from training.sampler import StackedRandomGenerator, samplers_to_kwargs
from training.structure import Structure, StructuredDataBatch
from training.networks.egnn import EGNNMultiHeadJump
from training.loss import JumpLossFinalDim
import matplotlib.pyplot as plt
import datetime
import yaml
from training.dataset.qm9 import plot_data3d
from training.dataset import datasets_to_kwargs
import time
import os
from training.diffusion_utils import VP_SDE, StepForwardRate, CFM_ODE

device = 'cuda'

def convert_inner_dicts_to_easydicts(input_dict):
    for key in input_dict.keys():
        if type(input_dict[key]) == dict:
            input_dict[key] = convert_inner_dicts_to_easydicts(input_dict[key])
    input_dict = dnnlib.util.EasyDict(input_dict)
    return input_dict
model_path = "./models_pretrained/unconditional/"
model_path = Path(model_path)
with open(model_path.joinpath('training_options.json'), "r") as stream:
    c = dnnlib.util.EasyDict(json.load(stream))
c = convert_inner_dicts_to_easydicts(c)

dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs, train_or_valid='valid')  # subclass of training.dataset.Dataset

structure = Structure(**c.structure_kwargs, dataset=dataset_obj)

batch_size = 3
seeds = torch.arange(batch_size)
rnd = StackedRandomGenerator(device, seeds)
indices = rnd.randint(len(dataset_obj), size=[batch_size, 1], device=device)
unstacked_data = [dataset_obj.__getitem__(i.item(), will_augment=False) for i in indices]
unstacked_data_no_dims = [d[1:] for d in unstacked_data]
dims = torch.tensor([d[0] for d in unstacked_data])
data = tuple(torch.stack([datum[t] for datum in unstacked_data_no_dims]).to(device) for t in range(len(unstacked_data_no_dims[0])))
st_batch = StructuredDataBatch(data, dims, structure.observed,
    structure.exist, dataset_obj.is_onehot, structure.graphical_structure
)
ref_st_batch = st_batch
idx = 0
for idx in range(batch_size):
    num_atoms = st_batch.get_dims()[idx].item()
    positions = st_batch.tuple_batch[0][idx, 0:num_atoms, :].cpu().detach()
    atom_types = torch.argmax(st_batch.tuple_batch[1][idx, 0:num_atoms, :], dim=1).cpu().detach()
    plot_data3d(positions, atom_types, dataset_obj.dataset_info, spheres_3d=False)
    plt.savefig(f"./flow_demo/ref_{idx}.png")
    plt.close()


max_problem_dim = structure.graphical_structure.max_problem_dim
vp_sde_beta_min = c["loss_kwargs"]["vp_sde_beta_min"]
vp_sde_beta_max = c["loss_kwargs"]["vp_sde_beta_max"]


noise_schedule = VP_SDE(max_problem_dim, vp_sde_beta_min, vp_sde_beta_max)
noise_schedule_cfm = CFM_ODE(max_problem_dim, sigma=0.1)
forward_rate = StepForwardRate(max_problem_dim, rate_cut_t=0.1)

def sample_diffusion(t, noise_schedule):
    st_batch = StructuredDataBatch(data, dims, structure.observed,
    structure.exist, dataset_obj.is_onehot, structure.graphical_structure
)
    ts = torch.full((batch_size,), t)
    dims_xt = forward_rate.get_dims_at_t(
        start_dims=st_batch.get_dims(), ts=ts
    ).int()  # (B,)
    st_batch.delete_dims(new_dims=dims_xt)

    st_batch.gs.adjust_st_batch(st_batch)

    mean, std = noise_schedule.get_p0t_stats(st_batch, ts.to(device))
    noise = torch.randn_like(mean)
    noise_st_batch = StructuredDataBatch.create_copy(st_batch)
    noise_st_batch.set_flat_lats(noise)
    noise_st_batch.delete_dims(new_dims=dims_xt)
    noise_st_batch.gs.adjust_st_batch(noise_st_batch)
    noise = noise_st_batch.get_flat_lats()
    xt = mean + std * noise

    st_batch.set_flat_lats(xt)

# make sure all masks are still correct
    st_batch.delete_dims(new_dims=dims_xt)
# adjust
    st_batch.gs.adjust_st_batch(st_batch)

    idx = 0
    num_atoms = st_batch.get_dims()[idx].item()
    positions = st_batch.tuple_batch[0][idx, 0:num_atoms, :].cpu().detach()
    atom_types = torch.argmax(st_batch.tuple_batch[1][idx, 0:num_atoms, :], dim=1).cpu().detach()
    plot_data3d(positions, atom_types, dataset_obj.dataset_info, spheres_3d=False)
    plt.savefig(f"./flow_demo/diff_{int(t*10)}.png")
    plt.close()

    return st_batch.get_flat_lats()

def sample_flow(t, noise_schedule_diff, noise_schedule_cfm):
    st_batch = StructuredDataBatch(data, dims, structure.observed,
    structure.exist, dataset_obj.is_onehot, structure.graphical_structure
)
    ts = torch.full((batch_size,), t)
    dims_xt = forward_rate.get_dims_at_t(
        start_dims=st_batch.get_dims(), ts=ts
    ).int()  # (B,)
    st_batch.delete_dims(new_dims=dims_xt)

    st_batch.gs.adjust_st_batch(st_batch)

    mean, std = noise_schedule_diff.get_p0t_stats(st_batch, ts.to(device))
    noise = torch.randn_like(mean)
    noise_st_batch = StructuredDataBatch.create_copy(st_batch)
    noise_st_batch.set_flat_lats(noise)
    noise_st_batch.delete_dims(new_dims=dims_xt)
    noise_st_batch.gs.adjust_st_batch(noise_st_batch)
    noise = noise_st_batch.get_flat_lats()
    xt = mean + std * noise

    st_batch.set_flat_lats(xt)

# make sure all masks are still correct
    st_batch.delete_dims(new_dims=dims_xt)
# adjust
    st_batch.gs.adjust_st_batch(st_batch)

    for idx in range(batch_size):
        num_atoms = st_batch.get_dims()[idx].item()
        positions = st_batch.tuple_batch[0][idx, 0:num_atoms, :].cpu().detach()
        atom_types = torch.argmax(st_batch.tuple_batch[1][idx, 0:num_atoms, :], dim=1).cpu().detach()
        plot_data3d(positions, atom_types, dataset_obj.dataset_info, spheres_3d=False)
        plt.savefig(f"./flow_demo/diff_{idx}_{int(t*100)}.png")
        plt.close()
    diff_st_batch = StructuredDataBatch.create_copy(st_batch)

    st_batch = StructuredDataBatch(data, dims, structure.observed,
    structure.exist, dataset_obj.is_onehot, structure.graphical_structure
)
    st_batch.delete_dims(new_dims=dims_xt)

    st_batch.gs.adjust_st_batch(st_batch)

    mean, std, noise = noise_schedule_cfm.get_p0t_stats(st_batch, ts.to(device))
    noise_st_batch = StructuredDataBatch.create_copy(st_batch)
    noise_st_batch.set_flat_lats(noise)
    noise_st_batch.delete_dims(new_dims=dims_xt)
    noise_st_batch.gs.adjust_st_batch(noise_st_batch)
    noise = noise_st_batch.get_flat_lats()
    xt = mean + std * noise

    st_batch.set_flat_lats(xt)

# make sure all masks are still correct
    st_batch.delete_dims(new_dims=dims_xt)
# adjust
    st_batch.gs.adjust_st_batch(st_batch)

    for idx in range(batch_size):
        num_atoms = st_batch.get_dims()[idx].item()
        positions = st_batch.tuple_batch[0][idx, 0:num_atoms, :].cpu().detach()
        atom_types = torch.argmax(st_batch.tuple_batch[1][idx, 0:num_atoms, :], dim=1).cpu().detach()
        plot_data3d(positions, atom_types, dataset_obj.dataset_info, spheres_3d=False)
        plt.savefig(f"./flow_demo/cfm_{idx}_{int(t*100)}.png")
        plt.close()
    cfm_st_batch = StructuredDataBatch.create_copy(st_batch)
    return diff_st_batch.get_flat_lats(), cfm_st_batch.get_flat_lats()

loss_diff = []
loss_cfm = []
ref = ref_st_batch.get_flat_lats()
for t in range(101):
    print(t)
    xt_diff, xt_cfm = sample_flow(1 - t/100, noise_schedule, noise_schedule_cfm)
    loss_diff.append(torch.norm(ref - xt_diff).item())
    loss_cfm.append(torch.norm(ref - xt_cfm).item())

plt.plot(loss_diff, label="diff")
plt.plot(loss_cfm, label="cfm")
plt.legend()
plt.savefig("./flow_demo/loss.png")
