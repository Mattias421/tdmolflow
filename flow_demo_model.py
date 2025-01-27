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

model_path = Path('./models_pretrained/cfm_dev/')
device = 'cuda'
sampler_class = 'JumpSampler'

sampler_kwargs = {
    'dt': 0.001,
    'corrector_steps': 5,
    'corrector_snr': 0.3,
    'corrector_start_time': 0.1,
    'corrector_finish_time': 0.003,
    'do_conditioning': False,
    'condition_type': 'sweep',
    'condition_sweep_idx': 0,
    'condition_sweep_path': None,
    'guidance_weight': 1.0,
    'do_jump_corrector': False,
    'sample_near_atom': True,
    'dt_schedule': 'C',
    'dt_schedule_h': 0.05,
    'dt_schedule_l': 0.001,
    'dt_schedule_tc': 0.5,
    'no_noise_final_step': True,
}

with open(model_path.joinpath('training_options.json'), "r") as stream:
    c = dnnlib.util.EasyDict(json.load(stream))

def convert_inner_dicts_to_easydicts(input_dict):
    for key in input_dict.keys():
        if type(input_dict[key]) == dict:
            input_dict[key] = convert_inner_dicts_to_easydicts(input_dict[key])
    input_dict = dnnlib.util.EasyDict(input_dict)
    return input_dict

c = convert_inner_dicts_to_easydicts(c)

dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs, train_or_valid='valid')  # subclass of training.dataset.Dataset

structure = Structure(**c.structure_kwargs, dataset=dataset_obj)
net = torch.load(model_path.joinpath('state_dict_cfm_dev.pt'))["net"]
net = net.eval().requires_grad_(False).to(device)
# modify_network_pkl(net)

# Setup sampler
sampler_class_name = 'training.sampler.' + sampler_class
usable_sampler_kwargs = dnnlib.EasyDict(class_name=sampler_class_name)
for kwarg_name, _, _ in samplers_to_kwargs[sampler_class]:
    # new_kwarg_name = "_".join(kwarg_name.split("_")[1:])
    usable_sampler_kwargs[kwarg_name] = sampler_kwargs[kwarg_name]
sampler = dnnlib.util.construct_class_by_name(**usable_sampler_kwargs, structure=structure)

# infer the task from the dataset
dataset_class_name = c.dataset_kwargs['class_name'].split('training.dataset.')[1]
if dataset_class_name not in ['QM9Dataset']:
    raise ValueError('Unknown dataset: ', dataset_class_name)

del(c.loss_kwargs['class_name'])
loss = JumpLossFinalDim(**c.loss_kwargs, structure=structure)

batch_size = 8
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
known_dims = None

in_st_batch = st_batch
state_st_batch = StructuredDataBatch.create_copy(in_st_batch)
max_problem_dim = state_st_batch.gs.max_problem_dim

x0, y = state_st_batch.get_flat_lats_and_obs()

B = x0.shape[0]

xT = rnd.randn_like(x0)  # (initialize at N(0, I))

flow_target = xT - x0

state_st_batch.set_flat_lats(xT)

# start at dimension 1
num_dims = torch.ones((B,)).long()

state_st_batch.delete_dims(new_dims=num_dims)

state_st_batch.gs.adjust_st_batch(state_st_batch)

device = x0[0].device
ts = torch.ones((B,), device=device)
steps_since_added_dim = torch.inf * torch.ones((B,), device=device)

dt = 0.001
finish_at = dt / 2

nfe = 0
will_finish = False

pos_error_list = []
ohe_error_list = []
num_dims_list = []

debug_sampler = False
while debug_sampler:

    if (ts - dt).clamp(
        min=finish_at / 2
    ).max() < finish_at:
        will_finish = True

    xt = state_st_batch.get_flat_lats()

    def set_unfinished_lats(xt):
        state_st_batch.set_flat_lats(
            xt * (1 - is_finished)
            + state_st_batch.get_flat_lats() * is_finished
        )

    # implement corrector steps for after adding a dimension
    is_finished = (ts < finish_at).float().view(-1, 1)

    if loss.noise_schedule_name == "cfm_ode":
        beta_t = loss.noise_schedule.get_beta_t(ts)  # (B, problem_dim)
        beta_t = state_st_batch.convert_problem_dim_to_tensor_dim(
            beta_t
        )  # (B, tensor_dim)

        D_xt, rate_xt, mean_std, _, _ = net(
            state_st_batch,
            ts,
            nearest_atom=None,
            sample_nearest_atom=True,
            forward_rate=loss.forward_rate,
            predict="eps",
            rnd=rnd,
        )
        score = D_xt
        # score = flow_target  # test with ideal flow
        nfe += 1

        mask = state_st_batch.get_mask(
            B=B, include_obs=False, include_onehot_channels=True
        ).to(device)

        xt = xt + mask * dt * score

    # TODO this noise is just for corrector? is corrector used/useful?
    noise = rnd.randn_like(xt)
    noise_st_batch = StructuredDataBatch.create_copy(state_st_batch)
    noise_st_batch.set_flat_lats(noise)
    noise_st_batch.delete_dims(new_dims=num_dims)
    noise_st_batch.gs.adjust_st_batch(noise_st_batch)
    noise = noise_st_batch.get_flat_lats()

    xt = xt + mask * torch.sqrt(beta_t * dt) * noise

    set_unfinished_lats(xt)
    state_st_batch.gs.adjust_st_batch(state_st_batch)
    xt = state_st_batch.get_flat_lats()

    # jump bit
    rate_xt = rate_xt.squeeze(1)
    increase_mask = (
        rnd.rand((B,), device=device) < rate_xt * dt
    ) * (
        num_dims.to(device) < max_problem_dim
    )  # (B,)

    increase_mask = (
        1 - is_finished.view(-1)
    ).bool() * increase_mask  # don't increase dimension after we've finished

    next_dims_mask = state_st_batch.get_next_dim_added_mask(
        B=B, include_onehot_channels=True, include_obs=False
    ).to(device)
    mean = mean_std[0]
    std = torch.nn.functional.softplus(mean_std[1])
    new_values = next_dims_mask * (mean + rnd.randn_like(std) * std)
    xt[increase_mask, :] = (
        xt[increase_mask, :] * (1 - next_dims_mask[increase_mask, :])
        + new_values[increase_mask, :]
    )

    num_dims[increase_mask.to(num_dims.device)] = (
        num_dims[increase_mask.to(num_dims.device)] + 1
    )

    num_dims_list.append(num_dims[0].item())
    print(num_dims[0].item())

    state_st_batch.set_dims(num_dims)
    set_unfinished_lats(xt.detach())
    state_st_batch.delete_dims(num_dims)
    state_st_batch.gs.adjust_st_batch(state_st_batch)
    xt = state_st_batch.get_flat_lats().detach()
    # if increase_mask.sum() > 0:
    #     breakpoint()

    dt = dt
    ts -= dt
    ts = ts.clamp(
        min=finish_at / 2
    )  # don't make zero in case of numerical weirdness
    if (
        ts.max() < finish_at
    ):  # miss the last step, as it seems to improve RMSE...
        break

    # get oracle mol

    from training.diffusion_utils import VP_SDE, StepForwardRate, CFM_ODE
    st_batch = StructuredDataBatch(data, dims, structure.observed,
        structure.exist, dataset_obj.is_onehot, structure.graphical_structure
    )
    noise_schedule = CFM_ODE(max_problem_dim, sigma=0.01)
    forward_rate = StepForwardRate(max_problem_dim, rate_cut_t=0.1)
    st_batch.delete_dims(new_dims=num_dims)

    st_batch.gs.adjust_st_batch(st_batch)

    mean, std, noise = noise_schedule.get_p0t_stats(st_batch, ts.to(device))
    noise = torch.randn_like(mean)
    noise_st_batch = StructuredDataBatch.create_copy(st_batch)
    noise_st_batch.set_flat_lats(noise)
    noise_st_batch.delete_dims(new_dims=num_dims)
    noise_st_batch.gs.adjust_st_batch(noise_st_batch)
    noise = noise_st_batch.get_flat_lats()
    xt = mean + std * noise

    st_batch.set_flat_lats(xt)

# make sure all masks are still correct
    st_batch.delete_dims(new_dims=num_dims)
# adjust
    st_batch.gs.adjust_st_batch(st_batch)
    error = (st_batch.get_flat_lats() - state_st_batch.get_flat_lats()).square().mean()
    print(f"error {error}")
    print(f"ts {ts[0]}")
    print(f"jump percent {ts[0] % 0.1}")
    idx = 0
    num_atoms = state_st_batch.get_dims()[idx].item()
    positions = state_st_batch.tuple_batch[0][idx, :num_atoms, :].cpu().detach()
    positions_label = st_batch.tuple_batch[0][idx, :num_atoms, :].cpu().detach()
    pos_error = (positions - positions_label).square().mean()
    print(f"pos error {pos_error}")
    ohe = state_st_batch.tuple_batch[1][idx, :num_atoms, :].cpu().detach()
    ohe_label = st_batch.tuple_batch[1][idx, :num_atoms, :].cpu().detach()
    ohe_error = (ohe - ohe_label).square().mean()
    print(f"ohe error {ohe_error}")

    pos_error_list.append(pos_error)
    ohe_error_list.append(ohe_error)

    # state_st_batch = st_batch # teacher forcing

# plt.plot(pos_error_list)
# plt.title("teacher-forced performance")
# plt.ylabel("pos mse")
# plt.xlabel("ODE step")
# plt.show()
# plt.plot(ohe_error_list)
# plt.title("teacher-forced performance")
# plt.ylabel("ohe mse")
# plt.xlabel("ODE step")
# plt.show()
# plt.plot(num_dims_list)
# plt.title("teacher-forced performance")
# plt.ylabel("num_dims")
# plt.xlabel("ODE step")
# plt.show()
#
x0_st_batch = sampler.sample(net, in_st_batch, loss, rnd, known_dims=known_dims,
                                dataset_obj=dataset_obj)

print(dataset_obj.log_batch(in_st_batch, x0_st_batch, wandb_log=False))

num_to_plot = min(batch_size, 8)
for idx in range(num_to_plot):
    num_atoms = x0_st_batch.get_dims()[idx].item()
    positions = x0_st_batch.tuple_batch[0][idx, 0:num_atoms, :].cpu().detach()
    atom_types = torch.argmax(x0_st_batch.tuple_batch[1][idx, 0:num_atoms, :], dim=1).cpu().detach()
    plot_data3d(positions, atom_types, dataset_obj.dataset_info, spheres_3d=False)
    plt.show()
