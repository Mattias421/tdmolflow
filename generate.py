"""Generate molecules using a pretrained model."""

import os
import pickle
import time
import json

import numpy as np
import torch
from training.dataset.qm9 import plot_data3d
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

import dnnlib
from torch_utils import distributed as dist
from training.sampler import StackedRandomGenerator, samplers_to_kwargs
from training.structure import Structure, StructuredDataBatch
from training.loss import JumpLossFinalDim

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

def convert_inner_dicts_to_easydicts(input_dict):
    for key in input_dict.keys():
        if type(input_dict[key]) == dict:
            input_dict[key] = convert_inner_dicts_to_easydicts(input_dict[key])
    input_dict = dnnlib.util.EasyDict(input_dict)
    return input_dict

def generate_molecules(
    model_folder,
    device=torch.device("cuda"),
    num_molecules=10000,
    plot_data=False,
):
    """Generates a specified number of molecules using a pretrained model."""
    
    run_dir = f"{model_folder}/generated_mols"
    os.makedirs(run_dir, exist_ok=True)

    # Initialize.
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False

    if device != "cpu":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            False
        )
        
    # Load training options from JSON.
    training_options_path = f"{model_folder}/training_options.json"
    dist.print0(f"Loading training options from {training_options_path}...")

    with open(training_options_path, "r") as stream:
        c = dnnlib.util.EasyDict(json.load(stream))

    c = convert_inner_dicts_to_easydicts(c)

    dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs, train_or_valid='valid')

    structure = Structure(**c.structure_kwargs, dataset=dataset_obj)

    dist.print0("Constructing network...")
    net_state = torch.load(f'{model_folder}/state_dict.pt')
    if "net" in net_state.keys():
        net = net_state["net"]
    else:
        net = dnnlib.util.construct_class_by_name(**c.network_kwargs, structure=structure)
        net.load_state_dict(net_state)
    net.eval().requires_grad_(False).to(device)  # Set to eval mode

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

    generated_molecules = []
    start_time = time.time()
    batch_size = 50  # Adjust batch size as needed
    num_batches = (num_molecules + batch_size - 1) // batch_size

    dist.print0(f"Generating {num_molecules} molecules in {num_batches} batches...")

    for i in tqdm(range(num_batches), desc="Generating Molecules"):
        with torch.no_grad():
            rnd = StackedRandomGenerator(
                device, seeds=np.arange(i * batch_size, (i + 1) * batch_size)
            )
            indices = rnd.randint(
                len(dataset_obj), size=[batch_size, 1], device=device
            ).squeeze(1)
            unstacked_data = [
                dataset_obj.__getitem__(i.item(), will_augment=False)
                for i in indices
            ]
            unstacked_data_no_dims = [d[1:] for d in unstacked_data]
            dims = torch.tensor([d[0] for d in unstacked_data])
            data = tuple(
                torch.stack(d).to(device) for d in zip(*unstacked_data_no_dims)
            )
            
            st_batch = StructuredDataBatch(data, dims, structure.observed,
                structure.exist, dataset_obj.is_onehot, structure.graphical_structure
            )

            known_dims = None
            x0_st_batch = sampler.sample(net, st_batch, loss, rnd, known_dims=known_dims, dataset_obj=dataset_obj)
            molecules = {
              "x": x0_st_batch.tuple_batch[0],
              "one_hot": x0_st_batch.tuple_batch[1],
            }
            node_mask = torch.zeros(batch_size, dataset_obj.graphical_structure.max_problem_dim)
            for j in range(batch_size):
              node_mask[j, 0 : dims[j]] = 1
            node_mask = node_mask.unsqueeze(2).to(device)

            molecules["node_mask"] = node_mask


            if plot_data:
                idx = 0
                for idx in range(batch_size):
                    num_atoms = x0_st_batch.get_dims()[idx].item()
                    positions = x0_st_batch.tuple_batch[0][idx, 0:num_atoms, :].cpu().detach()
                    atom_types = torch.argmax(x0_st_batch.tuple_batch[1][idx, 0:num_atoms, :], dim=1).cpu().detach()
                    plot_data3d(positions, atom_types, dataset_obj.dataset_info, spheres_3d=False)
                    plt.show()
            
            with open(f'{run_dir}/batch_{i}.pkl', 'wb') as handle:
              pickle.dump(molecules, handle, protocol=pickle.HIGHEST_PROTOCOL)



    end_time = time.time()
    dist.print0(
        f"Generated {len(generated_molecules)} molecules in {end_time - start_time:.2f} seconds."
    )

    # Save the generated molecules to a file.
    output_file = os.path.join(run_dir, "generated_molecules.txt")
    with open(output_file, "w") as f:
        for molecule in generated_molecules:
            f.write(molecule + "\n")
    dist.print0(f"Generated molecules saved to {output_file}")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_folder", type=str,
        default="models_pretrained/cfm_dev",
                            help="Path to the pretrained model folder.")
    parser.add_argument("--num_molecules", type=int, default=10000,
                            help="Number of molecules to generate.")
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()

    # CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generate_molecules(
        args.model_folder,
        device=device,
        num_molecules=args.num_molecules,
        plot_data=args.plot,
    )
