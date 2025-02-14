"""Generate molecules using a pretrained model."""

import os
import pickle
import time
import json
import copy

import numpy as np
import torch
from tqdm import tqdm
import argparse

import dnnlib
from torch_utils import distributed as dist
from training.sampler import StackedRandomGenerator
from training.structure import Structure, StructuredDataBatch


def generate_molecules(
    model_folder,
    resume_pkl=None,  # Start from the given network snapshot, None = random initialization.
    device=torch.device("cuda"),
    num_molecules=10000,
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
    with open(training_options_path, "r") as f:
        training_options = json.load(f)

    # Extract kwargs from training options.
    dataset_kwargs = training_options["dataset_kwargs"]
    data_loader_kwargs = training_options["data_loader_kwargs"]
    network_kwargs = training_options["network_kwargs"]
    loss_kwargs = training_options["loss_kwargs"]
    sampler_kwargs = training_options["sampler_kwargs"]
    structure_kwargs = training_options["structure_kwargs"]

    # Load dataset (needed for structure and other dataset-specific info).
    dist.print0("Loading dataset...")
    train_dataset_kwargs = copy.deepcopy(dataset_kwargs)
    train_dataset_kwargs["train_or_valid"] = "train"  # Or "valid" if you prefer
    dataset_obj = dnnlib.util.construct_class_by_name(
        **train_dataset_kwargs
    )  # subclass of training.dataset.Dataset

    # Construct objects describing problem structure.
    structure = Structure(**structure_kwargs, dataset=dataset_obj)

    # Construct network.
    dist.print0("Constructing network...")
    net = torch.load(f'{model_folder}/state_dict.pt')["net"]
    net.eval().requires_grad_(False).to(device)  # Set to eval mode

    # Setup sampler
    sampler = dnnlib.util.construct_class_by_name(
        **sampler_kwargs, structure=structure
    )

    loss_fn = dnnlib.util.construct_class_by_name(
        **loss_kwargs, structure=structure
    )  # training.loss.(VP|VE|EDM)Loss

    generated_molecules = []
    start_time = time.time()
    batch_size = data_loader_kwargs.get("batch_size", 32)  # Adjust batch size as needed
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
            
            st_batch = StructuredDataBatch(
                data,
                dims,
                structure_kwargs["observed"],
                structure_kwargs["exist"],
                dataset_obj.is_onehot,
                structure.graphical_structure,
            )

            x0_st_batch = sampler.sample(net, st_batch, loss_fn, rnd, dataset_obj=dataset_obj)
            molecules = {
              "x": x0_st_batch.tuple_batch[0],
              "one_hot": x0_st_batch.tuple_batch[1],
            }
            node_mask = torch.zeros(batch_size, dataset_obj.graphical_structure.max_problem_dim)
            for j in range(batch_size):
              node_mask[j, 0 : dims[j]] = 1
            node_mask = node_mask.unsqueeze(2).to(device)

            molecules["node_mask"] = node_mask

            breakpoint()
            
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
    args = parser.parse_args()

    # CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generate_molecules(
        args.model_folder,
        device=device,
        num_molecules=args.num_molecules,
    )
