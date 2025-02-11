import torch
import json
import dnnlib
import copy
import glob
import os
import pickle
from tqdm import tqdm
import argparse

from training.dataset.qm9 import analyze_stability_for_molecules

def evaluate_molecules(model_folder):

    training_options_path = f"{model_folder}/training_options.json"
    with open(training_options_path, "r") as f:
        training_options = json.load(f)

    dataset_kwargs = training_options["dataset_kwargs"]
    train_dataset_kwargs = copy.deepcopy(dataset_kwargs)
    train_dataset_kwargs["train_or_valid"] = "train"  # Or "valid" if you prefer
    dataset_obj = dnnlib.util.construct_class_by_name(
        **train_dataset_kwargs
    )  # subclass of training.dataset.Dataset

    mol_folder = f"{model_folder}generated_mols"
    num_batches = len(glob.glob(os.path.join(mol_folder, "*.pkl")))

    results_csv = open(f"{mol_folder}/_results.csv", "w")
    print("BATCH,ATM_STB,MOL_STB,VALID,UNIQ,NOVEL", file=results_csv)
    print(mol_folder)

    for i in tqdm(range(num_batches), desc="Generating Molecules"):
            
        with open(f'{mol_folder}/batch_{i}.pkl', 'rb') as handle:
            molecules = pickle.load(handle, encoding='unicode_escape')

        validity_dict, rdkit_tuple = analyze_stability_for_molecules(
        molecules, dataset_obj.dataset_info, dataloaders=dataset_obj.dataloaders
        )

        combined_dict = {}
        for key in validity_dict:
            combined_dict[key] = validity_dict[key]
        if rdkit_tuple is not None:
            combined_dict["Validity"] = rdkit_tuple[0][0]
            combined_dict["Uniqueness"] = rdkit_tuple[0][1]
            combined_dict["Novelty"] = rdkit_tuple[0][2]

        print(f"{i},{combined_dict['atm_stable']},{combined_dict['mol_stable']},{combined_dict['Validity']},{combined_dict['Uniqueness']},{combined_dict['Novelty']}", file=results_csv)

    results_csv.close()

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

    evaluate_molecules(
        args.model_folder,
    )
