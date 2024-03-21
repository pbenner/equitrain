###########################################################################################
# Training utils
# Authors: David Kovacs, Ilyes Batatia
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import dataclasses
import logging
import ast
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed
from prettytable import PrettyTable

from equitrain.mace.data import Configurations, load_from_xyz, random_train_valid_split, test_config_types, compute_average_E0s

@dataclasses.dataclass
class SubsetCollection:
    train: Configurations
    valid: Configurations
    tests: List[Tuple[str, Configurations]]


def get_dataset_from_xyz(
    train_path: str,
    valid_path: str,
    valid_fraction: float,
    config_type_weights: Dict,
    test_path: str = None,
    seed: int = 1234,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipoles",
    charges_key: str = "charges",
) -> Tuple[SubsetCollection, Optional[Dict[int, float]]]:
    """Load training and test dataset from xyz file"""
    atomic_energies_dict, all_train_configs = load_from_xyz(
        file_path=train_path,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipole_key=dipole_key,
        charges_key=charges_key,
        extract_atomic_energies=True,
    )
    logging.info(
        f"Loaded {len(all_train_configs)} training configurations from '{train_path}'"
    )
    if valid_path is not None:
        _, valid_configs = load_from_xyz(
            file_path=valid_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
            virials_key=virials_key,
            dipole_key=dipole_key,
            charges_key=charges_key,
            extract_atomic_energies=False,
        )
        logging.info(
            f"Loaded {len(valid_configs)} validation configurations from '{valid_path}'"
        )
        train_configs = all_train_configs
    else:
        logging.info(
            "Using random %s%% of training set for validation", 100 * valid_fraction
        )
        train_configs, valid_configs = random_train_valid_split(
            all_train_configs, valid_fraction, seed
        )

    test_configs = []
    if test_path is not None:
        _, all_test_configs = load_from_xyz(
            file_path=test_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            dipole_key=dipole_key,
            charges_key=charges_key,
            extract_atomic_energies=False,
        )
        # create list of tuples (config_type, list(Atoms))
        test_configs = test_config_types(all_test_configs)
        logging.info(
            f"Loaded {len(all_test_configs)} test configurations from '{test_path}'"
        )
    return (
        SubsetCollection(train=train_configs, valid=valid_configs, tests=test_configs),
        atomic_energies_dict,
    )

def get_config_type_weights(ct_weights):
    """
    Get config type weights from command line argument
    """
    try:
        config_type_weights = ast.literal_eval(ct_weights)
        assert isinstance(config_type_weights, dict)
    except Exception as e:  # pylint: disable=W0703
        logging.warning(
            f"Config type weights not specified correctly ({e}), using Default"
        )
        config_type_weights = {"Default": 1.0}
    return config_type_weights

def get_atomic_energies(E0s, train_collection, z_table)->dict:
    if E0s is not None:
        logging.info(
            "Atomic Energies not in training file, using command line argument E0s"
        )
        if E0s.lower() == "average":
            logging.info(
                "Computing average Atomic Energies using least squares regression"
            )
            # catch if colections.train not defined above
            try:
                assert train_collection is not None
                atomic_energies_dict = compute_average_E0s(
                    train_collection, z_table
                )
            except Exception as e:
                raise RuntimeError(
                    f"Could not compute average E0s if no training xyz given, error {e} occured"
                ) from e
        else:
            try:
                atomic_energies_dict = ast.literal_eval(E0s)
                assert isinstance(atomic_energies_dict, dict)
            except Exception as e:
                raise RuntimeError(
                    f"E0s specified invalidly, error {e} occured"
                ) from e
    else:
        raise RuntimeError(
            "E0s not found in training file and not specified in command line"
        )
    return atomic_energies_dict

def get_loss_fn(loss: str,
                energy_weight: float,
                forces_weight: float,
                stress_weight: float,
                virials_weight: float,
                dipole_weight: float,
                dipole_only: bool,
                compute_dipole: bool) -> torch.nn.Module:
    if loss == "weighted":
        loss_fn = modules.WeightedEnergyForcesLoss(
            energy_weight=energy_weight, forces_weight=forces_weight
        )
    elif loss == "forces_only":
        loss_fn = modules.WeightedForcesLoss(forces_weight=forces_weight)
    elif loss == "virials":
        loss_fn = modules.WeightedEnergyForcesVirialsLoss(
            energy_weight=energy_weight,
            forces_weight=forces_weight,
            virials_weight=virials_weight,
        )
    elif loss == "stress":
        loss_fn = modules.WeightedEnergyForcesStressLoss(
            energy_weight=energy_weight,
            forces_weight=forces_weight,
            stress_weight=stress_weight,
        )
    elif loss == "dipole":
        assert (
            dipole_only is True
        ), "dipole loss can only be used with AtomicDipolesMACE model"
        loss_fn = modules.DipoleSingleLoss(
            dipole_weight=dipole_weight,
        )
    elif loss == "energy_forces_dipole":
        assert dipole_only is False and compute_dipole is True
        loss_fn = modules.WeightedEnergyForcesDipoleLoss(
            energy_weight=energy_weight,
            forces_weight=forces_weight,
            dipole_weight=dipole_weight,
        )
    else:
        loss_fn = modules.EnergyForcesLoss(
            energy_weight=energy_weight, forces_weight=forces_weight
        )
    return loss_fn

def get_files_with_suffix(dir_path:str, suffix:str)-> List[str]:
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(suffix)]

def custom_key(key):
    """
    Helper function to sort the keys of the data loader dictionary
    to ensure that the training set, and validation set
    are evaluated first
    """
    if key == 'train':
        return (0, key)
    elif key == 'valid':
        return (1, key)
    else:
        return (2, key)
