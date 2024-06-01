import numpy as np
import torch

from pymatgen import Structure

from equitrain.model       import get_model
from equitrain.dataloaders import get_dataloader

# %%

def predict_structure(model: torch.nn.Module, struct: Structure):
    """Predict energy, forces, and stress of a structure"""
    atoms_to_graphs = AtomsToGraphs(
        max_neigh=200,
        radius=6,
        r_energy=True,
        r_forces=True,
        r_stress=True,
        r_distances=True,
        r_edges=True,
        r_fixed=True,
        r_pbc=False,
    )
    atoms = struct.to_ase_atoms()
    data = atoms_to_graphs.convert(atoms)
    data = data.to(device)

    return model(data)


def _predict(args):

    energy_pred = []
    forces_pred = [[], [], []]
    stress_pred = [[], [], [], [], [], [], [], [], []]

    data_loader, r_max = get_dataloader(args.predict_file, args)

    model = get_model(r_max, args)

    for step, data in enumerate(data_loader):

        e_pred, f_pred, s_pred = model(data)

        if e_pred is not None:
            energy_pred.extend([ e.item() for e in e_pred ])

        if f_pred is not None:

            for i in range(3):

                forces_pred[i].extend([ f[i].item() for f in f_pred ])

        if s_pred is not None:

            for i in range(3):

                for j in range(3):

                    k = i*3 + j

                    stress_pred[k].extend([ s[i][j].item() for s in s_pred ])

    return np.array(energy_pred), np.array(forces_pred), np.array(stress_pred)

def predict(args):

    if args.predict_file is None:
        raise ValueError("--predict-file is a required argument")
    if args.statistics_file is None:
        raise ValueError("--statistics-file is a required argument")
    if args.load_checkpoint_model is None:
        raise ValueError("--load-checkpoint-model is a required argument")

    return _predict(args)
